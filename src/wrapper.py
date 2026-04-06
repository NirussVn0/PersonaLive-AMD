from omegaconf import OmegaConf
import os
import torch
try:
    import torch_directml
except ImportError:
    pass
import numpy as np
from PIL import Image
import time
import gc
import cv2
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.pose_guider import PoseGuider
from src.models.motion_encoder.encoder import MotEncoder
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.liveportrait.motion_extractor import MotionExtractor
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from collections import deque
from threading import Lock, Thread
from torchvision import transforms as T
from einops import rearrange
from src.utils.util import draw_keypoints, get_boxes
import torch.nn.functional as F


class DMLAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        inner_dtype = query.dtype
        
        q = query.clone().to(torch.float32).contiguous()
        k = key.clone().to(torch.float32).transpose(-1, -2).contiguous()
        v = value.clone().contiguous()

        batch_size, n_tokens, dim = q.shape
        slice_size = 256 # Limits intermediate DML matrix size to ~1GB (Safe for 8GB VRAM cards)
        
        hidden_states = torch.zeros(batch_size, n_tokens, v.shape[-1], device=query.device, dtype=inner_dtype)

        for i in range(0, n_tokens, slice_size):
            end = min(n_tokens, i + slice_size)
            q_slice = q[:, i:end, :]
            
            # [B, slice, D] @ [B, D, M] -> [B, slice, M]
            attn_slice = torch.bmm(q_slice, k) * attn.scale
            
            if attention_mask is not None:
                mask_len = attention_mask.shape[-2]
                if mask_len == n_tokens:
                    mask_slice = attention_mask[..., i:end, :]
                else: 
                    mask_slice = attention_mask
                attn_slice = attn_slice + mask_slice.to(torch.float32).contiguous()
                
            probs_slice = attn_slice.softmax(dim=-1).to(inner_dtype).contiguous()
            
            # [B, slice, M] @ [B, M, D] -> [B, slice, D]
            out_slice = torch.bmm(probs_slice, v)
            hidden_states[:, i:end, :] = out_slice
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def map_device(device_or_str):
    return device_or_str if isinstance(device_or_str, torch.device) else torch.device(device_or_str)

class PersonaLive:
    def __init__(self, args, device=None):
        cfg = OmegaConf.load(args.config_path)
        if(device is None):
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = map_device(device)

        self.temporal_adaptive_step = cfg.temporal_adaptive_step
        self.temporal_window_size = cfg.temporal_window_size

        if cfg.dtype == "fp16":
            self.numpy_dtype = np.float16
            self.dtype = torch.float16
        elif cfg.dtype == "fp32":
            self.numpy_dtype = np.float32
            self.dtype = torch.float32

        infer_config = OmegaConf.load(cfg.inference_config)
        sched_kwargs = OmegaConf.to_container(
            infer_config.noise_scheduler_kwargs
        )

        self.num_inference_steps = cfg.num_inference_steps

        # initialize models
        self.pose_guider = PoseGuider().to(device=self.device, dtype=self.dtype)
        pose_guider_state_dict = torch.load(cfg.pose_guider_path, map_location="cpu")
        self.pose_guider.load_state_dict(pose_guider_state_dict)
        del pose_guider_state_dict

        self.motion_encoder = MotEncoder().to(dtype=self.dtype, device=self.device).eval()
        motion_encoder_state_dict = torch.load(cfg.motion_encoder_path, map_location="cpu")
        self.motion_encoder.load_state_dict(motion_encoder_state_dict)
        del motion_encoder_state_dict

        self.pose_encoder = MotionExtractor(num_kp=21).to(device=self.device, dtype=self.dtype).eval()
        pose_encoder_state_dict = torch.load(cfg.pose_encoder_path, map_location="cpu")
        self.pose_encoder.load_state_dict(pose_encoder_state_dict, strict=False)
        del pose_encoder_state_dict

        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=self.dtype, device=self.device)

        self.reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=self.dtype, device=self.device)
        reference_unet_state_dict = torch.load(cfg.reference_unet_weight_path, map_location="cpu")
        self.reference_unet.load_state_dict(reference_unet_state_dict)
        del reference_unet_state_dict

        self.denoising_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False
        )
        self.denoising_unet.load_state_dict(
            torch.load(
                cfg.temporal_module_path,
                map_location="cpu",
            ),
            strict=False,
        )

        is_dml = "privateuseone" in str(self.device)
        if is_dml:
            self.denoising_unet.set_attn_processor(DMLAttnProcessor())
            self.reference_unet.set_attn_processor(DMLAttnProcessor())

        self.reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            batch_size=cfg.batch_size,
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            batch_size=cfg.batch_size,
            fusion_blocks="full",
            cache_kv=True,
        )

        self.vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
            device=self.device, dtype=self.dtype
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path,
        ).to(device=self.device, dtype=self.dtype)

        # miscellaneous
        self.scheduler = DDIMScheduler(**sched_kwargs)
        self.timesteps = torch.tensor([999, 666, 333, 0], device=self.device).long()
        self.scheduler.set_step_length(333)
        
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(cfg.seed)

        self.batch_size = cfg.batch_size
        self.vae_scale_factor = 8
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=True)
        
        self.cfg = cfg
        self.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            pass # Attention slicing is intentionally disabled for DMLAttnProcessor compatibility
        except Exception as e:
            print("Failed to enable Attention Slicing:", e)
    
    def reset(self):
        self.first_frame = True
        self.motion_bank = None
        self.count = 0
        self.num_khf = 0
        self.latents_pile = deque([])
        self.pose_pile = deque([])
        self.motion_pile = deque([])
        self.reference_control_writer.clear()
        self.reference_control_reader.clear()



    def fast_resize(self, images, target_width, target_height) -> torch.Tensor:
        tgt_cond_tensor = F.interpolate(
            images,
            size=(target_width, target_height),
            mode="bilinear",
            align_corners=False,
        )
        return tgt_cond_tensor

    @torch.no_grad()
    def fuse_reference(self, ref_image): # pil input
        clip_image = self.clip_image_processor.preprocess(
            ref_image, return_tensors="pt"
        ).pixel_values
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=self.cfg.reference_image_height, width=self.cfg.reference_image_width
        )  # (bs, c, width, height)
        clip_image_embeds = self.image_encoder(
            clip_image.to(self.image_encoder.device, dtype=self.image_encoder.dtype)
        ).image_embeds
        self.encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        self.ref_image_tensor = ref_image_tensor.squeeze(0)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
        self.reference_unet(
            ref_image_latents.to(self.reference_unet.device),
            torch.zeros((self.batch_size,),dtype=self.dtype,device=self.reference_unet.device),
            encoder_hidden_states=self.encoder_hidden_states,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer)
        self.encoder_hidden_states = self.encoder_hidden_states.to(self.device)

        ref_cond_tensor = self.cond_image_processor.preprocess(
            ref_image, height=256, width=256
        ).to(device=self.device, dtype=self.pose_encoder.dtype)  # (1, c, h, w)
        self.ref_cond_tensor = ref_cond_tensor / 2 + 0.5 # to [0, 1]
        self.ref_image_latents = ref_image_latents

        padding_num = (self.temporal_adaptive_step - 1) * self.temporal_window_size
        init_latents = ref_image_latents.to("cpu").unsqueeze(2).repeat(1, 1, padding_num, 1, 1).to(self.device)
        noise = torch.randn_like(init_latents)
        init_timesteps = reversed(self.timesteps).repeat_interleave(self.temporal_window_size, dim=0)
        noisy_latents_first = self.scheduler.add_noise(init_latents, noise, init_timesteps[:padding_num])
        for i in range(self.temporal_adaptive_step-1):
            l = i * self.temporal_window_size
            r = (i+1) * self.temporal_window_size
            self.latents_pile.append(noisy_latents_first[:,:,l:r])
    
    def crop_face(self, image_pil, boxes):
        image = np.array(image_pil)

        left, top, right, bot = boxes

        face_patch = image[int(top) : int(bot), int(left) : int(right)]
        face_patch = Image.fromarray(face_patch).convert("RGB")
        return face_patch
    
    def crop_face_tensor(self, image_tensor, boxes):
        left, top, right, bot = boxes
        left, top, right, bottom = map(int, (left, top, right, bot))

        face_patch = image_tensor[:, top:bottom, left:right]
        face_patch = F.interpolate(
            face_patch.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        return face_patch
    
    def interpolate_tensors(self, a: torch.Tensor, b: torch.Tensor, num: int = 10) -> torch.Tensor:
        """
        在张量 a 和 b 之间线性插值。
        输入 shape: (B, 1, D1, D2, ...)
        输出 shape: (B, num, D1, D2, ...)
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

        B, _, *rest = a.shape
        # 插值系数 (num,) → reshape 成 (1, num, 1, 1, ...)
        alphas = torch.linspace(0, 1, num, device=a.device, dtype=a.dtype)
        view_shape = (1, num) + (1,) * len(rest)
        alphas = alphas.view(view_shape)  # (1, num, 1, 1, ...)

        # 插值 (B, num, D1, D2, ...)
        result = (1 - alphas) * a + alphas * b
        return result
    
    def calculate_dis(self, A, B, threshold=10.):
        """
        A: (b, f1, c1, c2)  bank
        B: (b, f2, c1, c2)  new data
        """

        A_flat = A.view(A.size(1), -1).clone()
        B_flat = B.view(B.size(1), -1).clone()

        A_f32 = A_flat.to(torch.float32)
        B_f32 = B_flat.to(torch.float32)
        diff = B_f32.unsqueeze(1) - A_f32.unsqueeze(0)
        dist = torch.norm(diff, p=2, dim=-1)

        min_dist, min_idx = dist.min(dim=1)  # (f2,)

        idx_to_add = torch.nonzero(min_dist[:1] > threshold, as_tuple=False).squeeze(1).tolist()

        if len(idx_to_add) > 0:  # 有需要添加的元素
            B_to_add = B[:, idx_to_add]  # (1, k, c1, c2)
            A_new = torch.cat([A, B_to_add], dim=1)  # (1, f1+k, c1, c2)
        else:
            A_new = A  # 没有需要添加的

        return idx_to_add, A_new, min_idx

    @torch.no_grad()
    def process_input(self, images):
        batch_size = self.batch_size
        device = self.device
        temporal_window_size = self.temporal_window_size
        temporal_adaptive_step = self.temporal_adaptive_step

        tgt_cond_tensor = self.fast_resize(images, 256, 256)
        tgt_cond_tensor = tgt_cond_tensor / 2 + 0.5

        if self.first_frame:
            mot_bbox_param, kps_ref, kps_frame1, kps_dri = self.pose_encoder.interpolate_kps_online(self.ref_cond_tensor, tgt_cond_tensor, num_interp=12+1)
            self.kps_ref = kps_ref
            self.kps_frame1 = kps_frame1
        else:
            mot_bbox_param, kps_dri = self.pose_encoder.get_kps(self.kps_ref, self.kps_frame1, tgt_cond_tensor)

        keypoints = draw_keypoints(mot_bbox_param, device=device)
        boxes = get_boxes(kps_dri)
        keypoints = rearrange(keypoints.to("cpu").unsqueeze(2), 'f c b h w -> b c f h w')
        keypoints = keypoints.to(device=device, dtype=self.pose_guider.dtype)

        if self.first_frame:
            ref_box = get_boxes(mot_bbox_param[:1])
            ref_face = self.crop_face_tensor(self.ref_image_tensor, ref_box[0])
            motion_face = [ref_face]
            for i, frame in enumerate(images):
                motion_face.append(self.crop_face_tensor(frame, boxes[i]))
            pose_cond_tensor = torch.cat(motion_face, dim=0).transpose(0, 1)
            pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
            # pose_cond_tensor = pose_cond_tensor.to(
            #     device=device, dtype=self.motion_encoder.dtype
            # )
            motion_hidden_states = self.motion_encoder(pose_cond_tensor)
            ref_motion = motion_hidden_states[:, :1]
            dri_motion = motion_hidden_states[:, 1:]

            init_motion_hidden_states = self.interpolate_tensors(ref_motion, dri_motion[:,:1], num=12+1)[:,:-1]
            for i in range(temporal_adaptive_step-1):
                l = i * temporal_window_size
                r = (i+1) * temporal_window_size
                self.motion_pile.append(init_motion_hidden_states[:,l:r])
            self.motion_pile.append(dri_motion)

            self.motion_bank = ref_motion
        else:
            motion_face = []
            for i, frame in enumerate(images):
                motion_face.append(self.crop_face_tensor(frame, boxes[i]))
            pose_cond_tensor = torch.cat(motion_face, dim=0).transpose(0, 1)
            pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
            motion_hidden_states = self.motion_encoder(pose_cond_tensor)
            self.motion_pile.append(motion_hidden_states)

        pose_fea = self.pose_guider(keypoints)
        if self.first_frame:
            for i in range(temporal_adaptive_step):
                l = i * temporal_window_size
                r = (i+1) * temporal_window_size
                self.pose_pile.append(pose_fea[:,:,l:r])
            self.first_frame = False
        else:
            self.pose_pile.append(pose_fea)

        latents = self.ref_image_latents.to("cpu").unsqueeze(2).repeat(1, 1, temporal_window_size, 1, 1).to(self.device)
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.timesteps[:1])
        self.latents_pile.append(latents)

        jump = 1
        motion_hidden_state = torch.cat(list(self.motion_pile), dim=1)
        pose_cond_fea=torch.cat(list(self.pose_pile), dim=2)

        idx_to_add = []
        if self.count > 8:
            idx_to_add, self.motion_bank, idx_his = self.calculate_dis(self.motion_bank, motion_hidden_state, threshold=17.)

        latents_model_input = torch.cat(list(self.latents_pile), dim=2)
        for j in range(jump):
            timesteps = reversed(self.timesteps[j::jump]).repeat_interleave(temporal_window_size, dim=0)
            timesteps = torch.stack([timesteps] * batch_size)#.to(device)
            timesteps = rearrange(timesteps, 'b f -> (b f)')
            noise_pred = self.denoising_unet(
                latents_model_input,
                timesteps,
                encoder_hidden_states=[self.encoder_hidden_states,
                                       motion_hidden_state],
                pose_cond_fea=pose_cond_fea,
                return_dict=False,
            )[0]

            clip_length = noise_pred.shape[2]
            mid_noise_pred = rearrange(noise_pred, 'b c f h w -> (b f) c h w')
            mid_latents = rearrange(latents_model_input, 'b c f h w -> (b f) c h w')
            latents_model_input, pred_original_sample = self.scheduler.step(
                mid_noise_pred, timesteps, mid_latents, generator=self.generator, return_dict=False
            )
            latents_model_input = rearrange(latents_model_input, '(b f) c h w -> b c f h w', f=clip_length)
            pred_original_sample = rearrange(pred_original_sample, '(b f) c h w -> b c f h w', f=clip_length)
            latents_model_input = torch.cat([
                pred_original_sample[:,:,:temporal_window_size],
                latents_model_input[:,:,temporal_window_size:]], dim=2)
            latents_model_input = latents_model_input.to(dtype=self.dtype)

        if len(idx_to_add) > 0 and self.num_khf < 3:
            self.reference_control_writer.clear()
            self.reference_unet(
                pred_original_sample[:,:,0].to(self.reference_unet.dtype),
                torch.zeros((batch_size,),dtype=self.dtype,device=self.reference_unet.device),
                encoder_hidden_states=self.encoder_hidden_states,
                return_dict=False,
            )
            self.reference_control_reader.update_hkf(self.reference_control_writer)
            print('add_keyframes')
            self.num_khf += 1
        
        
        for i in range(len(self.latents_pile)):
            self.latents_pile[i] = latents_model_input[:, :, i * temporal_adaptive_step : (i + 1) * temporal_adaptive_step, :, :]

        self.pose_pile.popleft()
        self.motion_pile.popleft()
        latents = self.latents_pile.popleft()
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "b c h w -> b h w c")
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().numpy()
        self.count += 1
        return video