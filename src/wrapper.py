from omegaconf import OmegaConf
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time
import gc
import cv2
import logging
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

logger = logging.getLogger(__name__)


def detect_hip_device() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    amd_identifiers = ("gfx", "radeon", "amd", "rdna", "navi")
    device_name = props.name.lower()
    return any(ident in device_name for ident in amd_identifiers)


IS_AMD_HIP = detect_hip_device()


def configure_hip_backend() -> None:
    if not IS_AMD_HIP:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault(
        "PYTORCH_HIP_ALLOC_CONF",
        "garbage_collection_threshold:0.9,max_split_size_mb:512",
    )
    logger.info("HIP backend configured for AMD RDNA2")


def flush_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HIPAttnProcessor:
    def __call__(
        self,
        attn: object,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=attn.scale,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def map_device(device_or_str: object) -> torch.device:
    if isinstance(device_or_str, torch.device):
        return device_or_str
    return torch.device(device_or_str)


class PersonaLive:
    def __init__(self, args: object, device: object = None):
        configure_hip_backend()

        cfg = OmegaConf.load(args.config_path)

        if device is None:
            self.device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
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

        self._load_models(cfg, infer_config)
        self._setup_attention_processors()
        self._setup_reference_control(cfg)
        self._load_vae_and_encoder(cfg)
        self._setup_scheduler(sched_kwargs, cfg)

        self.cfg = cfg
        self.reset()
        flush_vram()

    def _load_models(self, cfg: object, infer_config: object) -> None:
        self.pose_guider = PoseGuider().to(
            device=self.device, dtype=self.dtype
        )
        pose_guider_state_dict = torch.load(
            cfg.pose_guider_path, map_location="cpu"
        )
        self.pose_guider.load_state_dict(pose_guider_state_dict)
        del pose_guider_state_dict
        flush_vram()

        self.motion_encoder = (
            MotEncoder().to(dtype=self.dtype, device=self.device).eval()
        )
        motion_encoder_state_dict = torch.load(
            cfg.motion_encoder_path, map_location="cpu"
        )
        self.motion_encoder.load_state_dict(motion_encoder_state_dict)
        del motion_encoder_state_dict
        flush_vram()

        self.pose_encoder = (
            MotionExtractor(num_kp=21)
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        pose_encoder_state_dict = torch.load(
            cfg.pose_encoder_path, map_location="cpu"
        )
        self.pose_encoder.load_state_dict(
            pose_encoder_state_dict, strict=False
        )
        del pose_encoder_state_dict
        flush_vram()

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

        reference_unet_state_dict = torch.load(
            cfg.reference_unet_weight_path, map_location="cpu"
        )
        self.reference_unet.load_state_dict(reference_unet_state_dict)
        del reference_unet_state_dict
        flush_vram()

        self.denoising_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        self.denoising_unet.load_state_dict(
            torch.load(cfg.temporal_module_path, map_location="cpu"),
            strict=False,
        )
        flush_vram()

    def _setup_attention_processors(self) -> None:
        self.denoising_unet.set_attn_processor(HIPAttnProcessor())
        self.reference_unet.set_attn_processor(HIPAttnProcessor())

    def _setup_reference_control(self, cfg: object) -> None:
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

    def _load_vae_and_encoder(self, cfg: object) -> None:
        self.vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
            device=self.device, dtype=self.dtype
        )
        self.vae.enable_slicing()
        self.vae.enable_tiling()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path,
        ).to(device=self.device, dtype=self.dtype)
        flush_vram()

    def _setup_scheduler(self, sched_kwargs: dict, cfg: object) -> None:
        self.scheduler = DDIMScheduler(**sched_kwargs)
        self.timesteps = torch.tensor(
            [999, 666, 333, 0], device=self.device
        ).long()
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
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )

    def reset(self) -> None:
        self.first_frame = True
        self.motion_bank = None
        self.count = 0
        self.num_khf = 0
        self.latents_pile = deque([])
        self.pose_pile = deque([])
        self.motion_pile = deque([])
        self.reference_control_writer.clear()
        self.reference_control_reader.clear()

    def fast_resize(
        self,
        images: torch.Tensor,
        target_width: int,
        target_height: int,
    ) -> torch.Tensor:
        return F.interpolate(
            images,
            size=(target_width, target_height),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def fuse_reference(self, ref_image: Image.Image) -> None:
        clip_image = self.clip_image_processor.preprocess(
            ref_image, return_tensors="pt"
        ).pixel_values
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image,
            height=self.cfg.reference_image_height,
            width=self.cfg.reference_image_width,
        )
        clip_image_embeds = self.image_encoder(
            clip_image.to(
                self.image_encoder.device, dtype=self.image_encoder.dtype
            )
        ).image_embeds
        self.encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        self.ref_image_tensor = ref_image_tensor.squeeze(0)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215
        self.reference_unet(
            ref_image_latents.to(self.reference_unet.device),
            torch.zeros(
                (self.batch_size,),
                dtype=self.dtype,
                device=self.reference_unet.device,
            ),
            encoder_hidden_states=self.encoder_hidden_states,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer)
        self.encoder_hidden_states = self.encoder_hidden_states.to(self.device)

        ref_cond_tensor = self.cond_image_processor.preprocess(
            ref_image, height=256, width=256
        ).to(device=self.device, dtype=self.pose_encoder.dtype)
        self.ref_cond_tensor = ref_cond_tensor / 2 + 0.5
        self.ref_image_latents = ref_image_latents

        padding_num = (
            (self.temporal_adaptive_step - 1) * self.temporal_window_size
        )
        init_latents = (
            ref_image_latents.to("cpu")
            .unsqueeze(2)
            .repeat(1, 1, padding_num, 1, 1)
            .to(self.device)
        )
        noise = torch.randn_like(init_latents)
        init_timesteps = reversed(self.timesteps).repeat_interleave(
            self.temporal_window_size, dim=0
        )
        noisy_latents_first = self.scheduler.add_noise(
            init_latents, noise, init_timesteps[:padding_num]
        )
        for i in range(self.temporal_adaptive_step - 1):
            l = i * self.temporal_window_size
            r = (i + 1) * self.temporal_window_size
            self.latents_pile.append(noisy_latents_first[:, :, l:r])

    def crop_face(self, image_pil: Image.Image, boxes: tuple) -> Image.Image:
        image = np.array(image_pil)
        left, top, right, bot = boxes
        face_patch = image[int(top) : int(bot), int(left) : int(right)]
        return Image.fromarray(face_patch).convert("RGB")

    def crop_face_tensor(
        self, image_tensor: torch.Tensor, boxes: tuple
    ) -> torch.Tensor:
        left, top, right, bot = boxes
        left, top, right, bottom = map(int, (left, top, right, bot))
        face_patch = image_tensor[:, top:bottom, left:right]
        return F.interpolate(
            face_patch.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

    def interpolate_tensors(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        num: int = 10,
    ) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"
            )

        _B, _, *rest = a.shape
        alphas = torch.linspace(0, 1, num, device=a.device, dtype=a.dtype)
        view_shape = (1, num) + (1,) * len(rest)
        alphas = alphas.view(view_shape)
        return (1 - alphas) * a + alphas * b

    def calculate_dis(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        threshold: float = 10.0,
    ) -> tuple:
        A_flat = A.view(A.size(1), -1).clone()
        B_flat = B.view(B.size(1), -1).clone()

        A_f32 = A_flat.to(torch.float32)
        B_f32 = B_flat.to(torch.float32)
        diff = B_f32.unsqueeze(1) - A_f32.unsqueeze(0)
        dist = torch.norm(diff, p=2, dim=-1)

        min_dist, min_idx = dist.min(dim=1)

        idx_to_add = torch.nonzero(
            min_dist[:1] > threshold, as_tuple=False
        ).squeeze(1).tolist()

        if len(idx_to_add) > 0:
            B_to_add = B[:, idx_to_add]
            A_new = torch.cat([A, B_to_add], dim=1)
        else:
            A_new = A

        return idx_to_add, A_new, min_idx

    @torch.no_grad()
    def process_input(self, images: torch.Tensor) -> np.ndarray:
        flush_vram()

        batch_size = self.batch_size
        device = self.device
        temporal_window_size = self.temporal_window_size
        temporal_adaptive_step = self.temporal_adaptive_step

        tgt_cond_tensor = self.fast_resize(images, 256, 256)
        tgt_cond_tensor = tgt_cond_tensor / 2 + 0.5

        if self.first_frame:
            mot_bbox_param, kps_ref, kps_frame1, kps_dri = (
                self.pose_encoder.interpolate_kps_online(
                    self.ref_cond_tensor, tgt_cond_tensor, num_interp=12 + 1
                )
            )
            self.kps_ref = kps_ref
            self.kps_frame1 = kps_frame1
        else:
            mot_bbox_param, kps_dri = self.pose_encoder.get_kps(
                self.kps_ref, self.kps_frame1, tgt_cond_tensor
            )

        from src.utils.util import draw_keypoints, get_boxes

        keypoints = draw_keypoints(mot_bbox_param, device=device)
        boxes = get_boxes(kps_dri)
        keypoints = rearrange(
            keypoints.to("cpu").unsqueeze(2), "f c b h w -> b c f h w"
        )
        keypoints = keypoints.to(device=device, dtype=self.pose_guider.dtype)

        if self.first_frame:
            ref_box = get_boxes(mot_bbox_param[:1])
            ref_face = self.crop_face_tensor(self.ref_image_tensor, ref_box[0])
            motion_face = [ref_face]
            for i, frame in enumerate(images):
                motion_face.append(self.crop_face_tensor(frame, boxes[i]))
            pose_cond_tensor = torch.cat(motion_face, dim=0).transpose(0, 1)
            pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
            motion_hidden_states = self.motion_encoder(pose_cond_tensor)
            ref_motion = motion_hidden_states[:, :1]
            dri_motion = motion_hidden_states[:, 1:]

            init_motion_hidden_states = self.interpolate_tensors(
                ref_motion, dri_motion[:, :1], num=12 + 1
            )[:, :-1]
            for i in range(temporal_adaptive_step - 1):
                l = i * temporal_window_size
                r = (i + 1) * temporal_window_size
                self.motion_pile.append(init_motion_hidden_states[:, l:r])
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
                r = (i + 1) * temporal_window_size
                self.pose_pile.append(pose_fea[:, :, l:r])
            self.first_frame = False
        else:
            self.pose_pile.append(pose_fea)

        latents = (
            self.ref_image_latents.to("cpu")
            .unsqueeze(2)
            .repeat(1, 1, temporal_window_size, 1, 1)
            .to(self.device)
        )
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(
            latents, noise, self.timesteps[:1]
        )
        self.latents_pile.append(latents)

        jump = 1
        motion_hidden_state = torch.cat(list(self.motion_pile), dim=1)
        pose_cond_fea = torch.cat(list(self.pose_pile), dim=2)

        idx_to_add = []
        KEYFRAME_DISTANCE_THRESHOLD = 17.0
        KEYFRAME_CHECK_DELAY = 8
        if self.count > KEYFRAME_CHECK_DELAY:
            idx_to_add, self.motion_bank, idx_his = self.calculate_dis(
                self.motion_bank,
                motion_hidden_state,
                threshold=KEYFRAME_DISTANCE_THRESHOLD,
            )

        latents_model_input = torch.cat(list(self.latents_pile), dim=2)
        for j in range(jump):
            timesteps = reversed(self.timesteps[j::jump]).repeat_interleave(
                temporal_window_size, dim=0
            )
            timesteps = torch.stack([timesteps] * batch_size)
            timesteps = rearrange(timesteps, "b f -> (b f)")
            noise_pred = self.denoising_unet(
                latents_model_input,
                timesteps,
                encoder_hidden_states=[
                    self.encoder_hidden_states,
                    motion_hidden_state,
                ],
                pose_cond_fea=pose_cond_fea,
                return_dict=False,
            )[0]

            clip_length = noise_pred.shape[2]
            mid_noise_pred = rearrange(
                noise_pred, "b c f h w -> (b f) c h w"
            )
            mid_latents = rearrange(
                latents_model_input, "b c f h w -> (b f) c h w"
            )
            latents_model_input, pred_original_sample = self.scheduler.step(
                mid_noise_pred,
                timesteps,
                mid_latents,
                generator=self.generator,
                return_dict=False,
            )
            latents_model_input = rearrange(
                latents_model_input, "(b f) c h w -> b c f h w", f=clip_length
            )
            pred_original_sample = rearrange(
                pred_original_sample,
                "(b f) c h w -> b c f h w",
                f=clip_length,
            )
            latents_model_input = torch.cat(
                [
                    pred_original_sample[:, :, :temporal_window_size],
                    latents_model_input[:, :, temporal_window_size:],
                ],
                dim=2,
            )
            latents_model_input = latents_model_input.to(dtype=self.dtype)

        MAX_HISTORICAL_KEYFRAMES = 3
        if len(idx_to_add) > 0 and self.num_khf < MAX_HISTORICAL_KEYFRAMES:
            self.reference_control_writer.clear()
            self.reference_unet(
                pred_original_sample[:, :, 0].to(self.reference_unet.dtype),
                torch.zeros(
                    (batch_size,),
                    dtype=self.dtype,
                    device=self.reference_unet.device,
                ),
                encoder_hidden_states=self.encoder_hidden_states,
                return_dict=False,
            )
            self.reference_control_reader.update_hkf(
                self.reference_control_writer
            )
            logger.info("Historical keyframe added")
            self.num_khf += 1

        for i in range(len(self.latents_pile)):
            self.latents_pile[i] = latents_model_input[
                :,
                :,
                i * temporal_adaptive_step : (i + 1) * temporal_adaptive_step,
                :,
                :,
            ]

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