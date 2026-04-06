import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from threading import Thread, Event

import time
from typing import List
import torch
from .config import Args
from pydantic import BaseModel, Field
from PIL import Image
from src.wrapper import PersonaLive
import queue

page_content = """<h1 class="text-3xl font-bold">🎭 PersonaLive!</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/GVCLab/PersonaLive"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">PersonaLive
</a>
video-to-video pipeline with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "PersonaLive"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device):
        self.args = args
        self.device = device

        self.is_ready = False
        self.loading_status = "Initializing..."
        self.pipeline = None

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.reference_queue = queue.Queue()

        self.stop_event = Event()
        self.restart_event = Event()
        self.reset_event = Event()

        self.thread = Thread(target=self._load_and_run, daemon=True)
        self.thread.start()

    def _load_and_run(self):
        torch.set_grad_enabled(False)

        self.loading_status = "Loading models..."
        print("[Pipeline] Loading models...")
        load_start = time.time()
        self.pipeline = PersonaLive(self.args, self.device)
        print(f"[Pipeline] Models loaded in {time.time() - load_start:.1f}s")

        self.loading_status = "Warming up DirectML..."
        print("[Pipeline] Warming up DirectML JIT compiler...")
        self._warmup_dml()

        self.loading_status = "Ready"
        self.is_ready = True
        print("[Pipeline] Ready! Waiting for reference image...")

        self._generate_loop()

    def _warmup_dml(self):
        warmup_start = time.time()
        try:
            with torch.no_grad():
                dummy = torch.randn(
                    1, 3, 256, 256,
                    device=self.device,
                    dtype=self.pipeline.dtype,
                )
                latent = self.pipeline.vae.encode(dummy).latent_dist.mean
                self.pipeline.vae.decode(latent)
                del dummy, latent
            print(f"[Pipeline] DML warmup done in {time.time() - warmup_start:.1f}s")
        except Exception as e:
            print(f"[Pipeline] DML warmup skipped: {e}")

    def _generate_loop(self):
        chunk_size = 4

        reference_img = self.reference_queue.get()
        self.pipeline.fuse_reference(reference_img)
        print("[Pipeline] Fuse reference done, starting inference loop")

        while not self.stop_event.is_set():
            if self.restart_event.is_set():
                self._clear_queue(self.input_queue)
                self.restart_event.clear()

            images = self._read_images(chunk_size)
            if images is None:
                continue

            if self.reset_event.is_set():
                self.pipeline.reset()
                self._clear_queue(self.input_queue)
                self._clear_queue(self.reference_queue)
                print("[Pipeline] Waiting for new reference image...")
                reference_img = self.reference_queue.get()
                self.pipeline.fuse_reference(reference_img)
                print("[Pipeline] Fuse reference done")
                self.reset_event.clear()
                continue

            images = torch.cat([img.to(self.device) for img in images], dim=0)

            video = self.pipeline.process_input(images)
            for image in video:
                self.output_queue.put(image)

    def _read_images(self, num_needed):
        collected = []
        while len(collected) < num_needed:
            if self.stop_event.is_set() or self.reset_event.is_set():
                return None
            try:
                collected.append(self.input_queue.get(timeout=0.05))
            except queue.Empty:
                continue
        return collected

    def reset(self):
        self.reset_event.set()
        self._clear_queue(self.output_queue)

    def accept_new_params(self, params: "Pipeline.InputParams"):
        if hasattr(params, "image"):
            image_pil = params.image.to(self.device).float() / 255.0
            image_pil = image_pil * 2.0 - 1.0
            image_pil = image_pil.permute(2, 0, 1).unsqueeze(0)
            self.input_queue.put(image_pil.cpu())

        if hasattr(params, "restart") and params.restart:
            self.restart_event.set()
            self._clear_queue(self.output_queue)

    def fuse_reference(self, ref_image):
        self.reference_queue.put(ref_image)

    def produce_outputs(self) -> List[Image.Image]:
        results = []
        try:
            while True:
                data = self.output_queue.get_nowait()
                from .util import array_to_image
                results.append(array_to_image(data))
        except queue.Empty:
            pass
        return results

    def close(self):
        print("[Pipeline] Shutting down...")
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        print("[Pipeline] Closed")

    @staticmethod
    def _clear_queue(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass