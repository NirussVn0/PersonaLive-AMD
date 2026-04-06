import os
import signal
import sys
import json
import logging

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2
import threading
import uuid
import time
from types import SimpleNamespace
import asyncio
import mimetypes
import torch

from webcam.config import config, Args
from webcam.util import pil_to_frame, bytes_to_pil, is_firefox, bytes_to_tensor
from webcam.connection_manager import ConnectionManager, ServerFullException

logger = logging.getLogger(__name__)

mimetypes.add_type("application/javascript", ".js")

THROTTLE = 0.001


def get_amd_device() -> torch.device:
    if not torch.cuda.is_available():
        logger.warning("No CUDA/HIP device found, falling back to CPU")
        return torch.device("cpu")
    props = torch.cuda.get_device_properties(0)
    logger.info("Detected GPU: %s (VRAM: %dMB)", props.name, props.total_mem // (1024 * 1024))
    return torch.device("cuda:0")


def configure_hip_environment() -> None:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault(
        "PYTORCH_HIP_ALLOC_CONF",
        "garbage_collection_threshold:0.9,max_split_size_mb:512",
    )
    os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")


class App:
    def __init__(self, config: Args, pipeline: object):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()

        self.produce_predictions_stop_event = None
        self.produce_predictions_task = None
        self.shutdown_event = asyncio.Event()

        self.init_app()

    def init_app(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket) -> None:
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )

                sender_task = asyncio.create_task(push_results_to_client(user_id, websocket))

                if self.produce_predictions_task is None or self.produce_predictions_task.done():
                    start_prediction_thread(user_id)

                await handle_websocket_input(user_id, websocket)

            except ServerFullException as e:
                logger.error("Server Full: %s", e)
            except WebSocketDisconnect:
                logger.info("User disconnected: %s", user_id)
            except Exception as e:
                logger.error("WS Error: %s", e)
            finally:
                if "sender_task" in locals():
                    sender_task.cancel()

                await self.conn_manager.disconnect(user_id, self.pipeline)

                if self.produce_predictions_stop_event is not None:
                    self.produce_predictions_stop_event.set()
                logger.info("Cleaned up user: %s", user_id)

        async def handle_websocket_input(user_id: uuid.UUID, websocket: WebSocket) -> None:
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")

            try:
                while True:
                    message = await websocket.receive()

                    if "text" in message:
                        try:
                            text_data = message["text"]
                            data = json.loads(text_data)
                            status = data.get("status")

                            if status == "pause":
                                params = SimpleNamespace(**{"restart": True})
                                await self.conn_manager.update_data(user_id, params)
                            elif status == "resume":
                                await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                        except Exception as e:
                            logger.error("JSON Parse Error: %s", e)

                    elif "bytes" in message:
                        image_data = message["bytes"]
                        if len(image_data) > 0:
                            input_tensor = bytes_to_tensor(image_data)
                            params = SimpleNamespace()
                            params.image = input_tensor
                            self.pipeline.accept_new_params(params)

            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error("Input Loop Error: %s", e)
                raise

        async def push_results_to_client(user_id: uuid.UUID, websocket: WebSocket) -> None:
            MIN_FPS = 10
            MAX_FPS = 30
            SMOOTHING = 0.8

            last_burst_time = time.time()
            last_queue_size = 0
            sleep_time = 1 / 40

            last_frame_time = None
            frame_time_list = []

            ema_frame_interval = sleep_time

            try:
                while True:
                    queue_size = await self.conn_manager.get_output_queue_size(user_id)
                    if queue_size > last_queue_size:
                        current_burst_time = time.time()
                        elapsed = current_burst_time - last_burst_time

                        if queue_size > 0 and elapsed > 0:
                            raw_interval = elapsed / queue_size
                            ema_frame_interval = SMOOTHING * ema_frame_interval + (1 - SMOOTHING) * raw_interval
                            sleep_time = min(max(ema_frame_interval, 1 / MAX_FPS), 1 / MIN_FPS)

                        last_burst_time = current_burst_time

                    last_queue_size = queue_size

                    frame = await self.conn_manager.get_frame(user_id)
                    if frame is None:
                        await asyncio.sleep(0.001)
                        continue

                    await websocket.send_bytes(frame)

                    if last_frame_time is None:
                        last_frame_time = time.time()
                    else:
                        frame_time_list.append(time.time() - last_frame_time)
                        if len(frame_time_list) > 100:
                            frame_time_list.pop(0)
                        last_frame_time = time.time()

                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Push Result Error: %s", e)

        def start_prediction_thread(user_id: uuid.UUID) -> None:
            self.produce_predictions_stop_event = threading.Event()

            def prediction_loop(
                uid: uuid.UUID,
                loop: asyncio.AbstractEventLoop,
                stop_event: threading.Event,
            ) -> None:
                while not stop_event.is_set():
                    images = self.pipeline.produce_outputs()
                    if len(images) == 0:
                        time.sleep(THROTTLE)
                        continue

                    frames = list(map(pil_to_frame, images))
                    asyncio.run_coroutine_threadsafe(
                        self.conn_manager.put_frames_to_output_queue(uid, frames),
                        loop,
                    )

            self.produce_predictions_task = asyncio.create_task(
                asyncio.to_thread(
                    prediction_loop,
                    user_id,
                    asyncio.get_running_loop(),
                    self.produce_predictions_stop_event,
                )
            )

        @self.app.get("/api/queue")
        async def get_queue_size() -> JSONResponse:
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/status")
        async def get_status() -> JSONResponse:
            return JSONResponse({
                "is_ready": self.pipeline.is_ready,
                "status": self.pipeline.loading_status,
            })

        @self.app.get("/api/settings")
        async def settings() -> JSONResponse:
            info_schema = pipeline.Info.schema()
            info = pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        @self.app.post("/api/upload_reference_image")
        async def upload_reference_image(ref_image: UploadFile = File(...)) -> dict:
            if not self.pipeline.is_ready:
                raise HTTPException(status_code=503, detail="Pipeline is still loading, please wait")
            try:
                data = await ref_image.read()
                img = bytes_to_pil(data)
                self.pipeline.fuse_reference(img)
                return {"status": "ok"}
            except Exception as e:
                logger.error("Reference image error: %s", e)
                raise HTTPException(status_code=500, detail="Failed to process reference image")

        @self.app.post("/api/reset")
        async def reset() -> None:
            try:
                self.pipeline.reset()
            except Exception as e:
                logger.error("Reset Error: %s", e)
                raise HTTPException(status_code=500, detail="Failed to reset pipeline")

        if not os.path.exists("./webcam/frontend/public"):
            os.makedirs("./webcam/frontend/public")

        self.app.mount(
            "/", StaticFiles(directory="./webcam/frontend/public", html=True), name="public"
        )

        @self.app.on_event("shutdown")
        async def shutdown_event() -> None:
            await self.cleanup()

    async def cleanup(self) -> None:
        logger.info("Starting cleanup process...")
        self.shutdown_event.set()

        if self.produce_predictions_stop_event is not None:
            self.produce_predictions_stop_event.set()

        if self.produce_predictions_task is not None:
            self.produce_predictions_task.cancel()
            try:
                await self.produce_predictions_task
            except asyncio.CancelledError:
                pass

        try:
            await self.conn_manager.disconnect_all(self.pipeline)
        except Exception as e:
            logger.error("Error during disconnect_all: %s", e)

        logger.info("Cleanup completed")


app_instance = None


def signal_handler(signum: int, frame: object) -> None:
    logger.info("Received signal %d, shutting down gracefully...", signum)
    if app_instance:
        def trigger_cleanup() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app_instance.cleanup())
                loop.close()
            except Exception as e:
                logger.error("Error during cleanup: %s", e)

        cleanup_thread = threading.Thread(target=trigger_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        cleanup_thread.join(timeout=5)

    sys.exit(0)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    configure_hip_environment()
    device = get_amd_device()

    if config.acceleration == "tensorrt":
        from webcam.vid2vid_trt import Pipeline
    else:
        from webcam.vid2vid import Pipeline

    pipeline = Pipeline(config, device)

    app_obj = App(config, pipeline)
    app = app_obj.app
    app_instance = app_obj

    logger.info("Initialization complete")

    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=config.reload,
            ssl_certfile=config.ssl_certfile,
            ssl_keyfile=config.ssl_keyfile,
        )
    except KeyboardInterrupt:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(app_obj.cleanup())
            loop.close()
        except Exception as e:
            logger.error("Error during cleanup: %s", e)
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        sys.exit(1)