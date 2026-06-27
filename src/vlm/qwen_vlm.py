"""Qwen2.5-VL Vision-Language Model wrapper.

Loads ``Qwen/Qwen2.5-VL-7B-Instruct`` (configurable) with optional 4-bit/8-bit
quantization — essential to fit a 7B VLM alongside YOLO on an 8 GB RTX 5060.

Public methods are synchronous and thread-safe (guarded by a lock); the
asynchronous, event-driven scheduling lives in
:class:`~src.vlm.vlm_worker.VLMWorker` so this class stays a clean inference
primitive that can also be called directly for image analysis.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np

from src.config.settings import VLMConfig
from src.utils.image_utils import to_pil
from src.utils.logger import logger
from src.vlm import prompts


class QwenVLM:
    def __init__(self, config: VLMConfig):
        self.config = config
        self._lock = threading.Lock()
        self.model = None
        self.processor = None
        self.loaded = False
        self.load_error: Optional[str] = None

    # ------------------------------------------------------------- loading
    def load(self) -> bool:
        """Load weights/processor. Returns True on success.

        Failure is non-fatal: the pipeline continues with detection+tracking
        and surfaces the error in the UI, so a missing model never crashes the
        whole system.
        """
        if self.loaded:
            return True
        try:
            import torch
            from transformers import (AutoProcessor,
                                      Qwen2_5_VLForConditionalGeneration)

            logger.info("Loading VLM: {} (quant={})",
                        self.config.model_id, self.config.quantization)
            quant_cfg = self._build_quant_config()
            dtype = torch.float16

            kwargs = dict(torch_dtype=dtype, device_map=self.config.device)
            if quant_cfg is not None:
                kwargs["quantization_config"] = quant_cfg
                kwargs["device_map"] = "auto"

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_id, **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )
            self.model.eval()
            self.loaded = True
            logger.info("VLM ready.")
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_error = str(exc)
            logger.exception("Failed to load VLM: {}", exc)
            return False

    def _build_quant_config(self):
        if self.config.quantization == "none":
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except Exception:  # noqa: BLE001
            logger.warning("bitsandbytes unavailable; loading VLM unquantized")
            return None
        if self.config.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    # ----------------------------------------------------------- inference
    def infer(self, images: List[np.ndarray], prompt: str,
              system: Optional[str] = None,
              max_new_tokens: Optional[int] = None) -> str:
        """Run a single VL generation over one or more BGR frames.

        ``max_new_tokens`` overrides the configured cap for this call — used by the
        structured report (multi-section answers) so they are not truncated."""
        if not self.loaded and not self.load():
            return f"[VLM unavailable: {self.load_error}]"

        import torch

        pil_images = [to_pil(im) for im in images]
        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": system or prompts.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        with self._lock:
            t0 = time.time()
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Use Qwen's official vision pre-processor so the count of image
                # placeholder tokens in `text` exactly matches the pixel tensors.
                # A mismatch here is the usual trigger for CUDA device-side
                # asserts during generation.
                try:
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                except Exception:  # noqa: BLE001
                    image_inputs, video_inputs = pil_images, None
                inputs = self.processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(self.model.device)

                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                        do_sample=self.config.temperature > 0,
                        temperature=max(self.config.temperature, 1e-4),
                    )
                trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
                answer = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,  # BPE: cleanup corrupts spacing
                )[0].strip()
                logger.debug("VLM inference {:.2f}s -> {} chars",
                             time.time() - t0, len(answer))
                return answer
            except Exception as exc:  # noqa: BLE001
                logger.exception("VLM inference failed: {}", exc)
                return f"[VLM error: {exc}]"

    # ------------------------------------------ high-level analysis helpers
    def scene_summary(self, image: np.ndarray, modality: str,
                      detections: List[str], events: List[str]) -> str:
        p = (prompts.event_reasoning_prompt(modality, detections, events)
             if events else
             prompts.scene_summary_prompt(modality, detections, events))
        return self.infer([image], p)

    def analyze_eo(self, image: np.ndarray, detections: List[str]) -> str:
        return self.infer([image], prompts.eo_analysis_prompt(detections))

    def analyze_ir(self, image: np.ndarray, detections: List[str]) -> str:
        return self.infer([image], prompts.ir_analysis_prompt(detections))

    def fused_assessment(self, eo_img: np.ndarray, ir_img: np.ndarray,
                         eo_summary: str, ir_summary: str,
                         detections: List[str]) -> str:
        p = prompts.fused_assessment_prompt(eo_summary, ir_summary, detections)
        return self.infer([eo_img, ir_img], p)

    def mission_summary(self, image: Optional[np.ndarray],
                        timeline: List[str], detections_summary: str) -> str:
        p = prompts.mission_summary_prompt(timeline, detections_summary)
        return self.infer([image] if image is not None else [], p)

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.processor = None
            self.loaded = False
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
