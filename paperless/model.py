"""
Model wrapper for Dolphin-v2 (Qwen2.5-VL) OCR inference.
"""

from typing import Any, Dict, List, Self, Tuple

import logging

from deepspeed import InferenceEngine  # type: ignore
from PIL.Image import Image as PILImage
from qwen_vl_utils import process_vision_info  # type: ignore
from transformers import AutoProcessor  # type: ignore
from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

import deepspeed
import torch

from paperless.message import Message


class Dolphin:
    """
    Inference wrapper for the Dolphin-v2 vision-language model.

    Responsible for loading the model + processor, moving to GPU when
    available, and generating text outputs from chat-style inputs.
    """
    def __init__(self: Self) -> None:
        # Initialize logging.
        self._logger = logging.getLogger("uvicorn.error")

        # Load model.
        if torch.cuda.is_available():
            # Enable TF32 for faster FP32 operations (e.g. internal accumulations).
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            device_map = "cuda"
            # Check if the GPU supports bfloat16 (Ampere or newer).
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
            attn_implementation = "flash_attention_2"
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            attn_implementation = "eager"

        self._model: Qwen2_5_VLForConditionalGeneration | InferenceEngine = \
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "ByteDance/Dolphin-v2",
                device_map=device_map,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation
            )
        self._model.eval()  # type: ignore

        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(
                "Loaded ByteDance/Dolphin-v2 on %s.",
                device_map
            )

        # Optimize with DeepSpeed if available.
        if torch.cuda.is_available():
            try:
                self._model = deepspeed.init_inference(
                    model=self._model,
                    dtype=torch_dtype,
                    replace_with_kernel_inject=True
                )
                if self._logger.isEnabledFor(logging.INFO):
                    self._logger.info(
                        "DeepSpeed inference optimization enabled."
                    )
            except ImportError:
                pass
            except Exception as error:
                self._logger.warning(
                    "DeepSpeed optimization skipped: %s", error
                )

        # Load the preprocessor.
        self._processor: Any = AutoProcessor.from_pretrained(
            "ByteDance/Dolphin-v2", use_fast=False)
        self._processor.tokenizer.padding_side = "left"

    def generate(self: Self, context: List[Message]) -> List[str]:
        """
        Generate text responses for a batch of message contexts.

        Args:
            context: List of Message objects representing a single prompt
                sequence with image and text parts.

        Returns:
            List of decoded model outputs, one per context.
        """
        messages: List[List[Dict[str, Any]]] = [
            [message.model_dump()] for message in context
        ]

        # Apply the template to the text inputs.
        texts: List[str] = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False)

        # Collect all the image inputs.
        images: List[PILImage] | None = None
        items: Tuple[Any, Any, Any] = process_vision_info(messages)
        if items[0] is not None:
            images = [items[0]]

        # Prepare the model inputs.
        inputs: Any = self._processor(
            text=texts,
            images=images if images else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)  # type: ignore

        # Generate the output sequence with inference optimizations.
        with torch.inference_mode():
            outputs: Any = self._model.generate(  # type: ignore
                **inputs,
                do_sample=False,
                max_new_tokens=4096,
                temperature=None,
                # repetition_penalty=1.05
            )
            outputs = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(
                    inputs.input_ids, outputs)]

            # Decode the output sequence.
            return self._processor.batch_decode(
                outputs,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True
            )
