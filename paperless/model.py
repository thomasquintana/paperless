"""
Docstring for model
"""

from typing import Any, Dict, List, Self, Tuple

from PIL.Image import Image as PILImage
from qwen_vl_utils import process_vision_info  # type: ignore
from transformers import AutoProcessor  # type: ignore
from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

import torch

from paperless.message import Message


class Dolphin:
    """
    Docstring for Dolphin
    """
    def __init__(self: Self) -> None:
        # Load model.
        self._model: Qwen2_5_VLForConditionalGeneration = \
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "ByteDance/Dolphin-v2")
        self._model.eval()  # type: ignore

        # Load the preprocessor.
        self._processor: Any = AutoProcessor.from_pretrained(
            "ByteDance/Dolphin-v2", use_fast=False)
        self._processor.tokenizer.padding_side = "left"

        # Move the model to the GPU if CUDA is available.
        if torch.cuda.is_available():
            self._model.to("cuda")  # type: ignore
            self._model = self._model.bfloat16()  # type: ignore

    def generate(self: Self, context: List[Message]) -> List[str]:
        """
        Docstring for generate

        :param self: Description
        """
        messages: List[Dict[str, Any]] = [
            message.model_dump() for message in context]

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

        # Generate the output sequence.
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
