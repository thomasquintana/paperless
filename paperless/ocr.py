"""
Optical Character Recognition (OCR) pipeline.

This module provides a lightweight OCR orchestrator that:
1) Renders PDF pages into images.
2) Asks a layout/reading-order.
3) Crops each located element ("pip" / picture-in-picture region) and routes it
   to a specialized extractor (text, code, equations, tables, figures).
4) Returns a normalized list of OCR results with labels, bounding boxes,
   reading order, tags, and extracted text (or base64 image data for figures).

Design notes:
- Layout parsing relies on `parse_layout_string(...)` and may fail on malformed
  model output; this module degrades gracefully by treating the full page as a
  single "distorted_page" region.
- Element crops smaller than MIN_PIP_* are ignored to reduce noise and avoid
  pointless model calls.
"""
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Self, Tuple

import base64
import logging

from PIL.Image import Image as PILImage

from paperless.model import Dolphin
from paperless.preprocessor.helper import check_dimensions
from paperless.message import Message
from paperless.preprocessor.pdf import PortableDocumentFormat
from paperless.postprocessor.helper import parse_layout_string

# Minimum crop size thresholds (in pixels) for a candidate element region.
# Crops smaller than this are ignored to avoid tiny/noisy regions.
MIN_PIP_HEIGHT: int = 3
MIN_PIP_WIDTH: int = 3


class OpticalCharaterRecognizer:
    """
    OCR orchestrator.

    The class is responsible for:
    - Batch prompting the model for layout, reading order, and content
        extraction.
    - Routing cropped regions to the right extraction prompt based on label.
    - Returning results in a consistent JSON-like schema.

    Each output dict includes:
      - label: detected semantic type (e.g., "code", "equ", "fig", "tab",
        "text")
      - bbox: [x1, y1, x2, y2] crop coordinates in page space
      - text: extracted text, or base64 PNG for figures
      - reading_order: integer indicating element order on the page
      - tags: metadata tags returned by the layout parser (if any)
    """
    _model: Dolphin

    def __init__(self: Self) -> None:
        self._model = Dolphin()

    def _extract_codes(
        self: Self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract source code text from a list of cropped "code" elements.

        Args:
            elements: List of element descriptors, each containing:
                - image (PILImage): cropped region
                - label (str)
                - bbox (List[float|int])
                - reading_order (int)
                - tags (List[str])

        Returns:
            List of normalized OCR outputs for code regions.
        """
        messages: List[Message] = [Message.model_validate({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": element["image"],
                },
                {
                    "type": "text",
                    "text": "Read code in the image."
                }
            ]
        }) for element in elements]

        outputs: List[str] = self._prompt(messages)

        return [{
            "label": element["label"],
            "bbox": element["bbox"],
            "text": output.strip(),
            "reading_order": element["reading_order"],
            "tags": element["tags"]
        } for element, output in zip(elements, outputs)]

    def _extract_equations(
        self: Self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract mathematical formulas/equations from cropped "equ" elements.

        Args:
            elements: List of element descriptors (see _extract_codes).

        Returns:
            List of normalized OCR outputs for equation regions.
        """
        messages: List[Message] = [Message.model_validate({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": element["image"],
                },
                {
                    "type": "text",
                    "text": "Read formula in the image."
                }
            ]
        }) for element in elements]

        outputs: List[str] = self._prompt(messages)

        return [{
            "label": element["label"],
            "bbox": element["bbox"],
            "text": output.strip(),
            "reading_order": element["reading_order"],
            "tags": element["tags"]
        } for element, output in zip(elements, outputs)]

    def _extract_figures(
        self: Self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract figures as base64-encoded PNG images.

        Figures are not OCR'd to text here. Instead, the cropped image is
        serialized as PNG bytes and returned as base64 text so downstream
        consumers can store/transmit it as JSON.

        Args:
            elements: List of element descriptors (see _extract_codes).

        Returns:
            List of normalized outputs where "text" is base64 PNG data.
        """
        outputs: List[Dict[str, Any]] = []
        for element in elements:
            buffer: BytesIO = BytesIO()
            element["image"].save(buffer, format='PNG')
            image: bytes = buffer.getvalue()
            text: str = base64.b64encode(image).decode('utf-8')
            outputs.append({
                "label": element["label"],
                "bbox": element["bbox"],
                "text": text,
                "reading_order": element["reading_order"],
                "tags": element["tags"]
            })
        return outputs

    def _extract_tables(
        self: Self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract structured table content from cropped "tab" elements.

        Args:
            elements: List of element descriptors (see _extract_codes).

        Returns:
            List of normalized OCR outputs for table regions.
        """
        messages: List[Message] = [Message.model_validate({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": element["image"],
                },
                {
                    "type": "text",
                    "text": "Parse the table in the image."
                }
            ]
        }) for element in elements]

        outputs: List[str] = self._prompt(messages)

        return [{
            "label": element["label"],
            "bbox": element["bbox"],
            "text": output.strip(),
            "reading_order": element["reading_order"],
            "tags": element["tags"]
        } for element, output in zip(elements, outputs)]

    def _extract_text(
        self: Self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract plain text from cropped elements.

        Args:
            elements: List of element descriptors (see _extract_codes).

        Returns:
            List of normalized OCR outputs for text regions.
        """
        messages: List[Message] = [Message.model_validate({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": element["image"],
                },
                {
                    "type": "text",
                    "text": "Read text in the image."
                }
            ]
        }) for element in elements]

        outputs: List[str] = self._prompt(messages)

        return [{
            "label": element["label"],
            "bbox": element["bbox"],
            "text": output.strip(),
            "reading_order": element["reading_order"],
            "tags": element["tags"]
        } for element, output in zip(elements, outputs)]

    def _locate_elements(
        self: Self,
        image: PILImage
    ) -> Tuple[List[Dict[str, Any]], ...]:
        """
        Identify and crop page elements, returning per-type element groups.

        This is a two-stage process:
        1) Page-level: Prompt the model for reading order / layout
            serialization.
        2) Element-level: Crop each region, filter by size, and route into
            lists.

        Args:
            image: Full page image.

        Returns:
            Tuple of element lists in this fixed order:
                (code_elements, equation_elements, figure_elements,
                 table_elements, text_elements)

            Each element descriptor contains:
                - image (PILImage): cropped region
                - label (str): element label from layout parser
                - bbox (List[float|int]): [x1, y1, x2, y2]
                - reading_order (int): stable order assigned during iteration
                - tags (List[str]): metadata tags from layout parser
        """
        message: Message = Message.model_validate({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "Parse the reading order of this "
                            "document."
                }
            ]
        })
        layout: List[Tuple[Tuple[float, ...], str, List[str]]]
        try:
            layout = parse_layout_string(
                self._prompt([message])[0])
        except ValueError as error:
            layout = [((0, 0, *image.size), 'distorted_page', [])]

            # Log bad parts.
            logging.warning(error)

        # Stage 2: Element-level content parsing.
        code_elements: List[Dict[str, Any]] = []
        equation_elements: List[Dict[str, Any]] = []
        figure_elements: List[Dict[str, Any]] = []
        table_elements: List[Dict[str, Any]] = []
        text_elements: List[Dict[str, Any]] = []

        reading_order: int = 0

        for coordinates, label, meta in layout:
            pip: PILImage = image.crop(coordinates)  # type: ignore
            if pip.size[0] > MIN_PIP_WIDTH and pip.size[1] > MIN_PIP_HEIGHT:
                descriptor: Dict[str, Any] = {
                    "image": pip,
                    "label": label,
                    "bbox": list(coordinates),
                    "reading_order": reading_order,
                    "tags": meta,
                }
                if label == "code":
                    code_elements.append(descriptor)
                elif label == "equ":
                    equation_elements.append(descriptor)
                elif label == "fig":
                    figure_elements.append(descriptor)
                elif label == "tab":
                    table_elements.append(descriptor)
                else:
                    text_elements.append(descriptor)

            reading_order += 1

        return (
            code_elements,
            equation_elements,
            figure_elements,
            table_elements,
            text_elements
        )

    def _prompt(
        self: Self,
        messages: List[Message],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Run generation on a list of messages, optionally batched.

        Args:
            messages: Model-ready Message objects.
            batch_size: Number of messages per batch. If None, uses all
                messages in one batch.

        Returns:
            List of model outputs (one string per input message).

        Notes:
            Batching is useful to control memory usage and latency behavior
            depending on your hardware/backend.
        """
        outputs: List[str] = []

        if len(messages) == 0:
            return outputs

        # Determine the batch size.
        if batch_size is None:
            batch_size = len(messages)

        # Process the messages in batches.
        for index in range(0, len(messages), batch_size):
            batch = messages[index:index + batch_size]

            output: List[str] = self._model.generate(batch)
            outputs.extend(output)

        return outputs

    def process(
        self: Self,
        pdf: PortableDocumentFormat,
        exclude: Optional[List[Literal['code',
                                       'equations',
                                       'figures',
                                       'tables',
                                       'text']]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform OCR over every page in a PDF document.

        For each page image:
          1) Validate image dimensions (guard rails).
          2) Locate elements via page-level layout prompt.
          3) Extract each element type with specialized prompts.
          4) Aggregate results into one list.

        Args:
            pdf: A PortableDocumentFormat wrapper capable of rendering pages
                into images.
            exclude: A list of page elements to exclude from the results.

        Returns:
            List of OCR results across all pages.
        """
        outputs: List[Dict[str, Any]] = []
        images: List[PILImage] = pdf.to_images()
        for index, image in enumerate(images):
            check_dimensions(image)

            # Stage 1: Locate the elements of interest in the document.
            elements: Tuple[List[Dict[str, Any]], ...] = self._locate_elements(
                image)

            # Stage 2: Process the elements.
            if exclude is None:
                exclude = []

            code: List[Dict[str, Any]] = self._extract_codes(
                elements[0]) if 'code' not in exclude else []
            equations: List[Dict[str, Any]] = self._extract_equations(
                elements[1]) if 'equations' not in exclude else []
            figures: List[Dict[str, Any]] = self._extract_figures(
                elements[2]) if 'figures' not in exclude else []
            tables: List[Dict[str, Any]] = self._extract_tables(
                elements[3]) if 'tables' not in exclude else []
            text: List[Dict[str, Any]] = self._extract_text(
                elements[4]) if 'text' not in exclude else []

            outputs.append({
                "page": index + 1,
                "codes": code,
                "equations": equations,
                "figures": figures,
                "tables": tables,
                "text": text
            })

        return outputs
