"""
Optical Character Recognition (OCR) pipeline.
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

    Extracts layout, reading order, and content from PDFs.
    Returns labeled regions (code, equations, figures, tables, text) with text or base64 data.
    """
    _batch_size: int
    _model: Dolphin

    def __init__(self: Self, batch_size: int = 32) -> None:
        self._batch_size: int = batch_size
        self._model: Dolphin = Dolphin()

    def _create_layout_message(self: Self, image: PILImage) -> Message:
        """
        Create the layout parsing message for a single page.

        Args:
            image: Page image to process.

        Returns:
            Layout parsing message.
        """
        return Message.model_validate({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "Parse the reading order of this document."
                }
            ]
        })

    def _process_layout_response(
        self: Self,
        image: PILImage,
        response: str
    ) -> Tuple[List[Dict[str, Any]], ...]:
        """
        Parse model output and crop elements for a single page.

        Args:
            image: Page image to crop from.
            response: Model output string containing layout information.

        Returns:
            Tuple of per-type element lists (code, equations, figures, tables, text).
        """
        layout: List[Tuple[Tuple[float, ...], str, List[str]]]
        try:
            layout = parse_layout_string(response)
        except ValueError as error:
            layout = [((0, 0, *image.size), 'distorted_page', [])]
            logging.warning(error)

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

    def _extract_content(
        self: Self,
        elements: List[Dict[str, Any]],
        type_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Generic extractor for text-based content.

        Args:
            elements: List of element descriptors to extract text from.
            type_prompt: The prompt text to send to the model.

        Returns:
            List of normalized OCR outputs.
        """
        if not elements:
            return []

        messages: List[Message] = [Message.model_validate({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": element["image"],
                },
                {
                    "type": "text",
                    "text": type_prompt
                }
            ]
        }) for element in elements]

        # Use the stored batch size for inference.
        outputs: List[str] = self._prompt(messages, batch_size=self._batch_size)

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

        Args:
            elements: List of element descriptors to process.

        Returns:
            List of normalized outputs with base64 encoded images.
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

    def _prompt(
        self: Self,
        messages: List[Message],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Run generation on a list of messages, optionally batched.

        Args:
            messages: List of messages to process.
            batch_size: Optional batch size override.

        Returns:
            List of generated strings.
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

        Args:
            pdf: PDF document wrapper.
            exclude: Optional list of element types to exclude.

        Returns:
            List of OCR results per page.
        """
        if exclude is None:
            exclude = []

        images: List[PILImage] = pdf.to_images()
        for image in images:
            check_dimensions(image)

        # Stage 1: Global Layout Detection
        layout_messages: List[Message] = [self._create_layout_message(img) for img in images]
        
        # Run inference in batches.
        layout_responses: List[str] = self._prompt(layout_messages, batch_size=self._batch_size)

        # Process responses to get elements for each page.
        all_page_elements: List[Tuple[List[Dict[str, Any]], ...]] = []
        for image, response in zip(images, layout_responses):
            all_page_elements.append(self._process_layout_response(image, response))

        # Stage 2: Global Content Extraction
        type_queues: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {
            'code': [],
            'equ': [],
            'fig': [],
            'tab': [],
            'text': []
        }

        # Index mapping.
        tuple_idx_to_key: Dict[int, Literal['code', 'equ', 'fig', 'tab', 'text']] = {
            0: 'code', 1: 'equ', 2: 'fig', 3: 'tab', 4: 'text'
        }

        for page_idx, page_elems in enumerate(all_page_elements):
            for tuple_idx, elem_list in enumerate(page_elems):
                key: Literal['code', 'equ', 'fig', 'tab', 'text'] = tuple_idx_to_key[tuple_idx]
                # Filter exclusions early
                if key == 'code' and 'code' in exclude: continue
                if key == 'equ' and 'equations' in exclude: continue
                if key == 'fig' and 'figures' in exclude: continue
                if key == 'tab' and 'tables' in exclude: continue
                if key == 'text' and 'text' in exclude: continue

                for elem in elem_list:
                    type_queues[key].append((page_idx, elem))

        # Run batch inference for each type
        code_input: List[Dict[str, Any]] = [item[1] for item in type_queues['code']]
        code_results: List[Dict[str, Any]] = self._extract_content(code_input, "Read code in the image.")
        

        equ_input: List[Dict[str, Any]] = [item[1] for item in type_queues['equ']]
        equ_results: List[Dict[str, Any]] = self._extract_content(equ_input, "Read formula in the image.")
        

        tab_input: List[Dict[str, Any]] = [item[1] for item in type_queues['tab']]
        tab_results: List[Dict[str, Any]] = self._extract_content(tab_input, "Parse the table in the image.")
        

        text_input: List[Dict[str, Any]] = [item[1] for item in type_queues['text']]
        text_results: List[Dict[str, Any]] = self._extract_content(text_input, "Read text in the image.")

        # Figures (no inference)
        fig_input: List[Dict[str, Any]] = [item[1] for item in type_queues['fig']]
        fig_results: List[Dict[str, Any]] = self._extract_figures(fig_input)

        # Reconstruct the per-page results structure.
        outputs: List[Dict[str, Any]] = [
            {
                "page": i + 1,
                "codes": [],
                "equations": [],
                "figures": [],
                "tables": [],
                "text": []
            } 
            for i in range(len(images))
        ]

        def distribute_results(
            original_queue: List[Tuple[int, Dict[str, Any]]], 
            processed_results: List[Dict[str, Any]], 
            output_key: Literal['codes', 'equations', 'figures', 'tables', 'text']
        ) -> None:
            for (page_idx, _), result in zip(original_queue, processed_results):
                outputs[page_idx][output_key].append(result)

        distribute_results(type_queues['code'], code_results, 'codes')
        distribute_results(type_queues['equ'], equ_results, 'equations')
        distribute_results(type_queues['fig'], fig_results, 'figures')
        distribute_results(type_queues['tab'], tab_results, 'tables')
        distribute_results(type_queues['text'], text_results, 'text')

        return outputs
