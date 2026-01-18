from typing import Any, Dict, List, Self, Tuple

import logging

from PIL.Image import Image as PILImage

from paperless.model import Dolphin
from paperless.preprocessor.helper import check_dimensions
from paperless.message import Message
from paperless.preprocessor.pdf import PortableDocumentFormat
from paperless.postprocessor.helper import parse_layout_string


class OpticalCharaterRecognizer:
    _model: Dolphin

    def __init__(self: Self) -> None:
        self._model = Dolphin()

    def process(
        self: Self,
        pdf: PortableDocumentFormat
    ) -> List[Dict[str, Any]]:
        images: List[PILImage] = pdf.to_images()
        for image in images:
            check_dimensions(image)

            # Stage 1: Page-level layout and reading order parsing.
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
                    self._model.generate([message])[0])
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

                # 
                if pip.size[0] > 3 and pip.size[1] > 3:
                    descriptor: Dict[str, Any] = {
                        "crop": pip,
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

        raise NotImplementedError()
