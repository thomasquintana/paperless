"""
OCR inference server.
"""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import unquote

import logging
import tempfile

from fastapi import FastAPI
from pydantic import AnyUrl, BaseModel, Field
from requests import Response  # type: ignore
from requests_file import FileAdapter  # type: ignore

import boto3
import requests

from paperless.ocr import OpticalCharaterRecognizer
from paperless.preprocessor.pdf import PortableDocumentFormat


class DocumentPart(BaseModel):
    """
    Represents a reference to a document used as model input.

    This model describes a single document that is not included directly in
    the request payload, but is instead referenced by a URL. The document is
    expected to be retrievable by the inference service at request time, using
    schemes like https, s3, or file (when available on the host).
    """
    url: AnyUrl = Field(
        description="Publicly reachable URL of the document.")
    kind: Literal["url"] = Field(
        "url", alias="type", description="Document source type.")
    exclude: Optional[List[Literal['code',
                                   'equations',
                                   'figures',
                                   'tables',
                                   'text']]] = Field(
        description="A list of element types to exclude.",
        default=None
    )

    class Config:
        populate_by_name = True


class ElementPart(BaseModel):
    """
    Normalized OCR output for a single detected page element.

    This model is the per-region payload produced after:
      1) the page-level layout step assigns a label + bounding box, and
      2) the element-level extractor returns either text
        (code/equations/tables/text) or base64-encoded PNG image data
        (figures).

    Fields are intentionally generic so callers can store/serialize results
    uniformly regardless of element type.
    """

    # Semantic element label predicted by the layout/reading-order step.
    # Common values: "code", "equ", "fig", "tab", "text", "distorted_page".
    label: str = Field(description="Semantic element type label.")

    # Bounding box coordinates in page pixel space: [x1, y1, x2, y2].
    # These coordinates correspond to the crop region taken from the page
    # image.
    bbox: List[float] = Field(
        description="Bounding box in page pixel coordinates: [x1, y1, x2, y2]."
    )

    # Extracted content for the region:
    # - For code/equations/tables/text: the extracted text.
    # - For figures: base64-encoded PNG bytes for the cropped region.
    text: str = Field(
        description="Extracted text, or base64 PNG data when label == 'fig'."
    )

    # Reading order index assigned during layout iteration.
    # This provides a stable ordering within the page
    # (0-based in current pipeline).
    reading_order: int = Field(
        description="Stable per-page reading order index assigned during "
                    "layout iteration."
    )

    # Optional metadata tags produced by the layout parser (e.g., style hints,
    # section markers, or other model-provided attributes).
    tags: List[str] = Field(
        default_factory=list,
        description="Optional metadata tags associated with the element."
    )


class PagePart(BaseModel):
    """
    Normalized OCR output for a single PDF page.

    This model groups extracted element regions by their semantic type, while
    keeping per-element bounding boxes and reading order available for
    downstream reconstruction, indexing, or re-rendering.

    The lists are populated from the orchestrator's routing logic:
      - codes: label == "code"
      - equations: label == "equ"
      - figures: label == "fig" (ElementPart.text is base64 PNG)
      - tables: label == "tab"
      - text: everything else (including any fallback "distorted_page" region)
    """

    # 1-based page number in the original PDF.
    page: int = Field(
        description="1-based page index within the PDF document.")

    # Extracted code regions found on the page.
    codes: List[ElementPart] = Field(
        default_factory=list,
        description="List of extracted code elements for this page."
    )

    # Extracted equation/formula regions found on the page.
    equations: List[ElementPart] = Field(
        default_factory=list,
        description="List of extracted equation elements for this page."
    )

    # Extracted figure regions found on the page (base64 PNG data).
    figures: List[ElementPart] = Field(
        default_factory=list,
        description="List of extracted figure elements for this page "
                    "(base64 PNG in ElementPart.text)."
    )

    # Extracted table regions found on the page.
    tables: List[ElementPart] = Field(
        default_factory=list,
        description="List of extracted table elements for this page."
    )

    # Extracted plain-text regions found on the page.
    text: List[ElementPart] = Field(
        default_factory=list,
        description="List of extracted text elements for this page."
    )


class OcrRequest(BaseModel):
    """
    Top-level request model for optical character recognition.
    """
    model: Literal["dolphin2"] = Field(
        description="Model to use.")
    documents: List[DocumentPart] = Field(
        description="One or more documents to process."
    )


class OcrResponse(BaseModel):
    """
    Top-level OCR response container.

    This model represents the full, normalized output of the OCR pipeline
    for an entire document. It is intentionally minimal and stable so it
    can serve as a long-lived API contract between the OCR service and
    downstream consumers (indexers, search, vectorization, UI renderers,
    etc.).

    The response is organized hierarchically:
      - The document consists of one or more pages.
      - Each page contains grouped OCR elements (code, equations, figures,
        tables, and text).
      - Each element includes bounding box coordinates, reading order, tags,
        and extracted content.

    The ordering of `pages` reflects the original PDF page order (1 → N).
    """

    # Ordered list of OCR results, one entry per PDF page.
    pages: List[PagePart] = Field(
        description="Ordered list of per-page OCR results for the document."
    )


ocr: OpticalCharaterRecognizer = OpticalCharaterRecognizer()
server: FastAPI = FastAPI()


@server.post("/v1/ocr")
async def parse(request: OcrRequest) -> OcrResponse:
    """
    Run the OCR pipeline on one or more documents.

    Downloads each document to a temporary workspace, renders pages to images,
    runs layout + element extraction, and returns normalized per-page results.
    """

    results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as workspace:
        session: requests.Session = requests.Session()
        session.mount('file://', FileAdapter())
        for document in request.documents:
            # Extract the file name from the URL.
            name: str = Path(document.url.path or "document.pdf").name
            name = unquote(name)
            path: Path = Path(workspace) / name

            # Download the file.
            if document.url.scheme == "s3":
                bucket_name: Optional[str] = document.url.host
                object_key: Optional[str] = document.url.path
                if object_key is not None:
                    object_key = object_key.lstrip('/')

                s3_client = boto3.client('s3')
                s3_client.download_file(bucket_name, object_key, path)
            else:
                response: Response = session.get(
                    document.url.encoded_string(),
                    stream=True,
                    timeout=60.0)
                response.raise_for_status()

                with path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk is not None:
                            output.write(chunk)

            pdf: PortableDocumentFormat = PortableDocumentFormat(path)
            try:
                results.extend(ocr.process(pdf, document.exclude))
            except Exception as error:
                logging.error(error)
            finally:
                pdf.close()

    return OcrResponse.model_validate({"pages": results})
