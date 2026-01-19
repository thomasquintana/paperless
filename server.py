"""
OCR inference server.
"""
from pathlib import Path
from typing import Any, Dict, List, Literal
from urllib.parse import unquote

import logging
import tempfile

from fastapi import FastAPI
from pydantic import BaseModel, Field, HttpUrl
from requests import Response  # type: ignore

import requests

from paperless.ocr import OpticalCharaterRecognizer
from paperless.preprocessor.pdf import PortableDocumentFormat


class RemoteDocumentPart(BaseModel):
    """
    Represents a reference to a remotely hosted document used as model input.

    This model describes a single document that is not included directly in
    the request payload, but is instead referenced by a URL. The document is
    expected to be retrievable by the inference service at request time.
    """
    url: HttpUrl = Field(
        description="Publicly reachable URL of the document.")
    kind: Literal["url"] = Field(
        "url", alias="type", description="Document source type.")

    class Config:
        populate_by_name = True


class OcrRequest(BaseModel):
    """
    Top-level request model for optical character recognition.
    """
    model: Literal["dolphin2"] = Field(
        description="Model to use.")
    documents: List[RemoteDocumentPart] = Field(
        description="One or more documents to process."
    )


class OcrResponse(BaseModel):
    """
    Top-level response model for optical character recognition.
    """
    pass


ocr: OpticalCharaterRecognizer = OpticalCharaterRecognizer()
server: FastAPI = FastAPI()


@server.post("/v1/ocr")
async def parse(request: OcrRequest) -> OcrResponse:
    """
    Docstring for parse

    :param request: Description
    :type request: DolphinRequest
    """

    results: List[List[Dict[str, Any]]] = []

    with tempfile.TemporaryDirectory() as workspace:
        for document in request.documents:
            # Extract the file name from the URL.
            name: str = Path(document.url.path or "document.pdf").name
            name = unquote(name)
            path: Path = Path(workspace) / name

            # Download the file.
            response: Response = requests.get(
                document.url.encoded_string(),
                stream=True,
                timeout=60.0)
            response.raise_for_status()

            with path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk is not None:
                        output.write(chunk)

            # 
            pdf: PortableDocumentFormat = PortableDocumentFormat(path)
            try:
                results.append(ocr.process(pdf))
            except Exception as error:
                logging.error(error)
            finally:
                pdf.close()

    print(results)

    raise NotImplementedError()
