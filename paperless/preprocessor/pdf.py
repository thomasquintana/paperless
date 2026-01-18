"""
Utilities for working with Portable Document Format (PDF) files using PyMuPDF.

This module provides a small wrapper around PyMuPDF to simplify rendering PDF
pages into Pillow images. It focuses on safe resource handling, predictable
image sizing, and compatibility with common image codecs such as PNG and JPEG.

Typical usage:
    pdf = PortableDocumentFormat(Path("example.pdf"))
    images = pdf.to_images(codec="png", max_dim_size=1024)
    pdf.close()

Callers are responsible for closing the document when finished to release
underlying file handles.
"""
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Self

from PIL import Image
from PIL.Image import Image as PILImage
from pymupdf import Document, Matrix, Page, Pixmap  # type: ignore

import pymupdf


class PortableDocumentFormat:
    """
    Lightweight wrapper around a PyMuPDF Document for rendering PDF pages.

    This class encapsulates a PDF file opened with PyMuPDF and provides a
    convenience API for converting each page into a Pillow Image. It manages
    the document lifecycle explicitly and enforces consistent rendering rules,
    such as limiting the maximum output image dimension and avoiding unintended
    upscaling.

    Notes:
    - Instances are stateful; once `close()` is called, the document can no
      longer be rendered.
    - This class does not automatically reopen documents or act as a context
      manager; callers should ensure `close()` is invoked when finished.
    """
    _document: Optional[Document]
    _path: Path

    def __init__(self: Self, path: Path) -> None:
        self._document = pymupdf.open(path)
        self._path = path

    def close(self: Self) -> None:
        """
        Close the underlying PyMuPDF document and release any file handles.

        After calling this method, the instance cannot be used for rendering
        unless a new document is opened.
        """
        if self._document is not None:
            self._document.close()
            self._document = None

    def to_images(
            self: Self,
            codec: str = "png",
            jpeg_quality: int = 95,
            max_dim_size: int = 896
        ) -> List[PILImage]:
        """
        Render each page of the PDF into a Pillow image.

        Pages are rasterized such that the largest page dimension (width or
        height) is at most ``max_dim_size`` pixels. Smaller pages are not
        upscaled.

        Args:
            codec: Output image codec to encode the rendered pixmap bytes
                (e.g., "png", "jpg", "jpeg"). The value is case-insensitive.
            jpeg_quality: JPEG encoding quality (only used when ``codec`` is
                "jpg" or "jpeg").
            max_dim_size: Maximum pixel size of the largest page dimension.

        Returns:
            A list of Pillow Image objects, one per page, in document order.

        Raises:
            RuntimeError: If the document is closed or a page has invalid
                dimensions.
        """

        if self._document is None:
            raise RuntimeError(f"The document {self._path} is closed.")

        # Normalize the codec name.
        codec = codec.lower()

        # Iterate over all the pages converting them to Pillow images.
        images: List[PILImage] = []

        for index in range(len(self._document)):
            page: Page = self._document[index]

            # Calculate the scaling factor for the image.
            largest_dim: int = max(page.rect.height, page.rect.width)

            if largest_dim <= 0:
                raise RuntimeError(
                    f"Invalid page dimensions in {self._path} at "
                    f"page {index + 1}")

            scale: float = min(1.0, max_dim_size / largest_dim)

            # Apply the scaling transformation and render the page.
            transformation: Matrix = Matrix(scale, scale)
            pixmap: Pixmap = page.get_pixmap(
                    matrix=transformation, alpha=False)

            # Convert to a pillow image.
            data: BytesIO
            if codec not in ["jpg", "jpeg"]:
                data = BytesIO(pixmap.tobytes(codec))
            else:
                data = BytesIO(pixmap.tobytes(
                    codec, jpg_quality=jpeg_quality))
            try:
                image: PILImage = Image.open(data)
                image.load()

                # Append to the list of images.
                images.append(image)
            finally:
                data.close()

        return images
