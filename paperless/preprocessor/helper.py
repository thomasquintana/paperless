from PIL.Image import Image as PILImage


def check_dimensions(
    image: PILImage,
    max_dim_size: int = 1600,
    min_dim_size: int = 28
) -> None:
    """
    Validate that a Pillow image meets minimum and maximum size constraints.

    This function checks the pixel dimensions of the provided image and
    ensures that:
    - the largest dimension (width or height) does not exceed ``max_dim_size``
    - the smallest dimension (width or height) is not smaller than
      ``min_dim_size``

    The function performs validation only and does not modify the image.

    Args:
        image: A Pillow Image instance whose dimensions will be validated.
        max_dim_size: Maximum allowed size, in pixels, for the image's largest
            dimension.
        min_dim_size: Minimum required size, in pixels, for the image's
            smallest dimension.

    Raises:
        ValueError: If the image dimensions fall outside the allowed range.
    """
    width: int = image.size[0]
    height: int = image.size[1]

    if max(width, height) > max_dim_size or min(width, height) < min_dim_size:
        raise ValueError(
            f"Invalid image dimensions: {width}x{height}. "
            f"Largest side must be ≤ {max_dim_size}px and "
            f"smallest side ≥ {min_dim_size}px."
        )
