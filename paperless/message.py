"""
Docstring for model
"""

from typing import Any, List, Literal

from pydantic import BaseModel, Field


class ImagePart(BaseModel):
    """
    Represents a single image content part within a Dolphin request message.

    Each instance corresponds to one image payload.
    """

    # Discriminator indicating this content part is an image.
    type: Literal["image"] = Field(
        description="Content type discriminator. Always set to 'image'."
    )

    # Image source, either a URL or a base64-encoded image string
    image: Any = Field(
        description=(
            "Image input provided either as a publicly accessible URL "
            "or as a base64-encoded image string."
        )
    )


class TextPart(BaseModel):
    """
    Represents a single text content part within a Dolphin request message.
    """

    # Discriminator indicating this content part is text.
    type: Literal["text"] = Field(
        description="Content type discriminator. Always set to 'text'."
    )

    # The text string.
    text: str = Field(description="")


class Message(BaseModel):
    """
    Represents a single message in a Dolphin inference request.

    Messages follow a chat-style structure. In this simplified schema,
    only 'user' messages are supported, and each message may contain
    one or more image parts.
    """

    # Role of the message sender.
    role: Literal["assistant", "user", "system"] = Field(
        description="Role of the message sender. Must be 'user'."
    )

    # Ordered list of content parts associated with this message
    content: List[ImagePart | TextPart] = Field(
        description="List of content parts for this message."
    )
