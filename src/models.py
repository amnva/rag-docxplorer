from typing import Union, Literal
from pydantic import BaseModel

class TextChunk(BaseModel): # pydantic models
    type: Literal["text"]
    source: str
    page: int
    content: str

class ImageChunk(BaseModel):
    type: Literal["image"]
    source: str
    page: int
    content: str  # base64 encoding
    mime_type: str

ChunkMetadata = Union[TextChunk, ImageChunk]
