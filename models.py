from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator
from pathlib import PurePosixPath


class PresignRequest(BaseModel):
    document_type: str
    filename: str


class DocumentKey(BaseModel):
    objectKey: str = Field(..., min_length=1)
    fileUrl: str  # app-relative or absolute URL/path to the file
    kind: Optional[int] = None
    method: Optional[int] = None

    @property
    def filename(self) -> str:
        # convenient helper: "99e39c26_Child-Planning.docx"
        return PurePosixPath(self.fileUrl).name


class CheckRequest(BaseModel):
    date: List[str]
    modules: List[str]
    documentKeys: List[DocumentKey]
    source: str

    # Clean up documentKeys: drop nulls if present in the array
    @field_validator("documentKeys", mode="before")
    @classmethod
    def _filter_nulls(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [item for item in v if item is not None]
        return v


class ImageResponse(BaseModel):
    doc_key: str
    mime_type: str
    data_url: str  # e.g. data:image/png;base64,....
