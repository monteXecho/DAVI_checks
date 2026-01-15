import os
import mimetypes
import zipfile
import fitz
import cv2
import pandas as pd
from fastapi import HTTPException

ALLOWED_EXTENSIONS = {
    "child-planning": [".docx", ".pdf", ".png"],
    "staff-planning": [".docx", ".pdf", ".png"],
    "child-registration": [".docx", ".pdf", ".png", ".xlsx"],
    "fixed-faces": [".pdf", ".docx", ".xlsx", ".txt", ".json"],
}


def get_mime_type(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


def validate_uploaded_file(document_type: str, file_path: str):
    """
    Lightweight file validation before OCR:
    1. File type validation - checks extension & MIME
    2. Content validation - confirms file contains images
    """

    # ---------- 1. File type validation ----------
    ext = os.path.splitext(file_path)[1].lower()
    if (
        document_type not in ALLOWED_EXTENSIONS
        or ext not in ALLOWED_EXTENSIONS[document_type]
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Ongeldig bestandstype '{ext}' voor {document_type}",
        )

    mime = get_mime_type(file_path)
    if not any(k in mime for k in ["pdf", "word", "image", "excel", "text", "json", "sheet"]):
        raise HTTPException(status_code=400, detail="Niet-ondersteund FILE-type")

    # ---------- 2. Content validation ----------
    if ext == ".pdf":
        doc = fitz.open(file_path)
        has_image = any(len(page.get_images(full=True)) > 0 for page in doc)
        if not has_image and (
            document_type == "staff-planning" or document_type == "child-registration"
        ):
            raise HTTPException(
                status_code=400, detail="PDF heeft geen ingesloten afbeeldingen"
            )

    elif ext == ".docx":
        with zipfile.ZipFile(file_path, "r") as z:
            image_files = [f for f in z.namelist() if f.startswith("word/media/")]
            if not image_files and document_type != "fixed-faces":
                raise HTTPException(
                    status_code=400,
                    detail="Word-bestand heeft geen ingesloten afbeeldingen",
                )

    elif ext in [".xlsx", ".xls"]:
        try:
            _ = pd.read_excel(file_path, nrows=1)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Ongeldig Excel-bestandsformaat"
            )

    elif ext == ".png":
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Ongeldig afbeeldingsbestand")

    return True
