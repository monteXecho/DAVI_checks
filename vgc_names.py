import json
import os
from typing import Union
import re


def read_vgc_file(file_path: str) -> str:
    """
    Reads a VGC list file (PDF, DOCX, DOC, or TXT) and returns its text content as a single string.
    """
    ext = os.path.splitext(file_path)[1].lower()
    vgc_list = {}
    text = ""

    if ext == ".txt":
        # Simple TXT reading
        with open(
            f"documents/fixed-faces/{file_path}", "r", encoding="utf-8", errors="ignore"
        ) as f:
            text = f.read()
            vgc_list = convert_txt_2_json(text)

    elif ext == ".pdf":
        try:
            import fitz  # PyMuPDF

            with fitz.open(f"documents/fixed-faces/{file_path}") as pdf:
                text = "\n".join([page.get_text() for page in pdf])
                vgc_list = convert_txt_2_json(text)
        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF reading. Install via: pip install pymupdf"
            )

    elif ext == ".docx":
        try:
            import docx

            doc = docx.Document(f"documents/fixed-faces/{file_path}")
            text = "\n".join([para.text for para in doc.paragraphs])
            vgc_list = convert_txt_2_json(text)
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX reading. Install via: pip install python-docx"
            )

    elif ext == ".doc":
        # Use textract for .doc (binary Word) files
        try:
            import textract

            text = textract.process(f"documents/fixed-faces/{file_path}").decode(
                "utf-8", errors="ignore"
            )
            vgc_list = convert_txt_2_json(text)
        except ImportError:
            raise ImportError(
                "textract is required for .doc reading. Install via: pip install textract"
            )

    elif ext == ".json":
        # Use textract for .doc (binary Word) files
        try:
            with open(f"documents/fixed-faces/{file_path}", "r", encoding="utf-8") as f:
                vgc_list = json.load(f)
        except ImportError:
            raise ImportError(
                "textract is required for .doc reading. Install via: pip install textract"
            )
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Normalize spaces and line breaks

    return vgc_list


def convert_txt_2_json(text=""):
    clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    result = {}
    for line in clean_text.strip().splitlines():
        if "When this child" in line:
            continue  # skip header line
        parts = re.split(r"\t+", line.strip())
        if len(parts) < 2:
            continue
        child = parts[0].strip()
        staff = [name.strip() for name in parts[1].split("&")]
        result[child] = staff

    return result


def extract_vgc_names(vgc_list_paths=[]):
    """
    Given a VGC mapping {child: [staff, ...]}, return unique child and staff names.

    Args:
        vgc_list (dict): mapping of child -> list of staff

    Returns:
        dict: {"children": [...], "staff": [...]}
    """
    vgc_list = {}
    for file_path in vgc_list_paths:
        _vgc_list = read_vgc_file(file_path)
        for item in _vgc_list:
            vgc_list[item] = _vgc_list[item]

    children = set(vgc_list.keys())
    staff = set()

    for staff_list in vgc_list.values():
        staff.update(staff_list)

    return vgc_list, sorted(children), sorted(staff)
