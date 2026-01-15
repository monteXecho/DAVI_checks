import os
import zipfile
import mimetypes
import cv2
import numpy as np


def extract_images_from_docx(
    docx_path: str, out_dir: str | None = None
) -> list[np.ndarray]:
    """
    Extracts all images from a .docx by pulling files from word/media/,
    optionally saves them into out_dir, and returns them as OpenCV BGR images.
    """
    images = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(docx_path) as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
        for member in media_files:
            # read raw bytes directly from ZIP
            data = zf.read(member)
            img_array = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode image: {member}")
            images.append(img)

            # optional: save to disk if out_dir specified
            if out_dir:
                fname = os.path.basename(member)
                out_path = os.path.join(out_dir, fname)
                with open(out_path, "wb") as f:
                    f.write(data)

    return images


def first_docx_image_bytes(docx_path: str) -> tuple[bytes, str]:
    """
    Returns (image_bytes, mime_type) for the first image found in word/media/*.
    Raises HTTPException if none.
    """
    try:
        with zipfile.ZipFile(docx_path) as zf:
            media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
            if not media_files:
                # raise HTTPException(status_code=404, detail="No images found in DOCX.")
                print("No images found in DOCX.")
                return None, None
            first = media_files[0]
            ext = os.path.splitext(first)[1].lower()
            mime, _ = mimetypes.guess_type(
                "x" + ext
            )  # prefix to keep behavior consistent
            if mime is None:
                # Fallbacks for common Office image extensions
                mime = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".bmp": "image/bmp",
                    ".tif": "image/tiff",
                    ".tiff": "image/tiff",
                    ".wmf": "image/wmf",
                    ".emf": "image/emf",
                    ".svg": "image/svg+xml",
                }.get(ext, "application/octet-stream")
            with zf.open(first) as imgf:
                data = imgf.read()
                return data, mime
    except zipfile.BadZipFile:
        # raise HTTPException(
        #     status_code=400, detail="File is not a valid .docx (ZIP) archive."
        # )
        print("File is not a valid .docx (ZIP) archive.")
        return None, None
    except FileNotFoundError:
        # raise HTTPException(status_code=404, detail="DOCX file not found.")
        print("DOCX file not found.")
        return None, None


def safe_join(base: str, *paths: str) -> str:
    # prevent path traversal
    final = os.path.abspath(os.path.join(base, *paths))
    if not final.startswith(
        os.path.abspath(base) + os.sep
    ) and final != os.path.abspath(base):
        # raise HTTPException(status_code=400, detail="Invalid path.")
        return None
    return final
