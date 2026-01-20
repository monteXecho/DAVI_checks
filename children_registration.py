import os
import cv2
import numpy as np
import re
import json
import csv
import io
import pandas as pd
from datetime import datetime
from collections import defaultdict

from normalize_child_name import normalize_childrennames_in_list
from extract_images_from_docx import extract_images_from_docx
from state import update_check_results

# Geweest

# ---------- helpers ----------


def get_png_filenames(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    return png_files


def load_image_as_bgr(image_path, save_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    if save_path:
        cv2.imwrite(save_path, img)
    return img


def deg_to_cv_h(deg: float) -> int:
    return int(round((deg % 360) / 2.0))


def pct_to_cv(p: float) -> int:
    return int(round(np.clip(p, 0, 100) * 255 / 100.0))


def hsv_band(h_deg, s_low_pct, s_high_pct, v_low_pct, v_high_pct, h_pad_deg=6):
    h = deg_to_cv_h(h_deg)
    hp = deg_to_cv_h(h_pad_deg)
    hl = max(0, h - hp)
    hh = min(179, h + hp)
    sl = pct_to_cv(s_low_pct)
    sh = pct_to_cv(s_high_pct)
    vl = pct_to_cv(v_low_pct)
    vh = pct_to_cv(v_high_pct)
    return np.array([hl, sl, vl], np.uint8), np.array([hh, sh, vh], np.uint8)


# ---------- core detection: parameterized by color ----------


COLOR_PARAMS = {"group": dict(h=205, s_low=55.3, s_high=57.3, v_low=93.1, v_high=95.1)}


def build_color_masks(hsv, *, h_pad_deg=6):
    """Return raw (non-overlap-resolved) masks for each color."""
    masks = {}
    for color, p in COLOR_PARAMS.items():
        lo, hi = hsv_band(
            p["h"],
            p["s_low"],
            p["s_high"],
            p["v_low"],
            p["v_high"],
            h_pad_deg=h_pad_deg,
        )
        masks[color] = cv2.inRange(hsv, lo, hi)
    return masks


def postprocess_mask(mask, *, close_kernel=(9, 3), close_iters=1):
    """Fill tiny holes (text/gradient) but keep separators; no erosion to avoid cutting boxes."""
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones(close_kernel, np.uint8), iterations=close_iters
    )


def components_from_mask(mask, *, min_area_frac=0.001, extent_min=0.50):
    """Pixel-perfect components → list of dicts {x,y,w,h,area,extent}."""
    H, W = mask.shape[:2]
    min_area = max(1, int(min_area_frac * H * W))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    regions = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if w / h < 20:  # only accept width:height over 20
            continue
        if area < min_area:
            continue
        region = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
        extent = float(region.sum()) / float(w * h)
        if extent < extent_min:
            continue
        regions.append(dict(x=x, y=y, w=w, h=h))
    regions.sort(key=lambda r: (r["y"] // 6, r["x"]))
    return regions


# ---------- pipeline: detect all colors, CSV, crops, outlined ----------


def recognize_text_from_crop_paddleocr(ocr, image_np):
    result = ocr.predict(image_np)
    out = []
    for res in result:
        rec_texts = getattr(res, "rec_texts", None)
        if not rec_texts and isinstance(res, dict) and "rec_texts" in res:
            rec_texts = res["rec_texts"]
        if rec_texts:
            out.extend([t.strip() for t in rec_texts if t.strip()])
    return out


def process_image_all_colors_with_crop(
    img,
    *,
    h_pad_deg=6,
    close_kernel=(9, 3),
    close_iters=1,
    min_area_frac=0.001,
    extent_min=0.35,
):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Build per-color masks, resolve overlaps, then postprocess
    masks = build_color_masks(hsv, h_pad_deg=h_pad_deg)
    masks = {
        c: postprocess_mask(m, close_kernel=close_kernel, close_iters=close_iters)
        for c, m in masks.items()
    }
    # Get components per color
    boxes_by_color = {
        c: components_from_mask(m, min_area_frac=min_area_frac, extent_min=extent_min)
        for c, m in masks.items()
    }

    group_title_block = None
    if len(boxes_by_color["group"]) > 0:
        group_title_block = boxes_by_color["group"][0]

    return group_title_block


# ---------- run ----------


def get_date_block(img, x0=10 / 1920, y0=90 / 1080, x1=90 / 1920, y1=110 / 1080):
    H, W = img.shape[:2]
    _y0 = int(round((y0 * H)))
    _y1 = int(round((y1 * H)))
    _x0 = int(round((x0 * W)))
    _x1 = int(round((x1 * W)))
    return img[_y0:_y1, _x0:_x1]


def get_date_text(ocr, img):
    date_block = get_date_block(img)
    date_text = ""
    ocr_out = recognize_text_from_crop_paddleocr(ocr, date_block)
    if len(ocr_out) > 0:
        date_text = ocr_out[0]
    return date_text


def get_children_list_block(
    img, x0=0 / 1920, y0=480 / 1080, x1=1200 / 1920, y1=970 / 1080
):
    H, W = img.shape[:2]
    _y0 = int(round((y0 * H)))
    _y1 = int(round((y1 * H)))
    _x0 = int(round((x0 * W)))
    _x1 = int(round((x1 * W)))
    return img[_y0:_y1, _x0:_x1]


# VALID_STATUSES = {"Afwezig", "Geweest", "Geruild", "Binnen", "Onbekend"}
VALID_STATUSES = {"Geweest", "Binnen"}
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")


def get_children_list_text(ocr, img):
    date_block = get_children_list_block(img)
    ocr_out = recognize_text_from_crop_paddleocr(ocr, date_block)

    structured = []
    if len(ocr_out) > 0:
        for i in range(0, len(ocr_out), 5):
            if len(ocr_out) < i + 5:
                break  # Incomplete row, skip

            name = ocr_out[i].strip()
            group = ocr_out[i + 1].strip()
            from_time = ocr_out[i + 2].strip()
            to_time = ocr_out[i + 3].strip()
            status = ocr_out[i + 4].strip()

            # Validate each field
            if not name or not group:
                continue  # skip empty names or groups

            if not TIME_PATTERN.match(from_time) or not TIME_PATTERN.match(to_time):
                continue  # skip invalid time formats

            if status not in VALID_STATUSES:
                continue  # skip unknown status

            structured.append(
                {
                    "name": name,
                    # "group": group,
                    "from": from_time,
                    "to": to_time,
                    "status": status,
                }
            )
    return structured


def process_ocr(ocr, date_arr, img, *, ignore_date_filter: bool = False):
    # ----- get date

    date_text = get_date_text(ocr, img)
    print("child-registration", date_text)
    if not ignore_date_filter:
        print("same date", date_text not in date_arr)
        if date_text not in date_arr:
            return None, []

    # ----- get children list

    return date_text, get_children_list_text(ocr, img)


def get_structured_children_register(children_list):
    # --- Combine by date ---
    combined = defaultdict(lambda: {})

    for entry in children_list:
        date, records = entry
        for rec in records:
            name = rec["name"]
            combined[date][name] = rec  # last occurrence kept

    # --- Build final list ---
    result = []
    for date, info in combined.items():
        records = list(info.values())
        # names = [rec["name"] for rec in records]
        # _names = llm_request_correct_names(names) # todo  remove comment
        # for i in range(len(records)):
        #     records[i]["name"] = _names[i]

        names_texts = [item["name"] for item in records]

        names = normalize_childrennames_in_list(names_texts)
        for i in range(len(names)):
            records[i]["name"] = names[i]

        result.append(
            {
                "date": date,
                "records": records,
            }
        )

    return result


def children_registration_main_process(
    check_id,
    ocr,
    date_arr,
    files,
    type,
    *,
    ignore_date_filter: bool = False,
    update_fn=update_check_results,
):
    children_list = []
    total = 0
    if type == "docx":
        images = extract_images_from_docx(f"documents/child-registration/{files[0]}")
        total = len(images)
        for img in images:
            date_text, children = process_ocr(
                ocr, date_arr, img, ignore_date_filter=ignore_date_filter
            )
            if date_text:
                children_list.append((date_text, children))
            if update_fn:
                update_fn(check_id, "OCR personeelsplanning", float(30 / total))

    elif type == "xlsx":
        for file_name in files:
            xlsx_result = readXLSX(file_name)
            for data in xlsx_result:
                [date_text, children] = data
                children_list.append((date_text, children))
    else:
        total = len(files)
        for source_path in files:
            img = load_image_as_bgr(source_path)
            date_text, children = process_ocr(
                ocr, date_arr, img, ignore_date_filter=ignore_date_filter
            )
            if date_text:
                children_list.append((date_text, children))
            if update_fn:
                update_fn(check_id, "OCR personeelsplanning", float(30 / total))

    children_register = get_structured_children_register(children_list)

    with open("children_register.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(children_register, ensure_ascii=False, default=str) + "\n")

    return children_register


def children_registration_main_process_docx(
    ocr,
    datestr,
    docx,
    *,
    ignore_date_filter: bool = False,
):
    """get children registration time

    Args:
        ocr (PaddleOCR): PaddleOCR
        docx (arr): Path of docx file.

    Returns:
        dict: children_register
    """
    images = extract_images_from_docx(f"documents/child-registration/{docx[0]}")
    children_list = []
    for img in images:
        date_text, children = process_ocr(
            ocr, datestr, img, ignore_date_filter=ignore_date_filter
        )
        if date_text:
            children_list.append((date_text, children))
    children_register = get_structured_children_register(children_list)

    with open("children_register.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(children_register, ensure_ascii=False, default=str) + "\n")

    return children_register


#   -------   read xlsx file  --------------------------------
def extract_date_from_text(text):
    patterns = [
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",  # 2025-05-19 or 2025/5/19
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",  # 19-05-2025 or 05/19/2025
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1)
            for fmt in (
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%m-%d-%Y",
                "%m/%d/%Y",
            ):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%d-%m-%Y")
                except ValueError:
                    continue

    return None


def readXLSX(file_name):
    try:
        file_full_path = f"documents/child-registration/{file_name}"

        df_dict = pd.read_excel(file_full_path, sheet_name=None)

        date_text = extract_date_from_text(file_name)

        result = []

        def to_time(num):
            """Convert 813 -> 08:13"""
            num_str = str(num).zfill(4)
            return f"{num_str[:2]}:{num_str[2:]}"

        for sheet_name, sheet_df in df_dict.items():
            if date_text is None:
                date_text = extract_date_from_text(sheet_name)
            if date_text is None:
                # continue
                date_text = "common"
            part = sheet_df.to_csv(index=False, header=False)

            reader = csv.reader(io.StringIO(part))
            rows = []
            for row in reader:
                if not row or not row[0].strip():
                    continue
                try:
                    raw_name = row[0].strip().strip('"')
                    name = re.sub(r"\s+\d+$", "", raw_name)

                    start = int(row[2]) if row[2].strip().isdigit() else None
                    end = int(row[3]) if row[3].strip().isdigit() else None

                    if start is not None and end is not None:
                        rows.append(
                            {
                                "name": name,
                                "from": to_time(start),
                                "to": to_time(end),
                                "status": "Geweest",
                            }
                        )
                except IndexError:
                    continue

            result.append([date_text, rows])

        return result

    except ImportError:
        raise ImportError(
            "pandas is required for Excel reading. Install via: pip install pandas openpyxl xlrd"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file {file_name}: {e}")
