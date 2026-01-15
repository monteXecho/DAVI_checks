import numpy as np
import cv2
from collections import namedtuple
import datetime
import json
import math
import re

from helpers import any_date_in_week
from normalize_staff_name import extract_names, normalize_names_in_list
from extract_images_from_docx import extract_images_from_docx
from state import update_check_results

# ---------- images ----------


def load_image_as_bgr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


# ---------- Crop whit background -----------


def get_group_block(img, x0=20 / 1920, y0=230 / 1080, x1=330 / 1920, y1=960 / 1080):
    H, W = img.shape[:2]
    _y0 = int(round((y0 * H)))
    _y1 = int(round((y1 * H)))
    _x0 = int(round((x0 * W)))
    _x1 = int(round((x1 * W)))
    return img[_y0:_y1, _x0:_x1]


# ---------- Crop date region -----------


def crop_date_region(
    img, x0_rel=560 / 1920, x1_rel=780 / 1920, y0_rel=255 / 1080, y1_rel=275 / 1080
):
    H, W = img.shape[:2]

    # Convert relative to absolute pixel coords
    x0 = int(round(x0_rel * W))
    x1 = int(round(x1_rel * W))
    y0 = int(round(y0_rel * H))
    y1 = int(round(y1_rel * H))

    # Crop
    cropped = img[y0:y1, x0:x1]
    return cropped


def get_working_days(week_str):
    match = re.search(r"week\s*?(\d{1,2})\D*?(\d{4})", week_str, re.IGNORECASE)
    if not match:
        # raise ValueError(f"Could not parse week string: {week_str}")
        return None
    week_num = int(match.group(1))
    year = int(match.group(2))

    monday = datetime.date.fromisocalendar(year, week_num, 1)

    working_days = [
        (monday + datetime.timedelta(days=i)).strftime("%d-%m-%Y") for i in range(5)
    ]
    return working_days


# ---------- HSV detection (from your crop-detect-all-blocks.py) ----------


def detect_color_blocks_in_memory(image_bgr, crop_pad_px=2, crop_pad_py=2):
    Detection = namedtuple("Detection", ["color", "x", "y", "w", "h", "crop"])

    # HSV detection parameters (from original)
    COLOR_PARAMS = {
        "title": dict(h=215, s_low=26.0, s_high=27.0, v_low=87.0, v_high=88.0),
        "green": dict(h=90, s_low=26.0, s_high=49.8, v_low=89.0, v_high=100.0),
        # "blue": dict(h=180, s_low=53.9, s_high=98.7, v_low=92.2, v_high=99.6),
        # "yellow": dict(h=59, s_low=20.4, s_high=56, v_low=92.2, v_high=99.9),
    }

    def deg_to_cv_h(deg):
        return int(round((deg % 360) / 2.0))

    def pct_to_cv(p):
        return int(round(np.clip(p, 0, 100) * 255 / 100.0))

    def hsv_band(h_deg, s_low_pct, s_high_pct, v_low_pct, v_high_pct, h_pad_deg=6):
        h = deg_to_cv_h(h_deg)
        hp = deg_to_cv_h(h_pad_deg)
        return (
            np.array(
                [max(0, h - hp), pct_to_cv(s_low_pct), pct_to_cv(v_low_pct)], np.uint8
            ),
            np.array(
                [min(179, h + hp), pct_to_cv(s_high_pct), pct_to_cv(v_high_pct)],
                np.uint8,
            ),
        )

    H, W = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    def build_color_masks():
        masks = {}
        h_pad_deg = 6
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

    def resolve_overlaps(raw_masks):
        dil = {
            c: cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
            for c, m in raw_masks.items()
        }
        out = {}
        for c in raw_masks:
            others = None
            for k in raw_masks:
                if k == c:
                    continue
                others = dil[k] if others is None else cv2.bitwise_or(others, dil[k])
            out[c] = cv2.bitwise_and(raw_masks[c], cv2.bitwise_not(others))
        return out

    def postprocess_mask(mask, close_kernel=(9, 3), close_iters=1):
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones(close_kernel, np.uint8),
            iterations=close_iters,
        )

    def components_from_mask(mask, min_area_frac=0.001, extent_min=0.35):
        min_area = max(1, int(min_area_frac * H * W))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        regions = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < min_area:
                continue
            region = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
            extent = float(region.sum()) / float(w * h)
            if extent < extent_min:
                continue
            regions.append(
                dict(x=x, y=y, w=w, h=h, area=int(area), extent=float(extent))
            )
        regions.sort(key=lambda r: (r["y"] // 6, r["x"]))
        return regions

    detections = []
    raw = build_color_masks()
    masks = resolve_overlaps(raw)
    masks = {c: postprocess_mask(m) for c, m in masks.items()}

    for color, mask in masks.items():
        # if color == "green":
        regions = components_from_mask(mask)
        for reg in regions:
            x0 = max(0, reg["x"] - crop_pad_px)
            y0 = max(0, reg["y"] - crop_pad_px)
            x1 = min(W, reg["x"] + reg["w"] + crop_pad_py)
            y1 = min(H, reg["y"] + reg["h"] + crop_pad_py)
            crop = image_bgr[y0:y1, x0:x1]
            detections.append(
                Detection(
                    color=color,
                    x=reg["x"],
                    y=reg["y"],
                    w=reg["w"],
                    h=reg["h"],
                    crop=crop,
                )
            )

    return detections


# ---------- PaddleOCR using .predict() with in-memory image ----------


def recognize_text_from_crop_paddleocr(ocr, image_np):
    # result = ocr.predict(image_np)
    result = ocr.predict(image_np)
    out = []
    for res in result:
        rec_texts = getattr(res, "rec_texts", None)
        if not rec_texts and isinstance(res, dict) and "rec_texts" in res:
            rec_texts = res["rec_texts"]
        if rec_texts:
            out.extend([t.strip() for t in rec_texts if t.strip()])
    return " ".join(out)


def remove_leading_non_letters(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():  # keep from first letter
            return s[i:]
    return ""


# ---------- Calculate From/To time with Time bar ----------


def calculate_timestamp_time_bar(y, ini_y=610, interval=27.5, s="06:00"):
    h = int(s[:2])
    m = int(s[3:5])
    diff = y - ini_y
    _h = math.floor(diff / interval)
    _m_diff = diff - interval * _h
    _m_interval = interval / 8
    _m_pre = math.floor(_m_diff / _m_interval)
    _m = round((_m_pre + 1) / 2) * 15
    if _m == 60:
        _m = 0
        _h += 1
    return f"{(h + _h):02}:{(m + _m):02}"


# ---------- Master Orchestration ----------


def process_img_blocks_and_ocr(ocr, date_arr, img):
    final_blocks = []

    date_crop = crop_date_region(img)
    cv2.imwrite("test.png", date_crop)
    date_ocr = recognize_text_from_crop_paddleocr(ocr, date_crop)
    print("staff-planning", date_ocr)
    print("date exist", any_date_in_week(date_arr, date_ocr))
    if not any_date_in_week(date_arr, date_ocr):
        return []

    blocks = detect_color_blocks_in_memory(img)
    block_datas = []
    minW = 0
    maxW = 0
    for blk in blocks:
        minW = blk.x if minW == 0 else min(minW, blk.x)
        maxW = (blk.x + blk.w) if maxW == 0 else max(maxW, (blk.x + blk.w))
        if blk.color == "title":
            continue
        text = recognize_text_from_crop_paddleocr(ocr, blk.crop)
        text = remove_leading_non_letters(text)
        obj = {
            "x": blk.x,
            "y": blk.y,
            "w": blk.w,
            "h": blk.h,
            "color": blk.color,
            "text": text,
        }
        block_datas.append(obj)

        # if (blk.color == "title"):
        #     title_datas.append(obj)
        # else:
        #     block_datas.append(obj)

    gap = (maxW - minW) / 5
    days = get_working_days(date_ocr)
    title_datas = []
    for i in range(0, 5):
        obj = {
            "x": minW + gap * i,
            "y": 0,
            "w": gap,
            "h": 0,
            "color": "title",
            "text": days[i],
        }
        title_datas.append(obj)

    mockindex = 0  # fix   remove this
    for block in block_datas:
        for title in title_datas:
            minX = title["x"]
            maxX = title["x"] + title["w"]
            center = block["x"] + block["w"] / 2
            if center > minX and center < maxX:
                block["day"] = title["text"]
                block["date"] = date_ocr
                block["from"] = calculate_timestamp_time_bar(block["y"])
                block["to"] = calculate_timestamp_time_bar(block["y"] + block["h"])
                final_blocks.append(block)
                mockindex += 1
                break

    filtered = [
        {k: item[k] for k in ["color", "text", "day", "from", "to"]}
        for item in final_blocks
    ]

    # with open("staff_planning.txt", "w", encoding="utf-8") as f:
    # f.write(json.dumps(filtered, ensure_ascii=False, default=str) + "\n")
    return filtered


def staff_planning_aday(ocr, date_arr, img):
    filtered_data = process_img_blocks_and_ocr(ocr, date_arr, img)

    names_texts = [item["text"] for item in filtered_data]

    raw_names = extract_names(names_texts)
    names = normalize_names_in_list(raw_names)
    for i in range(len(names)):
        filtered_data[i]["text"] = names[i]

    return filtered_data


def staff_planning_main_process(check_id, ocr, date_arr, files, type):
    filtered_data = []
    total = 0
    if type == "docx":
        images = extract_images_from_docx(f"documents/staff-planning/{files[0]}")
        total = len(images)
        for img in images:
            res = staff_planning_aday(ocr, date_arr, img)
            filtered_data += res
            update_check_results(
                check_id, "OCR-personeelsplanning", float(30 / total)
            )  #   OCR staff-planning
    else:
        total = len(files)
        for img_path in files:
            img = load_image_as_bgr(img_path)
            res = staff_planning_aday(ocr, date_arr, img)
            filtered_data += res
            update_check_results(
                check_id, "OCR-personeelsplanning", float(30 / total)
            )  #   OCR staff-planning
    with open("staff-planning.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(filtered_data, ensure_ascii=False, default=str) + "\n")

    return filtered_data


def staff_planning_main_process_docx(ocr, datestr, docx):
    filtered_data = []
    images = extract_images_from_docx(f"documents/staff-planning/{docx[0]}")
    for img in images:
        res = staff_planning_aday(ocr, datestr, img)
        filtered_data += res
    with open("staff-planning.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(filtered_data, ensure_ascii=False, default=str) + "\n")

    return filtered_data
