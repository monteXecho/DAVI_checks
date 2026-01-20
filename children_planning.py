import numpy as np
import cv2
from collections import namedtuple
import unicodedata
import os
import re
from datetime import datetime, date
import json
import fitz

from helpers import any_date_in_week
from normalize_child_name import normalize_childrennames_in_list
from extract_images_from_docx import extract_images_from_docx
from state import update_check_results


COLOR_PARAMS = {
    "title": dict(  # for page ss
        h=194, s_low=86.8, s_high=90.8, v_low=89.0, v_high=93.0
    ),
    "purple": dict(h=234, s_low=16.1, s_high=20.1, v_low=82.7, v_high=86.7),
    "red": dict(h=348, s_low=30.8, s_high=34.8, v_low=70.9, v_high=74.9),
    "blue": dict(h=204, s_low=39.9, s_high=43.9, v_low=90.5, v_high=94.5),
}

COLOR_PARAMS_PDF = {
    "title": dict(h=249, s_low=25.3, s_high=45.3, v_low=90.0, v_high=100.0),
    "purple": dict(h=234, s_low=16.1, s_high=20.1, v_low=82.7, v_high=86.7),
    "red": dict(h=348, s_low=30.8, s_high=34.8, v_low=70.9, v_high=74.9),
    "blue": dict(h=204, s_low=39.9, s_high=43.9, v_low=90.5, v_high=94.5),
}


# ---------- PDF to in-memory images ----------


def pdf_to_images_in_memory(pdf_path, zoom=2):
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    images_bgr = []

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)

        h, w, n = pix.height, pix.width, pix.n
        buf = np.frombuffer(pix.samples, dtype=np.uint8)

        if n == 3:
            img_rgb = buf.reshape(h, w, 3)
        elif n == 4:
            img_rgba = buf.reshape(h, w, 4)
            rgb, a = img_rgba[..., :3], img_rgba[..., 3:4].astype(np.float32) / 255.0
            white = np.full_like(rgb, 255, dtype=np.uint8)
            img_rgb = (rgb * a + white * (1.0 - a)).astype(np.uint8)
        elif n == 1:
            gray = buf.reshape(h, w)
            img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            buf = np.frombuffer(pix.samples, dtype=np.uint8)
            img_rgb = buf.reshape(pix.height, pix.width, 3)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        images_bgr.append(img_bgr)

    doc.close()
    return images_bgr


# ---------- Crop whit background -----------


def crop_non_white_region(images, white_thresh=240):
    croppeds = []
    # i = 0
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = gray < white_thresh

        coords = np.argwhere(mask)

        if coords.size == 0:
            return img

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        cropped = img[y0:y1, x0:x1]

        croppeds.append(cropped)

        # i += 1
        # cv2.imwrite(f"cropped-{i}.png", cropped)

    return croppeds


# ----------  images ----------


def get_png_filenames(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    return png_files


def load_image_as_bgr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def get_group_block(img, x0=20 / 1920, y0=230 / 1080, x1=330 / 1920, y1=960 / 1080):
    H, W = img.shape[:2]
    _y0 = int(round((y0 * H)))
    _y1 = int(round((y1 * H)))
    _x0 = int(round((x0 * W)))
    _x1 = int(round((x1 * W)))
    return img[_y0:_y1, _x0:_x1]


def get_main_block(img, type):
    x0 = 0 / 6348 if type == "pdfx" else 370 / 1920
    y0 = 900 / 4396 if type == "pdfx" else 450 / 1080
    x1 = 1 if type == "pdfx" else 1900 / 1920
    y1 = 1 if type == "pdfx" else 970 / 1080
    return get_region(img, x0, y0, x1, y1)


def get_region(img, x0=0, y0=0, x1=0, y1=0):
    H, W = img.shape[:2]
    _y0 = int(round((y0 * H)))
    _y1 = int(round((y1 * H)))
    _x0 = int(round((x0 * W)))
    _x1 = int(round((x1 * W)))
    return img[_y0:_y1, _x0:_x1]


# ---------- Crop Users region -----------


def get_users_block(img, type):
    x0 = 0 if type == "pdfx" else 0
    x1 = 1489 / 6348 if type == "pdfx" else 310 / 1530
    y0 = 413 / 3496 if type == "pdfx" else 70 / 520
    y1 = 3313 / 3496 if type == "pdfx" else 520 / 520
    pad = 145 / 3496 if type == "pdfx" else 31 / 520
    H, W = img.shape[:2]

    _x0 = int(round((x0 * W)))
    _y0 = int(round((y0 * H)))
    _x1 = int(round((x1 * W)))
    _y1 = int(round((y1 * H)))
    _pad = int(round((pad * H)))
    return img[_y0:_y1, _x0:_x1], _y0, _y1, _x0, _x1, _pad


def extract_name_and_dob(user_data):
    try:
        parts_before_date = []
        dob = None

        for item in user_data:
            tokens = item.strip().split()
            for token in tokens:
                if re.match(r"^\d{2}[- ]\d{2}[- ]\d{4}$", token):
                    try:
                        normalized = re.sub(r"[ ]", "-", token)
                        datetime.strptime(normalized, "%d-%m-%Y")
                        dob = normalized
                    except ValueError:
                        pass
                    continue

                if token in ("M", "J"):
                    continue
                if token.isdigit():
                    continue

                parts_before_date.append(token)

        name = " ".join(parts_before_date).replace(",", "").strip()
        return (name, dob) if dob else None
    except Exception as e:
        print(f"Error request: {e}")
        return None


def crop_users_region_read(ocr, img, type):
    crop, y0, y1, x0, x1, pad = get_users_block(img, type)

    users = []
    i = 0
    while i * pad <= (y1 - y0):
        y0_ = int(i * pad)
        y1_ = int(min((y1 - y0), (i + 1) * pad))
        cropped_line = crop[y0_:y1_, x0:x1]
        result = ocr.predict(cropped_line)
        for res in result:
            rec_texts = getattr(res, "rec_texts", None)
            if not rec_texts and isinstance(res, dict) and "rec_texts" in res:
                rec_texts = res["rec_texts"]
            _len = len(rec_texts)
            if rec_texts and _len >= 2:
                result = extract_name_and_dob(rec_texts)
                if result:
                    name, dob = result
                    users.append(
                        {
                            "index": i,
                            "y0": y0 + y0_,
                            "y1": y0 + y1_,
                            "name": name,
                            # "gender": rec_texts[_len - 3].strip(),
                            # "left_month": rec_texts[_len - 2].strip(),
                            "dob": dob,
                        }
                    )
                    # cv2.imwrite(f"zzz-{name}-{i}.png", cropped_line)
        i += 1
    return users


# ---------- HSV detection (from your crop-detect-all-blocks.py) ----------


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


def detect_color_blocks_in_memory(
    color,
    img,
    hsv,
    type="docx",
    hue_pad_deg=6,
    close_kernel=(0, 3),
    close_iters=1,
    min_cc_area=3000,
):
    Detection = namedtuple(
        "Detection", ["block_id", "color", "x", "y", "w", "h", "crop"]
    )

    h_deg = COLOR_PARAMS_PDF[color]["h"] if type == "pdfx" else COLOR_PARAMS[color]["h"]
    s_low = (
        COLOR_PARAMS_PDF[color]["s_low"]
        if type == "pdfx"
        else COLOR_PARAMS[color]["s_low"]
    )
    s_high = (
        COLOR_PARAMS_PDF[color]["s_high"]
        if type == "pdfx"
        else COLOR_PARAMS[color]["s_high"]
    )
    v_low = (
        COLOR_PARAMS_PDF[color]["v_low"]
        if type == "pdfx"
        else COLOR_PARAMS[color]["v_low"]
    )
    v_high = (
        COLOR_PARAMS_PDF[color]["v_high"]
        if type == "pdfx"
        else COLOR_PARAMS[color]["v_high"]
    )

    # HSV → mask
    lo, hi = hsv_band(h_deg, s_low, s_high, v_low, v_high, h_pad_deg=hue_pad_deg)
    mask = cv2.inRange(hsv, lo, hi)

    # Clean small holes but avoid bridging (clamp kernel dims ≥1)
    kh = max(
        1,
        int(
            close_kernel[0] if isinstance(close_kernel, (list, tuple)) else close_kernel
        ),
    )
    kw = max(
        1,
        int(
            close_kernel[1] if isinstance(close_kernel, (list, tuple)) else close_kernel
        ),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8), iterations=int(close_iters)
    )

    # Connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # --- splitting knobs (tuned to ignore letters/numbers) ---
    EDGE_FRAC_FOR_LINE = 0.02  # row must have ≥2% columns with edge pixels
    # AND row must have ≤15% of mask coverage (near-empty row)
    MASK_EMPTY_FRAC = 0.15
    MIN_GAP_PX = 2  # separator thickness in rows
    BORDER_MARGIN_PX = 2  # don't cut within 2px of top/bottom of region
    MIN_SEG_H = 5  # minimum height for a kept subsegment
    SOBEL_KSIZE = 3
    GRAD_THRESH = 25

    boxes = []

    index = 0

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_cc_area or w <= 0 or h <= 0:
            continue

        region_bgr = img[y : y + h, x : x + w]
        region_mask = mask[y : y + h, x : x + w]  # color-mask in the same region

        # Horizontal-edge map (Sobel dy)
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=SOBEL_KSIZE)
        sobel_y = np.uint8(np.clip(np.abs(sobel_y), 0, 255))
        _, edge_bin = cv2.threshold(sobel_y, GRAD_THRESH, 255, cv2.THRESH_BINARY)
        # Remove tiny edge specks (keep long-ish horizontal lines)
        edge_bin = cv2.morphologyEx(
            edge_bin, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8), iterations=1
        )

        # Row projections
        # how many edge pixels in the row
        row_edge = (edge_bin > 0).sum(axis=1)
        # how many mask pixels (filled color) in the row
        row_fill = (region_mask > 0).sum(axis=1)

        need_edges = max(1, int(EDGE_FRAC_FOR_LINE * w))
        empty_max = int(MASK_EMPTY_FRAC * w)

        # A separator row must have: enough edge pixels AND be almost empty in the color mask
        sep = (row_edge >= need_edges) & (row_fill <= empty_max)

        # Don’t cut at region borders
        sep[:BORDER_MARGIN_PX] = False
        sep[-BORDER_MARGIN_PX:] = False

        # Turn separator runs into cut positions (midpoint of each run)
        cuts = []
        i = 0
        while i < h:
            if sep[i]:
                j = i
                while j < h and sep[j]:
                    j += 1
                if (j - i) >= MIN_GAP_PX:
                    cuts.append((i + j) // 2)
                i = j
            else:
                i += 1

        # Build spans between cuts
        ys = [0] + cuts + [h]
        for a, b in zip(ys[:-1], ys[1:]):
            if (b - a) < MIN_SEG_H:
                continue
            yy0, yy1 = y + a, y + b
            index += 1
            boxes.append(
                Detection(
                    block_id=index,
                    color=color,
                    x=x,
                    y=y,
                    w=w,
                    h=b - a,
                    crop=img[yy0:yy1, x : (x + w)],
                )
            )

    return boxes


# ---------- PaddleOCR using .predict() with in-memory image ----------


def recognize_text_from_crop_paddleocr(ocr, image_np):
    try:
        result = ocr.predict(image_np)
        out = []
        for res in result:
            rec_texts = getattr(res, "rec_texts", None)
            if not rec_texts and isinstance(res, dict) and "rec_texts" in res:
                rec_texts = res["rec_texts"]
            if rec_texts:
                out.extend([t.strip() for t in rec_texts if t.strip()])
        return " ".join(out)
    except Exception as e:
        print(f"Error request: {e}")
        return ""


# ---------- Master Orchestration ----------


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ---------- parse date ----------


def get_date_text(ocr, img, type):
    x0 = 320 / 6348 if type == "pdfx" else 380 / 1920
    y0 = 800 / 4396 if type == "pdfx" else 360 / 1080
    x1 = 950 / 6348 if type == "pdfx" else 600 / 1920
    y1 = 900 / 4396 if type == "pdfx" else 380 / 1080
    date_block = get_region(img, x0, y0, x1, y1)
    ocr_out = recognize_text_from_crop_paddleocr(ocr, date_block)
    return ocr_out


def parse_dutch_date(date_str: str) -> str:
    s = date_str.strip().lower()

    month_map = {
        "januari": 1,
        "februari": 2,
        "maart": 3,
        "april": 4,
        "mei": 5,
        "juni": 6,
        "juli": 7,
        "augustus": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "december": 12,
        "juf": 7,
        "jun": 6,
        "jul": 7,
        "sep": 9,
        "okt": 10,
        "nov": 11,
        "dec": 12,
    }

    # Case 1: Maandag 19-05-2025 or 19-05-2025
    m_date = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", s)
    if m_date:
        day, month, year = map(int, m_date.groups())
        return f"{day:02d}-{month:02d}-{year:04d}"

    # Case 2: Maandag, 19 Juli 2025 or Maandag, 19 juli 2025
    m1 = re.search(r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})", s)
    if m1:
        day = int(m1.group(1))
        month_name = m1.group(2).strip(".").lower()
        year = int(m1.group(3))
        month = month_map.get(month_name)
        if not month:
            raise ValueError(f"Unknown month name: {month_name}")
        return f"{day:02d}-{month:02d}-{year}"

    # Case 3: 19 Maandag, Week 21, 2025
    m2 = re.search(r"(\d{1,2})\s+\w+,\s*week\s+(\d+),\s*(\d{4})", s)
    if m2:
        day = int(m2.group(1))
        week = int(m2.group(2))
        year = int(m2.group(3))
        # Get Monday of that ISO week
        monday = date.fromisocalendar(year, week, 1)
        return f"{day:02d}-{monday.month:02d}-{year}"

    raise ValueError(f"Unrecognized format: {date_str}")


def get_age(dob_str: str, fmt: str = "%d-%m-%Y") -> int:
    try:
        dob = datetime.strptime(dob_str, fmt).date()
        today = date.today()

        age = today.year - dob.year
        # subtract one year if birthday not yet reached this year
        if (today.month, today.day) < (dob.month, dob.day):
            age -= 1

        return age
    except Exception as e:
        print(f"Error request: {e}")
        return 0


def process_img_blocks_and_ocr(
    ocr,
    date_arr,
    img,
    type,
    *,
    ignore_date_filter: bool = False,
):
    date_text = get_date_text(ocr, img, type)
    print("child-planning", date_text)
    if not ignore_date_filter:
        print("date exist", any_date_in_week(date_arr, date_text))
        if not any_date_in_week(date_arr, date_text):
            return []

    final_blocks = []
    titles = []
    # -----
    main_block_img = get_main_block(img, type)

    users = crop_users_region_read(ocr, main_block_img, type)
    hsv = cv2.cvtColor(main_block_img, cv2.COLOR_BGR2HSV)
    blocks = []

    for color in (
        ["title", "purple", "blue"]
        if ignore_date_filter
        else ["title", "purple", "red"]
    ):
        boxes = detect_color_blocks_in_memory(color, main_block_img, hsv, type)
        blocks += boxes

    title_datas = []
    block_datas = []

    for blk in blocks:
        text = (
            recognize_text_from_crop_paddleocr(ocr, blk.crop)
            if blk.color == "title"
            else ""
        )

        obj = {
            "x": blk.x,
            "y": blk.y,
            "w": blk.w,
            "h": blk.h,
            "color": blk.color,
            "text": normalize_text(text),
        }

        if blk.color == "title":
            if text not in titles:
                obj["text"] = parse_dutch_date(text)
                title_datas.append(obj)
                titles.append(text)
        else:
            block_datas.append(obj)

    for block in block_datas:
        _block = {}
        for title in title_datas:
            minX = title["x"]
            maxX = title["x"] + title["w"]
            centerX = block["x"] + block["w"] / 2
            if centerX > minX and centerX < maxX:
                _block["date"] = title["text"]
                break
        for user in users:
            minY = user["y0"]
            maxY = user["y1"]
            centerY = block["y"] + block["h"] / 2
            if centerY > minY and centerY < maxY:
                _block["name"] = user["name"]
                _block["dob"] = user["dob"]
                final_blocks.append(_block)
                break

    # -----
    return final_blocks


def children_planning_main_process(
    check_id,
    ocr,
    date_arr,
    files,
    type="docx",
    *,
    ignore_date_filter: bool = False,
    update_fn=update_check_results,
):
    _children_list = []
    total = 0
    if type == "docx":  # doc
        images = extract_images_from_docx(f"documents/child-planning/{files[0]}")
        total = len(images)
        for img in images:
            children = process_img_blocks_and_ocr(
                ocr, date_arr, img, type, ignore_date_filter=ignore_date_filter
            )
            _children_list += children
            if update_fn:
                update_fn(check_id, "OCR kinderplanning", float(30 / total))
    elif type == "pdfx":  #   pdf
        images = pdf_to_images_in_memory(f"documents/child-planning/{files[0]}")
        rois = crop_non_white_region(images)
        total = len(rois)
        for page_num, img in enumerate(rois, 1):
            children = process_img_blocks_and_ocr(
                ocr, date_arr, img, type, ignore_date_filter=ignore_date_filter
            )
            _children_list += children
            if update_fn:
                update_fn(check_id, "OCR kinderplanning", float(30 / total))
    else:  #   image
        total = len(files)
        for source_path in files:
            img = load_image_as_bgr(f"documents/child-planning/{source_path}")
            children = process_img_blocks_and_ocr(
                ocr, date_arr, img, type, ignore_date_filter=ignore_date_filter
            )
            _children_list += children
            if update_fn:
                update_fn(check_id, "OCR kinderplanning", float(30 / total))
    # end

    children = []
    children_planning_pre = {}
    name_index = 0
    for entry in _children_list:
        dob = entry.get("dob")
        if dob is None:
            continue
        _date = entry["date"]
        name = entry["name"]
        age = get_age(dob)

        current_name_index = name_index
        if not any(child["name"] == name for child in children):
            children.append(
                {
                    "name": name,
                    "age": age,
                }
            )
            name_index += 1
        else:
            current_name_index = next(
                (i for i, item in enumerate(children) if item["name"] == name), -1
            )

        if children_planning_pre.get(_date) is None:
            children_planning_pre[_date] = set()
        children_planning_pre[_date].add(current_name_index)

    names_texts = [item["name"] for item in children]

    names = normalize_childrennames_in_list(names_texts)
    for i in range(len(names)):
        children[i]["name"] = names[i]

    with open("children.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(children, ensure_ascii=False, default=str) + "\n")

    return children, children_planning_pre


def children_planning_main_process_docx(
    ocr,
    datestr,
    docx,
    *,
    ignore_date_filter: bool = False,
):
    images = extract_images_from_docx(f"documents/child-planning/{docx[0]}")

    _children_list = []
    for img in images:
        children = process_img_blocks_and_ocr(
            ocr, datestr, img, "docx", ignore_date_filter=ignore_date_filter
        )
        _children_list += children
    # end

    children = []
    children_planning_pre = {}
    name_index = 0
    for entry in _children_list:
        dob = entry.get("dob")
        if dob is None:
            continue
        _date = entry["date"]
        name = entry["name"]
        age = get_age(dob)

        current_name_index = name_index
        if not any(child["name"] == name for child in children):
            children.append(
                {
                    "name": name,
                    "age": age,
                }
            )
            name_index += 1
        else:
            current_name_index = next(
                (i for i, item in enumerate(children) if item["name"] == name), -1
            )

        if children_planning_pre.get(_date) is None:
            children_planning_pre[_date] = set()
        children_planning_pre[_date].add(current_name_index)

    names_texts = [item["name"] for item in children]

    names = normalize_childrennames_in_list(names_texts)
    for i in range(len(names)):
        children[i]["name"] = names[i]

    with open("children.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(children, ensure_ascii=False, default=str) + "\n")

    return children, children_planning_pre
