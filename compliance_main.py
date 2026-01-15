from paddleocr import PaddleOCR
import difflib
import json
from datetime import datetime
import os

from vgc_names import extract_vgc_names
from children_planning import children_planning_main_process

from children_registration import children_registration_main_process
from staff_planning import staff_planning_main_process
from normalize_child_name import normalize_name_child_register
from compliance import compliance_day
from state import update_check_results


def get_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()

    if ext in (".docx", ".doc"):
        return "docx"
    elif ext == ".pdf":
        return "pdfx"
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"):
        return "imagex"
    if ext in (".xlsx", ".xls"):
        return "xlsx"
    else:
        return "unknown"


def convert_date_format(date_str: str) -> str:
    """
    Converts a date string from various common formats (YYYY-MM-DD, YYYY/MM/DD,
    DD.MM.YYYY, MM/DD/YYYY, etc.) to 'DD-MM-YYYY'.
    """
    # List of supported input formats
    formats = [
        "%Y-%m-%d",  # 2025-09-22
        "%Y/%m/%d",  # 2025/09/22
        "%d-%m-%Y",  # 22-09-2025
        "%d/%m/%Y",  # 22/09/2025
        "%d.%m.%Y",  # 22.09.2025
        "%m/%d/%Y",  # 09/22/2025 (US-style)
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            continue

    raise ValueError(f"Unsupported date format: {date_str}")


def correct_names(reference_names, noisy_names, cutoff=0.8):
    """
    Correct noisy_names by matching against reference_names using fuzzy matching.

    Parameters:
        reference_names (list[str]): List of correct names.
        noisy_names (list[str]): List of possibly incorrect names.
        cutoff (float): Similarity threshold (0.0 - 1.0).

    Returns:
        list[str]: Corrected names, keeping original order of noisy_names.
    """
    corrected = []
    for name in noisy_names:
        # Try fuzzy match
        match = difflib.get_close_matches(name, reference_names, n=1, cutoff=cutoff)
        if match:
            corrected.append(match[0])  # replace with correct version
        else:
            corrected.append(name)  # keep as-is if no match
    return corrected


def enrich_children_registration(_children, children_registration, cutoff=0.8):
    """
    Update children_registration with age from _children.

    - Matches records by name (exact or fuzzy).
    - Keeps order of children_registration intact.
    - Returns a NEW updated structure (doesn't modify input).

    Parameters:
        _children (list[dict]): [{"name": str, "age": int}, ...]
        children_registration (list[dict]): [{"date": str, "records": [...]}, ...]
        cutoff (float): similarity threshold for fuzzy matching (0.0 - 1.0).
    """

    # Build lookup dict for fast exact matches
    age_lookup = {child["name"]: child["age"] for child in _children}
    names_list = list(age_lookup.keys())

    res = {}
    for day in children_registration:
        _date = day["date"]
        res[_date] = []
        for record in day["records"]:
            name = record["name"]

            if name in age_lookup:
                record["age"] = age_lookup[name]
            else:
                # Fuzzy match if no exact match
                match = difflib.get_close_matches(name, names_list, n=1, cutoff=cutoff)
                if match:
                    record["age"] = age_lookup[match[0]]
                    record["name"] = match[0]  # normalize to correct spelling
                else:
                    # record["age"] = None  # not found
                    continue
            res[_date].append(record)

    return res


def expand_children_planning(children_planning_pre, children):
    """
    Replace indices in children_planning_pre with actual child objects from children.

    Args:
        children_planning_pre (dict): {date: set of indices}
        children (list): list of child dicts [{name, age}, ...]

    Returns:
        dict: {date: [child objects]}
    """
    updated = {}
    for date, indices in children_planning_pre.items():
        updated[date] = [children[i] for i in indices if 0 <= i < len(children)]

    with open("child-planning.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(updated, ensure_ascii=False, default=str) + "\n")
    return updated


def compliance_check(check_id, date_arr, modules, documents, source):
    if modules is None:
        modules = ["bkr"]
    print(f"--- {check_id} Started ---", datetime.now())
    update_check_results(check_id, "Activa voorbereiden...")  #   preparing assets

    # --- helpers ---
    def build_synthetic_child_register_for_day(
        planning_map, date_arr, default_from="06:00", default_to="19:00"
    ):
        """
        planning_map: children_planning_correct_name, keyed by day "DD-MM-YYYY",
                      values: list of {"name":..., "age":...}
        returns: dict { ddmmyyyy_day: [ {name, age, from, to}, ... ] }
        """
        res = {}
        for ddmmyyyy_day in date_arr:
            day_children = planning_map.get(ddmmyyyy_day, []) or []
            # attach full-day presence so compliance_time can compute presence
            synthetic = []
            for c in day_children:
                record = {}
                record["name"] = c.get("name")
                record["age"] = c.get("age")
                record["from"] = default_from
                record["to"] = default_to
                synthetic.append(record)

            res[ddmmyyyy_day] = synthetic
        return res

    # --- convert and unpack inputs ---
    date_arr = [convert_date_format(d) for d in date_arr]
    [
        img_child_planning,
        img_child_registration,
        img_staff_planning,
        img_fixed_faces,
    ] = documents

    print("modules =", modules, "| day =", date_arr)

    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False)

    # -------------------- VGC (names) --------------------
    # only compute if requested
    vgc_list = {}
    vgc_children_names = vgc_staff_names = None
    if "vgc" in modules:
        update_check_results(
            check_id, "Begonnen, VGC-lijst lezen", 10
        )  #   Started, Reading VGC List
        vgc_list, vgc_children_names, vgc_staff_names = extract_vgc_names(
            img_fixed_faces
        )
        update_check_results(check_id, "Lees VGC-lijst", 10)  #   Read VGC List
        print(f"--- {check_id} vgc_list read ---", datetime.now())
    else:
        update_check_results(
            check_id, "Begonnen, OCR-kinderplanning starten", 10
        )  #   Started, Starting OCR child-planning
        print(f"--- {check_id} skip vgc_list (module not selected) ---", datetime.now())

    # -------------------- Children planning (needed for BKR baseline & synthetic register) --------------------
    update_check_results(check_id, "OCR-kinderplanning")  #   OCR Child-Planning
    type = get_file_type(img_child_planning[0])
    children, children_planning_pre = children_planning_main_process(
        check_id, ocr, date_arr, img_child_planning, type
    )

    print(f"--- {check_id} children_planning finished ---", datetime.now())

    # Updated planning with ages; store for possible BKR-only path.
    children_planning_correct_name = expand_children_planning(
        children_planning_pre, children
    )
    update_check_results(check_id, "OCR Personeelsplanning")  #   OCR Staff-Planning

    # -------------------- Staff planning (always needed) --------------------
    type = get_file_type(img_staff_planning[0])
    staff_planning = staff_planning_main_process(
        check_id, ocr, date_arr, img_staff_planning, type
    )
    print(f"--- {check_id} staff_planning finished ---", datetime.now())

    # -------------------- Child registration (only if threeHours) --------------------
    # If 'threeHours' is selected, use the real registration (with precise from/to).
    # Otherwise, synthesize a lightweight full-day presence from planning for BKR/VGC checks.
    if "threeHours" in modules:
        update_check_results(
            check_id, "OCR-kinderregistratie"
        )  #   OCR Child registration
        type = get_file_type(img_child_registration[0])
        child_registration_raw = children_registration_main_process(
            check_id, ocr, date_arr, img_child_registration, type
        )
        print(f"--- {check_id} child_registration finished ---", datetime.now())

        # Enrich & normalize (adds age; normalizes names)
        _children_registration_with_age = enrich_children_registration(
            children, child_registration_raw
        )
        children_registration = normalize_name_child_register(
            _children_registration_with_age
        )
    else:
        print(
            f"--- {check_id} threeHours not selected → build synthetic registration ---",
            datetime.now(),
        )
        synthetic = build_synthetic_child_register_for_day(
            children_planning_correct_name, date_arr
        )
        # names may require normalization for consistency with staff/vgc matching
        children_registration = normalize_name_child_register(synthetic)

    with open("children_register_extended.json", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(children_registration, ensure_ascii=False, default=str) + "\n"
        )

    update_check_results(check_id, "Nalevingscontrole")  #   Compliance Checking

    result = []
    for day in date_arr:
        res = compliance_day(
            children_registration,
            staff_planning,
            vgc_list if "vgc" in modules else {},
            day,
            modules,
            interval_minutes=15,
        )
        result.append(res)

    with open("result.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    print(f"--- {check_id} finish ---", datetime.now())

    return check_id, result
