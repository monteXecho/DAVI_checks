import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from staff_planning import staff_planning_main_process
from children_planning import children_planning_main_process
from children_registration import children_registration_main_process


UpdateFn = Callable[..., Dict[str, Any]]


def _noop_update(*args, **kwargs) -> Dict[str, Any]:
    return {}


def _canon_name(name: str) -> str:
    """Normalize names so 'Dondorp, Eloise' and 'Dondorp Eloise' match better."""
    s = (name or "").strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_dt(day_ddmmyyyy: str, hhmm: str) -> datetime:
    return datetime.strptime(f"{day_ddmmyyyy} {hhmm}", "%d-%m-%Y %H:%M")


def _overlap_minutes(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0
    return int((end - start).total_seconds() // 60)


def _ext(filename: str) -> str:
    return (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()


def _guess_child_planning_type(filename: str) -> str:
    e = _ext(filename)
    if e == "docx":
        return "docx"
    if e == "pdf":
        return "pdfx"
    return "img"


def _guess_staff_planning_type(filename: str) -> str:
    e = _ext(filename)
    return "docx" if e == "docx" else "img"


def _guess_child_registration_type(filename: str) -> str:
    e = _ext(filename)
    if e == "docx":
        return "docx"
    if e in ("xlsx", "xls"):
        return "xlsx"
    return "img"


@dataclass
class PairStats:
    days: List = field(default_factory=list)
    minutes: int = 0


def _merge_child_planning(
    global_children_by_canon: Dict[str, Dict[str, Any]],
    global_planning_by_date: Dict[str, Dict[str, Dict[str, Any]]],
    children: List[Dict[str, Any]],
    planning_pre: Dict[str, Any],
) -> None:
    """children_planning_main_process returns (children, children_planning_pre).

    children_planning_pre maps date -> set(indices into children).
    We convert it to date -> {canon_name -> child_meta} and merge.
    """

    # Merge children master list
    for ch in children:
        cname = _canon_name(ch.get("name", ""))
        if not cname:
            continue
        if cname not in global_children_by_canon:
            global_children_by_canon[cname] = {
                "name": ch.get("name"),
                "age": ch.get("age"),
            }
        else:
            if (
                global_children_by_canon[cname].get("age") is None
                and ch.get("age") is not None
            ):
                global_children_by_canon[cname]["age"] = ch.get("age")

    # Merge day planning
    for day, idx_set in (planning_pre or {}).items():
        if day not in global_planning_by_date:
            global_planning_by_date[day] = {}
        # for idx in [
        #     int(x.strip()) for x in idx_set.strip("{}").split(",") if x.strip()
        # ]:
        for idx in list(idx_set):
            if idx < 0 or idx >= len(children):
                continue
            ch = children[idx]
            cname = _canon_name(ch.get("name", ""))
            if not cname:
                continue
            global_planning_by_date[day][cname] = {
                "name": ch.get("name"),
                "age": ch.get("age"),
            }


def _merge_child_registration(
    global_reg_by_date: Dict[str, Dict[str, Dict[str, Any]]],
    children_register: List[Dict[str, Any]],
) -> None:
    """Merge children_registration_main_process output.

    Output is: [{"date": "dd-mm-yyyy", "records": [{name, from, to, status}, ...]}, ...]
    We merge by (date, canon(name)).
    """

    for day_entry in children_register or []:
        day = day_entry.get("date")
        if not day:
            continue
        if day not in global_reg_by_date:
            global_reg_by_date[day] = {}

        for rec in day_entry.get("records", []) or []:
            cname = _canon_name(rec.get("name", ""))
            if not cname:
                continue
            global_reg_by_date[day][cname] = {
                "name": rec.get("name"),
                "from": rec.get("from"),
                "to": rec.get("to"),
                "status": rec.get("status"),
            }


def _registration_with_age(
    reg_by_date: Dict[str, Dict[str, Dict[str, Any]]],
    children_by_canon: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for day, recs in reg_by_date.items():
        out[day] = []
        for cname, rec in recs.items():
            age = None
            if cname in children_by_canon:
                age = children_by_canon[cname].get("age")
            out[day].append({**rec, "age": age})
        out[day].sort(key=lambda r: _canon_name(r.get("name", "")))
    return out


def _build_vgc_list(
    staff_planning: List[Dict[str, Any]],
    children_registration_ext: Dict[str, List[Dict[str, Any]]],
    child_planning: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    min_overlap_for_day: int = 15,
    k_under_1: int = 2,
    k_over_or_eq_1: int = 3,
    default_age: int = 3,
    allowed_statuses: Optional[Set[str]] = None,  # e.g. {"Geweest", "Aanwezig"}
) -> Dict[str, Any]:
    """
    Critical fix:
      - When using children_registration_ext["common"], apply it ONLY to children
        that are registered/planned that day in child_planning[day].
      - For those planned children, use the common record to supply from/to/status.

    Return:
      {"vgc_list": vgc_list, "pairs": pairs, "debug": debug}
    """

    # ----------------------------
    # Index staff shifts per day + staff lookup
    # ----------------------------
    staff_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    staff_lookup: Dict[str, str] = {}

    for s in staff_planning or []:
        day = s.get("day")
        staff_name = s.get("text") or ""
        if not day or not staff_name or not s.get("from") or not s.get("to"):
            continue
        staff_by_day[day].append(s)
        staff_lookup[_canon_name(staff_name)] = staff_name

    # ----------------------------
    # Prepare "common" map for fast lookup by canonical child name
    # ----------------------------
    common_list: List[Dict[str, Any]] = (children_registration_ext or {}).get(
        "common"
    ) or []
    common_by_child: Dict[str, Dict[str, Any]] = {}
    for rec in common_list:
        cname = _canon_name(rec.get("name") or "")
        if cname:
            common_by_child[cname] = rec

    # ----------------------------
    # Stats
    # ----------------------------
    child_days_present: Dict[str, List] = defaultdict(list)
    child_meta: Dict[str, Dict[str, Any]] = {}
    pair_stats: Dict[str, Dict[str, PairStats]] = defaultdict(
        lambda: defaultdict(PairStats)
    )

    counted_day_overlap: Set[Tuple[str, str, str]] = (
        set()
    )  # (child_key, staff_key, day)

    used_day_source: Dict[str, str] = {}  # day -> "exact" | "common" | "skipped"
    common_applied_children: Dict[str, List[str]] = defaultdict(
        list
    )  # day -> [child_keys]
    skipped_days: List[str] = []
    skipped_planned_children_missing_common: Dict[str, List[str]] = defaultdict(list)

    # Iterate across days we can actually compute overlaps for (days with staff)
    for day in sorted(staff_by_day.keys()):
        shifts = staff_by_day.get(day, [])
        if not shifts:
            continue

        day_regs_exact = (children_registration_ext or {}).get(day)
        if (
            day_regs_exact
            and isinstance(day_regs_exact, list)
            and len(day_regs_exact) > 0
        ):
            # Use exact day registration as-is
            used_day_source[day] = "exact"
            day_records = day_regs_exact

        else:
            # No exact registration => use "common" only for planned children on that day
            planned_for_day = (child_planning or {}).get(day) or {}
            if not planned_for_day:
                # no planned children => nothing to compute
                used_day_source[day] = "skipped"
                skipped_days.append(day)
                continue

            # if not common_by_child:
            #     used_day_source[day] = "skipped"
            #     skipped_days.append(day)
            #     continue

            day_records = []
            for child_key, plan_meta in planned_for_day.items():
                # child_key is already canonical in your sample (e.g. "zheng bo")
                ckey = (
                    child_key
                    if " " in child_key or child_key == child_key.lower()
                    else _canon_name(child_key)
                )
                if not ckey:
                    continue

                common_rec = common_by_child.get(ckey)
                if not common_rec:
                    common_rec = {"from": "08:00", "to": "17:00", "status": "Geweest"}
                    # skipped_planned_children_missing_common[day].append(ckey)
                    # continue

                # Build day-specific record with common times/status + planning display name/age
                day_records.append(
                    {
                        "name": plan_meta.get("name") or common_rec.get("name"),
                        "from": common_rec.get("from"),
                        "to": common_rec.get("to"),
                        "status": common_rec.get("status"),
                        # age: planning age first, then common age, then default
                        "age": (
                            plan_meta.get("age")
                            if plan_meta.get("age") is not None
                            else (
                                common_rec.get("age")
                                if common_rec.get("age") is not None
                                else default_age
                            )
                        ),
                    }
                )
                common_applied_children[day].append(ckey)

            if not day_records:
                used_day_source[day] = "skipped"
                skipped_days.append(day)
                continue

            used_day_source[day] = "common"

        # At this point, day_records is what we use for overlap computations
        for c in day_records:
            status = (c.get("status") or "").strip()
            if (
                allowed_statuses is not None
                and status
                and status not in allowed_statuses
            ):
                continue

            child_name = c.get("name") or ""
            child_key = _canon_name(child_name)
            if not child_key:
                continue

            if not c.get("from") or not c.get("to"):
                continue

            age = c.get("age")
            if age is None:
                age = default_age

            child_days_present[child_key].append(day)
            # store best meta (prefer non-null age)
            if child_key not in child_meta:
                child_meta[child_key] = {"name": child_name, "age": age}
            else:
                if child_meta[child_key].get("age") is None and age is not None:
                    child_meta[child_key]["age"] = age

            c_start = _parse_dt(day, c["from"])
            c_end = _parse_dt(day, c["to"])

            for s in shifts:
                staff_name = s.get("text") or ""
                staff_key = _canon_name(staff_name)
                if not staff_key:
                    continue

                s_start = _parse_dt(day, s["from"])
                s_end = _parse_dt(day, s["to"])

                mins = _overlap_minutes(c_start, c_end, s_start, s_end)
                if mins <= 0:
                    continue

                st = pair_stats[child_key][staff_key]
                st.minutes += mins

                if mins >= min_overlap_for_day:
                    key = (child_key, staff_key, day)
                    if key not in counted_day_overlap:
                        counted_day_overlap.add(key)
                        st.days.append(day)

    # ----------------------------
    # Build final VGC list (same return structure)
    # ----------------------------
    vgc_list: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []

    for child_key, staff_map in pair_stats.items():
        days_present = child_days_present.get(child_key, [])
        if len(days_present) <= 0:
            continue

        meta = child_meta.get(child_key, {"name": child_key, "age": default_age})
        age = meta.get("age")
        if age is None:
            age = default_age

        k = k_under_1 if (age is not None and age < 1) else k_over_or_eq_1

        ranked: List[Tuple[str, float, int, int]] = []
        for staff_key, st in staff_map.items():
            coverage = (len(st.days) / len(days_present)) if len(days_present) else 0.0
            ranked.append((staff_key, coverage, st.days, st.minutes))

        ranked.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)

        fixed_faces = []
        for staff_key, coverage, d, m in ranked[:k]:
            fixed_faces.append(
                {
                    "staff": staff_lookup.get(staff_key, staff_key),
                    "overlap_days": d,
                    "overlap_minutes": m,
                    "coverage": round(coverage, 4),
                }
            )
            pairs.append(
                {"child": meta["name"], "staff": staff_lookup.get(staff_key, staff_key)}
            )

        vgc_list.append(
            {
                "child": meta["name"],
                "age": age,
                "child_days_present": days_present,
                "fixed_faces": fixed_faces,
            }
        )

    vgc_list.sort(key=lambda x: _canon_name(x.get("child", "")))

    debug = {
        "child_days_present": dict(child_days_present),
        "pair_stats": {
            c: {
                s: {"days": st.days, "minutes": st.minutes}
                for s, st in staff_map.items()
            }
            for c, staff_map in pair_stats.items()
        },
        "registration_day_source": used_day_source,  # exact/common/skipped per day
        "common_applied_children": {d: v for d, v in common_applied_children.items()},
        "skipped_days_no_registration": skipped_days,
        "skipped_planned_children_missing_common": {
            d: v for d, v in skipped_planned_children_missing_common.items()
        },
        "default_age_used": default_age,
        "min_overlap_for_day": min_overlap_for_day,
    }

    return {"vgc_list": vgc_list, "pairs": pairs, "debug": debug}


def create_vgc_list(
    check_id: str,
    documents: List[Dict[str, Any]],
    update_fn: Optional[UpdateFn] = None,
) -> Dict[str, Any]:
    """OCR the 3 uploaded docs and generate a VGC list across all dates (no date filtering)."""

    update_fn = update_fn or _noop_update

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False)
    # ocr = None

    date_arr: List[str] = []

    staff_files: List[str] = []
    child_planning_files: List[str] = []
    child_registration_files: List[str] = []

    for d in documents or []:
        t = d.get("type")
        f = d.get("file")
        if not f:
            continue
        if t == "staff-planning":
            staff_files.append(f)
        elif t == "child-planning":
            child_planning_files.append(f)
        elif t == "child-registration":
            child_registration_files.append(f)

    # OCR staff planning
    update_fn(check_id, "OCR staff-planning", 0)

    staff_planning: List[Dict[str, Any]] = []
    for f in staff_files:
        t = _guess_staff_planning_type(f)
        if t == "docx":
            staff_planning.extend(
                staff_planning_main_process(
                    check_id,
                    ocr,
                    date_arr,
                    [f],
                    "docx",
                    ignore_date_filter=True,
                    update_fn=_noop_update,
                )
            )
        else:
            staff_planning.extend(
                staff_planning_main_process(
                    check_id,
                    ocr,
                    date_arr,
                    [f"documents/staff-planning/{f}"],
                    "img",
                    ignore_date_filter=True,
                    update_fn=_noop_update,
                )
            )
    # staff_planning = [
    #     {
    #         "color": "green",
    #         "text": "Jemima Joren",
    #         "day": "19-05-2025",
    #         "from": "07:30",
    #         "to": "17:00",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Jemima Joren",
    #         "day": "20-05-2025",
    #         "from": "07:30",
    #         "to": "17:00",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Samantha Hendriks",
    #         "day": "21-05-2025",
    #         "from": "07:30",
    #         "to": "17:00",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Kim Jonker",
    #         "day": "22-05-2025",
    #         "from": "07:30",
    #         "to": "17:00",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Celina Monsanto",
    #         "day": "23-05-2025",
    #         "from": "07:30",
    #         "to": "13:00",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Samantha",
    #         "day": "23-05-2025",
    #         "from": "07:30",
    #         "to": "10:45",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Amyvan Lieshout",
    #         "day": "19-05-2025",
    #         "from": "09:00",
    #         "to": "18:30",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Samantha Hendriks",
    #         "day": "20-05-2025",
    #         "from": "09:00",
    #         "to": "18:30",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Jemima Joren",
    #         "day": "21-05-2025",
    #         "from": "09:00",
    #         "to": "18:30",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Samantha Hendriks",
    #         "day": "22-05-2025",
    #         "from": "09:00",
    #         "to": "18:30",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Kim Jonker",
    #         "day": "23-05-2025",
    #         "from": "09:00",
    #         "to": "18:30",
    #     },
    #     {
    #         "color": "green",
    #         "text": "Sophie Vellekoop",
    #         "day": "23-05-2025",
    #         "from": "12:00",
    #         "to": "17:00",
    #     },
    # ]

    update_fn(check_id, "OCR staff-planning done", 25)

    # OCR child planning
    update_fn(check_id, "OCR child-planning", 0)

    children_by_canon: Dict[str, Dict[str, Any]] = {}
    planning_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for f in child_planning_files:
        t = _guess_child_planning_type(f)
        children, planning_pre = children_planning_main_process(
            check_id,
            ocr,
            date_arr,
            [f],
            t,
            ignore_date_filter=True,
            update_fn=_noop_update,
        )
        # planning_pre = {
        #     "20-05-2025": {0, 3, 4, 5, 6, 8, 15, 17, 18, 19},
        #     "23-05-2025": {0, 2, 7, 10, 11, 13, 15, 16, 18, 20},
        #     "19-05-2025": {1, 3, 4, 6, 9, 10, 12, 13, 15, 18, 19},
        #     "22-05-2025": {3, 5, 6, 8, 9, 14, 15, 16, 20},
        #     "21-05-2025": {6, 9, 10, 12, 14, 15, 18, 20},
        # }
        # children = [
        #     {"name": "Los Moos", "age": 4},
        #     {"name": "Zheng Bo", "age": 4},
        #     {"name": "Tuijl Van Jonas", "age": 3},
        #     {"name": "Dondorp Eloise", "age": 3},
        #     {"name": "Remmert Kaat", "age": 3},
        #     {"name": "Pallandt Van Fae", "age": 3},
        #     {"name": "Pagliero Abel", "age": 3},
        #     {"name": "Bolding Kate", "age": 3},
        #     {"name": "Hinrichs Reef", "age": 3},
        #     {"name": "Folmer Vieve", "age": 3},
        #     {"name": "Klijn De Brune", "age": 2},
        #     {"name": "Pothoven Hugo", "age": 2},
        #     {"name": "Jansen Luan", "age": 2},
        #     {"name": "Huber Alba", "age": 2},
        #     {"name": "Loon Vanbram", "age": 2},
        #     {"name": "Lecocq Elliot", "age": 2},
        #     {"name": "La Grouw Moos", "age": 2},
        #     {"name": "Brandsma Dani", "age": 1},
        #     {"name": "Graaf De Ida", "age": 1},
        #     {"name": "Veen Van Der Marius", "age": 1},
        #     {"name": "Aarts Vos Maevi", "age": 1},
        # ]

        _merge_child_planning(
            children_by_canon, planning_by_date, children, planning_pre
        )

    with open("children-planning.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(planning_by_date, ensure_ascii=False, default=str) + "\n")

    update_fn(check_id, "OCR child-planning done", 25)

    # OCR child registration
    update_fn(check_id, "OCR child-registration", 0)

    reg_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}

    reg_docx: List[str] = []
    reg_xlsx: List[str] = []
    reg_img: List[str] = []

    for f in child_registration_files:
        t = _guess_child_registration_type(f)
        if t == "docx":
            reg_docx.append(f)
        elif t == "xlsx":
            reg_xlsx.append(f)
        else:
            reg_img.append(f)

    for f in reg_docx:
        children_register = children_registration_main_process(
            check_id,
            ocr,
            date_arr,
            [f],
            "docx",
            ignore_date_filter=True,
            update_fn=_noop_update,
        )
        _merge_child_registration(reg_by_date, children_register)

    if reg_xlsx:
        children_register = children_registration_main_process(
            check_id,
            ocr,
            date_arr,
            reg_xlsx,
            "xlsx",
            ignore_date_filter=True,
            update_fn=_noop_update,
        )
        _merge_child_registration(reg_by_date, children_register)

    if reg_img:
        image_paths = [f"documents/child-registration/{f}" for f in reg_img]
        children_register = children_registration_main_process(
            check_id,
            ocr,
            date_arr,
            image_paths,
            "img",
            ignore_date_filter=True,
            update_fn=_noop_update,
        )
        _merge_child_registration(reg_by_date, children_register)

    update_fn(check_id, "OCR child-registration done", 20)

    # Generate VGC list
    update_fn(check_id, "Generating VGC list", 0)

    children_registration_ext = _registration_with_age(reg_by_date, children_by_canon)

    vgc_payload = _build_vgc_list(
        staff_planning,
        children_registration_ext,
        planning_by_date,
        min_overlap_for_day=15,
        k_under_1=2,
        k_over_or_eq_1=3,
    )

    vgc_payload["inputs"] = {
        "staff_planning_rows": len(staff_planning),
        "child_planning_children": len(children_by_canon),
        "child_planning_days": len(planning_by_date),
        "child_registration_days": len(children_registration_ext),
    }

    update_fn(check_id, "VGC list generated", 10)

    return vgc_payload
