from datetime import datetime, timedelta
import re
import unicodedata
from difflib import SequenceMatcher


import math


def check_bkr(present_children, present_staff):
    """
    Check BKR compliance based on official Dutch childcare rules.
    Supports KDV (0-4 years) and BSO (4-12 years).
    """

    if not present_children:
        return 0, "Doorgang", "Geen kinderen aanwezig"

    # Split children by age bands
    ages = [c.get("age", 0) for c in present_children]
    A = sum(1 for a in ages if a < 1)  # 0-1
    B = sum(1 for a in ages if 1 <= a < 2)
    C = sum(1 for a in ages if 2 <= a < 3)
    D = sum(1 for a in ages if 3 <= a < 4)
    E = sum(1 for a in ages if 4 <= a < 7)
    F = sum(1 for a in ages if a >= 7)

    required_staff = 0

    # Case 1: Day care (0–4 years, KDV)
    if A + B + C + D > 0:
        # Formula 1
        formula1 = math.ceil(A / 3 + B / 5 + C / 6 + D / 8)
        # Formula 2 (baby rule)
        formula2 = math.ceil(A + (B + C + D) / 1.2) if A > 0 else 0
        required_staff = max(formula1, formula2)
        if required_staff < 1:
            required_staff = 1

    # Case 2: Out-of-school care (4–12 years, BSO)
    elif E + F > 0:
        value = E * 0.10 + F * 0.083
        frac = value - math.floor(value)
        if frac >= 0.18:
            required_staff = math.ceil(value)
        else:
            required_staff = math.floor(value)
        if required_staff < 1:
            required_staff = 1

    if not present_staff:
        return required_staff, "", ""

    actual_staff = len(present_staff)

    if actual_staff >= required_staff:
        return required_staff, "Ja", ""
    else:
        return (
            required_staff,
            "Nee",
            f"BKR mislukt: {actual_staff} personeel voor {len(present_children)} kinderen ({required_staff} nodig)",
        )


def check_three_uurs(day_results):
    total_minutes_deviation = 0
    details = []
    hard_violations = []

    for entry in day_results:
        children = entry["#Children"]
        staff = entry["#Staff"]
        required = entry["RequiredStaff"]

        if children == 0:
            continue

        if staff == 0:
            hard_violations.append(
                f"{entry['From Time']}-{entry['To Time']} (0 medewerkers met {children} kinderen)"
            )
            continue

        if staff < required:
            if staff >= (required / 2):
                start = datetime.strptime(entry["From Time"], "%H:%M")
                end = datetime.strptime(entry["To Time"], "%H:%M")
                minutes = int((end - start).total_seconds() / 60)
                total_minutes_deviation += minutes
                details.append(
                    f"{entry['From Time']}-{entry['To Time']} (personeel {staff}/{required})"
                )
            else:
                hard_violations.append(
                    f"{entry['From Time']}-{entry['To Time']} (personeel {staff}/{required}, <50%)"
                )

    summary = {
        "3-UURS": "Ja",
        "Reason": f"Totale afwijking {total_minutes_deviation} minuten binnen de toegestane 180 minuten",
        "Deviations": details,
    }

    if hard_violations:
        summary["3-UURS"] = "Nee"
        summary["Reason"] = "Er zijn harde overtredingen opgetreden die niet door 3-UURS gedekt kunnen worden"
        summary["HardViolations"] = hard_violations
    elif total_minutes_deviation > 180:
        summary["3-UURS"] = "Nee"
        summary["Reason"] = (
            f"Totale afwijking {total_minutes_deviation} minuten overschrijdt de toegestane 180 minuten"
        )

    return summary


# ---------------------------------------------------
#  VGC check helpers
# ---------------------------------------------------

_DUTCH_PARTICLES = {
    "van",
    "de",
    "der",
    "den",
    "het",
    "ten",
    "ter",
    "v/d",
    "v",
    "’t",
    "te",
}  # keep but don't overweight


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


# remove accents, punctuation -> spaces, collapse spaces, lowercase
def _normalize_name(s: str) -> tuple[str, list[str]]:
    s = _strip_accents(s)
    s = re.sub(r"[,_\-./]+", " ", s)  # punctuation to space
    s = re.sub(r"\s+", " ", s).strip().lower()
    tokens = s.split()
    return s, tokens


def _token_score(a_tokens: list[str], b_tokens: list[str]) -> float:
    a_set, b_set = set(a_tokens), set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union


def _sequence_score(a_norm: str, b_norm: str) -> float:
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _weighted_name_score(a: str, b: str) -> float:
    a_norm, a_tokens = _normalize_name(a)
    b_norm, b_tokens = _normalize_name(b)

    # Down-weight particles so they don't dominate matching
    a_core = [t for t in a_tokens if t not in _DUTCH_PARTICLES]
    b_core = [t for t in b_tokens if t not in _DUTCH_PARTICLES]

    jacc = _token_score(a_core, b_core)
    seq = _sequence_score(" ".join(sorted(a_core)), " ".join(sorted(b_core)))

    # extra credit if full tokens (incl. particles) also match
    jacc_full = _token_score(a_tokens, b_tokens)

    # blend; tuned to be order-agnostic but typo tolerant
    return 0.55 * jacc + 0.35 * seq + 0.10 * jacc_full


def _best_vgc_key(
    child_name: str, vgc_keys: list[str], threshold: float = 0.82
) -> tuple[str | None, float]:
    best_key, best_score = None, 0.0
    for key in vgc_keys:
        sc = _weighted_name_score(child_name, key)
        if sc > best_score:
            best_key, best_score = key, sc
    return (best_key if best_score >= threshold else None, best_score)


def _normalize_person(s: str) -> str:
    # For staff equality check (case/punct insensitive)
    s = _strip_accents(s)
    s = re.sub(r"[,_\-./]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


# --- VGC main check -----------------------------------------------------------


def check_vgc(present_children, present_staff, vgc_list):
    """
    VGC Rule: Every child must have an authorized staff present.
    Fuzzy matches child names to VGC keys (order/case/punct tolerant, small typos).
    Staff comparison is case/punct-insensitive exact match.
    """
    if len(present_children) == 0 or len(present_staff) == 0:
        return "Onbekend", []
    failed = []

    # Precompute normalized staff set for quick membership tests
    present_staff_norm = {_normalize_person(s["text"]) for s in present_staff}

    v_pass = True

    for c in present_children:
        child_name = c["name"]
        vgc_key, score = _best_vgc_key(child_name, list(vgc_list.keys()))

        if not vgc_key:
            # no confident match → treat as no rule found; skip (or mark as failed if you prefer)
            continue

        authorized_staff = vgc_list.get(vgc_key, [])
        if not authorized_staff:
            continue

        v_pass = False

        # normalize authorized staff too
        authorized_norm = {_normalize_person(x) for x in authorized_staff}

        # satisfied if any authorized staff is present
        if present_staff_norm.isdisjoint(authorized_norm):
            failed.append(f"VGC mislukt: {', '.join(authorized_staff)} voor {vgc_key}")

    if v_pass:  # if no rule -> not enforceable
        return "Doorgang", []

    if failed:
        return "Nee", failed
    return "Ja", []


# --- compliance check based time -----------------------------------------------


def compliance_time(
    child_register, staff_planning, vgc_list, day, modules, start_str, end_str
):
    """
    Compliance check for a specific time window [start_str, end_str) on a given day.
    Returns one dict with compliance results.
    """

    start_time = datetime.strptime(start_str, "%H:%M")
    end_time = datetime.strptime(end_str, "%H:%M")

    children = child_register.get(day, [])
    staff_for_day = [s for s in staff_planning if s["day"] == day]

    # Who is present in this interval?
    present_children = [
        c
        for c in children
        if datetime.strptime(c["from"], "%H:%M") < end_time
        and datetime.strptime(c["to"], "%H:%M") > start_time
    ]
    present_staff = [
        s
        for s in staff_for_day
        if datetime.strptime(s["from"], "%H:%M") < end_time  #   todo
        and datetime.strptime(s["to"], "%H:%M") > start_time
    ]

    details = []

    # Run checks
    required_staff, bkr, bkr_detail = check_bkr(present_children, present_staff)
    if bkr_detail:
        details.append(bkr_detail)

    res = {
        "Date": day,
        "From Time": start_time.strftime("%H:%M"),
        "To Time": end_time.strftime("%H:%M"),
        "#Children": len(present_children),
        "RequiredStaff": required_staff,
        "#Staff": len(present_staff),
        "BKR": bkr,
    }

    if "vgc" in modules:
        vgc, vgc_detail = check_vgc(present_children, present_staff, vgc_list)
        if vgc_detail:
            details += vgc_detail
        res["VGC"] = vgc

    res["Details"] = details

    return res


def compliance_day(
    child_register, staff_planning, vgc_list, day, modules, interval_minutes=15
):
    """
    Full-day compliance check from 06:00 to 19:00 in fixed intervals.
    Uses compliance_time internally.
    """
    slots = []
    current_time = datetime.strptime("06:00", "%H:%M")
    day_end = datetime.strptime("19:00", "%H:%M")

    while current_time < day_end:
        slot_end = current_time + timedelta(minutes=interval_minutes)
        slot = compliance_time(
            child_register,
            staff_planning,
            vgc_list,
            day,
            modules,
            current_time.strftime("%H:%M"),
            slot_end.strftime("%H:%M"),
        )
        slots.append(slot)
        current_time = slot_end

    res = {
        "day": day,
        "slices": slots,
    }

    if "threeHours" in modules:
        three_uurs_result = check_three_uurs(slots)

        # attach 3-UURS outcome to every slice
        for r in slots:
            r["3-UURS"] = three_uurs_result["3-UURS"]
        res["three_uurs_summary"] = three_uurs_result

    return res


def compliance_to_markdown(slices):
    """
    Convert a list of compliance slice dictionaries to a Markdown table.
    """
    if not slices:
        return "Geen gegevens beschikbaar."

    headers = [
        "Date",
        "Time Window",
        "#Children",
        "Required Staff",
        "#Staff Present",
        "BKR Compliant",
        "VGC Compliant",
        "3-UURS Compliant",
        "Notes",
    ]

    # Build header row
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "|---" * len(headers) + "|\n"

    # Build rows
    for s in slices:
        row = [
            s.get("Date", ""),
            f"{s.get('From Time', '')}-{s.get('To Time', '')}",
            str(s.get("#Children", "")),
            str(s.get("RequiredStaff", "")),
            str(s.get("#Staff", "")),
            s.get("BKR", ""),
            s.get("VGC", ""),
            s.get("3-UURS", ""),
            s.get("Details", ""),
        ]
        md_table += "| " + " | ".join(row) + " |\n"

    return md_table
