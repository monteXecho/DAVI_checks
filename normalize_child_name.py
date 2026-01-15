import re
import difflib


def split_name(name):
    """
    Split a full name into surname and given name.
    Handles multiple commas, hyphenated surnames, and extra spaces.
    Input: 'Limpag-Oudshoorn, Roselyn'
    Output: ('Limpag-Oudshoorn', 'Roselyn')
    """
    # Split by commas and remove extra spaces
    parts = [p.strip() for p in name.split(",") if p.strip()]

    if not parts:
        return "", ""

    # Surname is the first part
    surname = parts[0]

    # Given name is everything else joined
    givenname = " ".join(parts[1:]).strip()

    return surname, givenname


def reorder_name(name):
    """
    Swap the position of surname and given name to 'Givenname Surname'.
    Input: 'Limpag-Oudshoorn, Roselyn'
    Output: 'Roselyn Limpag-Oudshoorn'
    """
    surname, givenname = split_name(name)
    return f"{givenname} {surname}".strip()


def normalize_name_child_register(children_registration):
    for date, records in children_registration.items():
        for record in records:
            record["name"] = reorder_name(record["name"])
    return children_registration


def fix_childname_grammar(name: str) -> str:
    # Strip leading/trailing spaces
    name = name.strip()

    # Replace dot with comma
    name = name.replace(".", ",")

    # Normalize commas → one comma + one space
    name = re.sub(r"\s*,\s*", ", ", name)

    # Replace multiple spaces with single space
    name = re.sub(r"\s+", " ", name)

    # Capitalize properly
    parts = []
    for part in name.split(" "):
        if len(part) == 2 and part[1] == ",":  # Handle initials like "R,"
            parts.append(part.upper())
        else:
            parts.append(part.capitalize())
    name = " ".join(parts)

    return name


def normalize_childrennames_in_list(names, cutoff=0.8):
    """
    Normalize names within a single list by fuzzy matching against
    the names already seen. Keeps duplicates and order.
    """
    normalized = []
    seen = []

    for n in names:
        n = fix_childname_grammar(n)
        # check if this name is close to an already "seen" one
        match = difflib.get_close_matches(n, seen, n=1, cutoff=cutoff)
        if match:
            normalized.append(match[0])  # use existing canonical
        else:
            normalized.append(n)
            seen.append(n)  # add as new canonical
    return normalized
