import re
import difflib


def extract_names(texts):
    results = []
    for t in texts:
        # cut at first status word if present
        name = re.split(r"\bDien|\bDiens|\bDienst|\bHuishoudel|\bVerlof", t)[0].strip()
        # normalize spacing
        name = " ".join(name.split())
        # basic capitalization
        parts = name.split()

        _name = ""
        for part in parts:
            ch = part[0]
            _name += (
                part
                if _name == "" or not _name[-1].isalpha()
                else (
                    f" {part.capitalize()}" if ch.isalpha() and ch.isupper() else part
                )
            )
        results.append(_name)
    return results


def normalize_names_in_list(names, cutoff=0.8):
    """
    Normalize names within a single list by fuzzy matching against
    the names already seen. Keeps duplicates and order.
    """
    normalized = []
    seen = []

    for n in names:
        # check if this name is close to an already "seen" one
        match = difflib.get_close_matches(n, seen, n=1, cutoff=cutoff)
        if match:
            normalized.append(match[0])  # use existing canonical
        else:
            normalized.append(n)
            seen.append(n)  # add as new canonical
    return normalized
