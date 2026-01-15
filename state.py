import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json


# --------------------------------------------------------
# Constants
# --------------------------------------------------------
UPLOAD_DIR = "documents"

ALLOWED_DOCUMENT_TYPES = [
    "child-planning",
    "staff-planning",
    "child-registration",
    "fixed-faces",
]

DATA_FILE = "data.json"

# Global in-memory stores
CHECK_RESULTS: Dict[str, Dict[str, Any]] = {}
FILE_STORE: Dict[str, Dict[str, Any]] = {}

# Thread locks
CHECK_RESULTS_LOCK = threading.Lock()
FILE_STORE_LOCK = threading.Lock()

# --------------------------------------------------------
# Initialize variables
# --------------------------------------------------------


def load_check_results() -> None:
    """
    Load CHECK_RESULTS from data.json into memory.
    Creates the file if it does not exist.
    """
    global CHECK_RESULTS
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        CHECK_RESULTS = {}
        return

    with CHECK_RESULTS_LOCK:
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                CHECK_RESULTS = json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted, reset to empty dict
            CHECK_RESULTS = {}


# save data only when the compliance check is done
def save_check_results(data) -> None:
    """
    Save the current CHECK_RESULTS to data.json.
    """
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def update_check_results(
    check_id: str,
    status_message: Optional[str] = "",
    status_progress: Optional[float] = 0.0,
    *,
    summary: Optional[str] = None,
    group: Optional[str] = None,
    date: List[str] = None,
    issues: Optional[List[Any]] = None,
    references: Optional[List[List[str]]] = None,
    modules: Optional[List[str]] = None,
    result: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Thread-safe update of CHECK_RESULTS."""
    with CHECK_RESULTS_LOCK:
        entry = CHECK_RESULTS.get(
            check_id,
            {
                "summary": "",
                "date": [],
                "issues": None,
                "references": [],
                "status": {"message": "queued", "progress": 0},
                "modules": [],
                "group": "KDV RI Vrolijke Verkenners",
                "result": None,
            },
        )

        if summary is not None:
            entry["summary"] = summary
        if date is not None:
            entry["date"] = date
        if issues is not None:
            entry["issues"] = issues
        if references is not None:
            entry["references"] = references
        if modules is not None:
            entry["modules"] = modules
        if result is not None:
            entry["result"] = result
        if group is not None:
            entry["group"] = group

        total_score = 80
        if "threeHours" in entry["modules"]:
            total_score += 30
        if "vgc" in entry["modules"]:
            total_score += 10

        status = entry.get("status") or {"message": "queued", "progress": 0}
        cur_progress = float(status["progress"])
        if status_message is not None:
            status["message"] = status_message
        if status_progress is not None:
            try:
                p = status_progress / total_score * 100
            except Exception:
                p = 0.0
            status["progress"] = max(0, min(100, cur_progress + p))

        entry["status"] = status
        entry["updatedAt"] = datetime.utcnow().isoformat() + "Z"

        CHECK_RESULTS[check_id] = entry
        if status_message == "completed":
            save_check_results(CHECK_RESULTS)
        return dict(entry)


def read_check_results():
    with CHECK_RESULTS_LOCK:
        return CHECK_RESULTS


def read_check_result(check_id):
    with CHECK_RESULTS_LOCK:
        result = CHECK_RESULTS.get(
            check_id,
            {
                "summary": None,
                "date": None,
                "issues": None,
                "references": None,
                "status": {"message": "queued", "progress": 0},
                "modules": [],
                "group": "KDV RI Vrolijke Verkenners",
                "result": None,
            },
        )
        return result


def list_check_results():
    with CHECK_RESULTS_LOCK:
        results = []
        for check_id, data in CHECK_RESULTS.items():
            results.append(
                {
                    "check_id": check_id,
                    "date": data.get("date"),
                    "modules": data.get("modules", []),
                    "group": data.get("group"),
                }
            )
        return results


def create_file_storage():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for sub in ALLOWED_DOCUMENT_TYPES:
        os.makedirs(os.path.join(UPLOAD_DIR, sub), exist_ok=True)


def get_object_id(filename: str) -> str:
    """Extract the object id from a filename (prefix before first underscore)."""
    return filename.split("_", 1)[0]


def scan_documents():
    """Scan all files in /documents and populate FILE_STORE."""
    with FILE_STORE_LOCK:
        global FILE_STORE
        FILE_STORE = {}
        for doc_type in os.listdir(UPLOAD_DIR):
            folder_path = os.path.join(UPLOAD_DIR, doc_type)
            if not os.path.isdir(folder_path):
                continue
            for filename in os.listdir(folder_path):
                object_key = get_object_id(filename)
                FILE_STORE[object_key] = {"type": doc_type, "file": filename}


def get_file_info_by_key(object_key: str):
    """Return file info for a given object key."""
    with FILE_STORE_LOCK:
        return FILE_STORE.get(object_key, None)


def get_file_by_original_name(original_filename: str):
    """Return list of matches for the original filename."""
    results = []
    with FILE_STORE_LOCK:
        for object_key, info in FILE_STORE.items():
            parts = info["file"].split("_", 1)
            if len(parts) == 2 and parts[1] == original_filename:
                results.append({"objectKey": object_key, **info})
    return results


def store_file(unique_id, document_type, filename):
    with FILE_STORE_LOCK:
        FILE_STORE[unique_id] = {"type": document_type, "file": filename}
