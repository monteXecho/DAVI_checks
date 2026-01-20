from fastapi import FastAPI, UploadFile, Form, Query, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
import base64

from compliance_main import compliance_check
from extract_images_from_docx import first_docx_image_bytes
from models import PresignRequest, CheckRequest, ImageResponse
from validate_uploaded_file import validate_uploaded_file

from creating_vgc_list import create_vgc_list

# --------------------------------------------------------
# Global variables
# --------------------------------------------------------
from state import (
    UPLOAD_DIR,
    ALLOWED_DOCUMENT_TYPES,
    create_file_storage,
    scan_documents,
    load_check_results,
    update_check_results,
    read_check_result,
    get_file_info_by_key,
    read_check_results,
    list_check_results,
    store_file,
    load_vgc_list_check_results,
    update_vgc_list_check_results,
    read_vgc_list_check_result,
    list_vgc_list_check_results,
)


# --------------------------------------------------------
# Compliance / Check logic
# --------------------------------------------------------
def run_check(background_tasks, date_arr, modules, documents, source):
    check_id = str(uuid.uuid4())
    # unique_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(
        run_check_background, check_id, date_arr, modules, documents, source
    )
    # Initial running status
    update_check_results(
        check_id,
        "queued",
        0.0,
        summary=f"Controles gestart voor {modules} van {source}",  #   Checks started for {modules} of {source}
        date=date_arr,
        issues=None,
        references=documents,
        modules=modules,
        result=None,
    )
    return check_id


def run_check_background(check_id, date_arr, modules, documents, source):
    """Run compliance check in background safely."""

    # Run actual check
    _, result = compliance_check(check_id, date_arr, modules, documents, source)

    update_check_results(
        check_id,
        "completed",
        100,
        summary=f"Controles voltooid voor {modules} van {source}",  #   Checks completed for {modules} of {source}
        date=date_arr,
        issues=None,
        references=documents,
        modules=modules,
        result=result,
    )


# --------------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------------
app = FastAPI(title="Compliance Document API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------
# 1. Requirements Endpoint
# --------------------------------------------------------
@app.get(
    "/requirements",
    summary="Get required documents",
    description="Returns all required documents and existing files for the selected modules",
)
def get_requirements(
    modules: Optional[str] = Query(
        None, description="Comma-separated module names (e.g., bkr,vgc,threeHours)"
    ),
):
    required = ["child-planning", "staff-planning"]
    modules_set = set((modules or "").split(","))
    if "threeHours" in modules_set:
        required.append("child-registration")
    if "vgc" in modules_set:
        required.append("fixed-faces")

    result = {}
    for doc_type in required:
        folder = os.path.join(UPLOAD_DIR, doc_type)
        result[doc_type] = os.listdir(folder) if os.path.exists(folder) else []

    return {"requiredDocuments": result}


# --------------------------------------------------------
# 2. Upload / Presign Endpoint
# --------------------------------------------------------
@app.post(
    "/uploads/presign",
    summary="Get presigned upload URL",
    description="Returns a presigned URL and object key for uploading a file (simulated locally here)",
)
def presign_upload(req: PresignRequest):
    unique_id = str(uuid.uuid4())[:8]
    key = f"{req.document_type}/{unique_id}_{req.filename}"
    file_url = f"/files/{key}"
    return {"uploadUrl": file_url, "objectKey": key}


@app.post(
    "/upload",
    summary="Upload file directly",
    description="Upload a file directly to the server under the specified document type",
)
async def direct_upload(
    document_type: str = Form(
        ...,
        description="Document type (child-planning, staff-planning, child-registration, fixed-faces)",
    ),
    file: UploadFile = None,
):
    if document_type not in ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Ongeldig documenttype. Toegestaan: {ALLOWED_DOCUMENT_TYPES}",
        )
    if file is None:
        raise HTTPException(status_code=400, detail="Geen bestand geüpload.")

    unique_id = str(uuid.uuid4())[:8]
    filename = f"{unique_id}_{file.filename}"

    folder_path = os.path.join(UPLOAD_DIR, document_type)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    validate_uploaded_file(document_type, file_path)

    store_file(unique_id, document_type, filename)

    file_url = f"/documents/{document_type}/{filename}"
    return {"objectKey": unique_id, "fileUrl": file_url}


@app.post(
    "/uploads",
    summary="Upload files directly",
    description="Upload multiple files directly to the server under the specified document type",
)
async def direct_uploads(
    document_type: str = Form(
        ...,
        description="Document type (child-planning, staff-planning, child-registration, fixed-faces)",
    ),
    files: List[UploadFile] = None,
):
    if document_type not in ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document_type. Allowed: {ALLOWED_DOCUMENT_TYPES}",
        )
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    folder_path = os.path.join(UPLOAD_DIR, document_type)
    os.makedirs(folder_path, exist_ok=True)

    uploaded_files = []
    for file in files:
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{file.filename}"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        store_file(unique_id, document_type, filename)

        file_url = f"/documents/{document_type}/{filename}"
        uploaded_files.append({"objectKey": unique_id, "fileUrl": file_url})

    return {"uploadedFiles": uploaded_files}


# --------------------------------------------------------
# 3. Document Status Endpoint
# --------------------------------------------------------
@app.get(
    "/documents/status",
    summary="Check document status",
    description="Indicates which document keys are already uploaded",
)
def get_status(
    keys: str = Query(..., description="Comma-separated list of object keys"),
):
    status = {}
    key_list = keys.split(",")
    for key in key_list:
        status[key] = get_file_info_by_key(key)
    return {"status": status}


# --------------------------------------------------------
# Get the first image of a doc
# --------------------------------------------------------


@app.get("/documents/first-image", response_model=ImageResponse)
def get_first_image(
    doc_key: str = Query(..., description="DOCX key"),
):
    doc = get_file_info_by_key(doc_key)
    type = doc["type"]
    file = doc["file"]

    img_bytes, mime = first_docx_image_bytes(f"documents/{type}/{file}")
    if img_bytes is None:
        return ImageResponse(doc_key=doc_key, mime_type="", data_url="")

    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    return ImageResponse(doc_key=doc_key, mime_type=mime, data_url=data_url)


# --------------------------------------------------------
# 4. Start Checks Endpoint
# --------------------------------------------------------
@app.post(
    "/checks",
    summary="Start compliance check",
    description="Starts a run for the selected modules asynchronously and returns a checkId",
)
def start_checks(req: CheckRequest, background_tasks: BackgroundTasks):
    documents = []
    for doc_key in req.documentKeys:
        document = get_file_info_by_key(doc_key.objectKey)
        if document:
            documents.append(document)

    # Organize document images
    img_child_planning, img_child_registration, img_staff_planning, img_fixed_faces = (
        [],
        [],
        [],
        [],
    )
    for doc in documents:
        if doc["type"] == "child-planning":
            img_child_planning.append(doc["file"])
        if doc["type"] == "child-registration":
            img_child_registration.append(doc["file"])
        if doc["type"] == "staff-planning":
            img_staff_planning.append(doc["file"])
        if doc["type"] == "fixed-faces":
            img_fixed_faces.append(doc["file"])

    # Compliance rules
    missing = []
    if not img_child_planning:
        missing.append("child-planning")
    if not img_staff_planning:
        missing.append("staff-planning")
    if "threeHours" in req.modules and not img_child_registration:
        missing.append("child-registration")
    if "vgc" in req.modules and not img_fixed_faces:
        missing.append("fixed-faces")
    if missing:
        return {
            "status": "error",
            "missingDocuments": missing,
            "summary": f"Missing required documents for modules: {req.modules}",
        }

    check_id = run_check(
        background_tasks,
        req.date,
        req.modules,
        [
            img_child_planning,
            img_child_registration,
            img_staff_planning,
            img_fixed_faces,
        ],
        req.source,
    )

    return {
        "check_id": check_id,
        "date": req.date,
        "modules": req.modules,
        "group": "",
        "message": "Check started. Use GET /checks/{checkId} to get the result.",
    }


# --------------------------------------------------------
# 5. Get Checks Result Endpoint
# --------------------------------------------------------
@app.get(
    "/checks/list",
    description="Returns simplified check results",
)
def get_check_list():
    checks = list_check_results()
    return checks


@app.get(
    "/checks/{check_id}",
    summary="Get check result",
    description="Returns results of the check run, including summary, issues, and status",
)
def get_check_result(check_id: str):
    result = read_check_result(check_id)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Check not found"})
    return result


@app.get(
    "/checks",
    summary="Get checking ids",
    description="Returns check ids",
)
def get_check_ids():
    checks = read_check_results()
    check_ids = list(checks.keys())
    return check_ids


@app.get("/status/{check_id}")
def get_check_status(check_id: str) -> Dict[str, Any]:
    """
    Returns a flattened status payload for polling:
    { "status": "<message>", "progress": <0..100> }
    """
    entry = read_check_result(check_id)

    if not entry:
        raise HTTPException(status_code=404, detail="check_id not found")

    status = entry.get("status") or {}
    message = status.get("message", "unknown")
    progress = status.get("progress", 0)

    # Ensure sane types
    try:
        progress = int(progress)
    except Exception:
        progress = 0
    progress = max(0, min(100, progress))

    return {"status": message, "progress": progress}


# --------------------------------------------------------
# Creating VGC List
# --------------------------------------------------------


class ObjectKey(BaseModel):
    objectKey: str


class CreateVGCRequest(BaseModel):
    documentKeys: List[ObjectKey]
    source: Optional[str] = "childcare"
    group: Optional[str] = None


def run_create_vgc(
    background_tasks, documents: List[Dict[str, Any]], source: str, group: Optional[str]
):
    check_id = str(uuid.uuid4())
    background_tasks.add_task(
        run_create_vgc_background, check_id, documents, source, group
    )

    update_vgc_list_check_results(
        check_id,
        "queued",
        0.0,
        summary=f"VGC-lijst generatie gestart ({source})",
        date=[],
        issues=None,
        references=[[d.get("type", ""), d.get("file", "")] for d in documents],
        modules=["createVGC"],
        group=group or "",
        result=None,
    )
    return check_id


def run_create_vgc_background(
    check_id: str, documents: List[Dict[str, Any]], source: str, group: Optional[str]
):
    try:
        result = create_vgc_list(check_id, documents, update_vgc_list_check_results)

        # progress already reached 100 via the create_vgc_list() stage updates
        update_vgc_list_check_results(
            check_id,
            "completed",
            0,
            summary=f"VGC-lijst generatie voltooid ({source})",
            references=[[d.get("type", ""), d.get("file", "")] for d in documents],
            modules=["createVGC"],
            group=group or "",
            result=result,
        )
    except Exception as e:
        update_vgc_list_check_results(
            check_id,
            "error",
            0,
            summary=f"VGC-lijst generatie mislukt: {e}",
            issues=[{"error": str(e)}],
            references=[[d.get("type", ""), d.get("file", "")] for d in documents],
            modules=["createVGC"],
            group=group or "",
            result=None,
        )


@app.post("/create-vgc")
def start_create_vgc(req: CreateVGCRequest, background_tasks: BackgroundTasks):
    documents = []
    for doc_key in req.documentKeys:
        d = get_file_info_by_key(doc_key.objectKey)
        if d:
            documents.append(d)

    # require 3 doc types
    staff_ok = any(d["type"] == "staff-planning" for d in documents)
    planning_ok = any(d["type"] == "child-planning" for d in documents)
    reg_ok = any(d["type"] == "child-registration" for d in documents)

    missing = []
    if not staff_ok:
        missing.append("staff-planning")
    if not planning_ok:
        missing.append("child-planning")
    if not reg_ok:
        missing.append("child-registration")
    if missing:
        return {"status": "error", "missingDocuments": missing}

    check_id = run_create_vgc(background_tasks, documents, req.source, req.group)
    return {"check_id": check_id}


@app.get(
    "/checks-create-vgc/list",
    description="Returns simplified check results",
)
def get_check_VGC_Creating_list():
    checks = list_vgc_list_check_results()
    return checks


@app.get("/checks-create-vgc/{check_id}")
def get_create_vgc_result(check_id: str):
    result = read_vgc_list_check_result(check_id)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Check not found"})
    return result


# --------------------------------------------------------
# Root
# --------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello World"}


# --------------------------------------------------------
# Static files
# --------------------------------------------------------
# app.mount("/documents", StaticFiles(directory=UPLOAD_DIR), name="documents")

app.mount(
    "/documents",
    StaticFiles(directory=UPLOAD_DIR, check_dir=False),
    name="documents",
)


@app.on_event("startup")
async def startup_event():
    create_file_storage()
    scan_documents()
    load_check_results()
    load_vgc_list_check_results()
