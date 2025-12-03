
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StoredFile:
    filename: str
    path: str
    url: Optional[str] = None


@dataclass
class ExtractedFieldConfidence:
    """Represents an extracted field with confidence score and source grounding."""
    field_name: str
    value: Any
    confidence: float
    page_number: Optional[int] = None
    bounding_box: Optional[List[float]] = None
    source_text: Optional[str] = None


@dataclass
class ConfidenceSummary:
    """Summary of confidence scores across all extracted fields."""
    total_fields: int = 0
    average_confidence: float = 0.0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    high_confidence_fields: List[Dict[str, Any]] = field(default_factory=list)
    medium_confidence_fields: List[Dict[str, Any]] = field(default_factory=list)
    low_confidence_fields: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ApplicationMetadata:
    id: str
    created_at: str
    external_reference: Optional[str]
    status: str
    files: List[StoredFile]
    persona: Optional[str] = None  # Persona this application belongs to
    document_markdown: Optional[str] = None
    markdown_pages: Optional[List[Dict[str, Any]]] = None
    cu_raw_result_path: Optional[str] = None
    llm_outputs: Optional[Dict[str, Any]] = None
    # New fields for confidence scoring
    extracted_fields: Optional[Dict[str, Any]] = None  # Raw extracted fields with confidence
    confidence_summary: Optional[Dict[str, Any]] = None  # Aggregated confidence statistics
    analyzer_id_used: Optional[str] = None  # Which analyzer was used for extraction


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_storage_root(root: str) -> Path:
    p = Path(root)
    _ensure_dir(p)
    return p


def get_application_dir(root: str, app_id: str) -> Path:
    base = get_storage_root(root) / "applications" / app_id
    _ensure_dir(base)
    return base


def save_uploaded_files(
    root: str,
    app_id: str,
    uploaded_files: List[Any],
    public_base_url: Optional[str] = None,
) -> List[StoredFile]:
    """Save uploaded files to disk and return metadata.
    
    Accepts dicts with 'name' and 'content' keys (from FastAPI after async read).
    """
    app_dir = get_application_dir(root, app_id)
    files_dir = app_dir / "files"
    _ensure_dir(files_dir)

    stored: List[StoredFile] = []

    for f in uploaded_files:
        # Handle dict format from FastAPI (pre-read content)
        if isinstance(f, dict):
            filename = f.get("name", f"upload-{len(stored)}.bin")
            data = f.get("content", b"")
        else:
            # Legacy support for objects with .name and .read()
            filename = getattr(f, "name", f"upload-{len(stored)}.bin")
            data = f.read() if hasattr(f, "read") else f
        
        target_path = files_dir / filename
        with open(target_path, "wb") as out:
            out.write(data)
        url = None
        if public_base_url:
            url = f"{public_base_url.rstrip('/')}/applications/{app_id}/files/{filename}"
        stored.append(StoredFile(filename=filename, path=str(target_path), url=url))

    return stored


def save_cu_raw_result(root: str, app_id: str, payload: Dict[str, Any]) -> str:
    app_dir = get_application_dir(root, app_id)
    cu_path = app_dir / "content_understanding.json"
    with open(cu_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(cu_path)


def save_application_metadata(root: str, metadata: ApplicationMetadata) -> None:
    app_dir = get_application_dir(root, metadata.id)
    meta_path = app_dir / "metadata.json"
    serializable = asdict(metadata)
    # Convert StoredFile dataclasses into plain dicts
    serializable["files"] = [asdict(f) for f in metadata.files]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_application(root: str, app_id: str) -> Optional[ApplicationMetadata]:
    app_dir = get_application_dir(root, app_id)
    meta_path = app_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    files = [StoredFile(**fd) for fd in data.get("files", [])]

    return ApplicationMetadata(
        id=data["id"],
        created_at=data.get("created_at"),
        external_reference=data.get("external_reference"),
        status=data.get("status", "unknown"),
        files=files,
        persona=data.get("persona"),
        document_markdown=data.get("document_markdown"),
        markdown_pages=data.get("markdown_pages"),
        cu_raw_result_path=data.get("cu_raw_result_path"),
        llm_outputs=data.get("llm_outputs"),
        extracted_fields=data.get("extracted_fields"),
        confidence_summary=data.get("confidence_summary"),
        analyzer_id_used=data.get("analyzer_id_used"),
    )


def list_applications(root: str, persona: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return lightweight list of available applications, optionally filtered by persona."""
    from app.personas import normalize_persona_id
    
    base = get_storage_root(root) / "applications"
    if not base.exists():
        return []

    # Normalize the filter persona (handles legacy 'claims' -> 'life_health_claims')
    if persona is not None:
        persona = normalize_persona_id(persona)

    apps: List[Dict[str, Any]] = []
    for app_dir in sorted(base.iterdir()):
        if not app_dir.is_dir():
            continue
        meta_path = app_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filter by persona if specified
        # Legacy apps without persona are treated as "underwriting"
        app_persona = data.get("persona") or "underwriting"
        # Normalize app persona as well (handles legacy 'claims' in stored data)
        app_persona = normalize_persona_id(app_persona)
        
        if persona is not None and app_persona != persona:
            continue
            
        apps.append(
            {
                "id": data.get("id"),
                "created_at": data.get("created_at"),
                "external_reference": data.get("external_reference"),
                "status": data.get("status", "unknown"),
                "persona": app_persona,
                "summary_title": data.get("llm_outputs", {})
                .get("application_summary", {})
                .get("customer_profile", {})
                .get("summary", "")
                or "",
            }
        )
    # Sort by created_at descending
    apps.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    return apps


def new_metadata(
    root: str,
    app_id: str,
    files: List[StoredFile],
    external_reference: Optional[str] = None,
    persona: Optional[str] = None,
) -> ApplicationMetadata:
    created_at = datetime.utcnow().isoformat() + "Z"
    md = ApplicationMetadata(
        id=app_id,
        created_at=created_at,
        external_reference=external_reference,
        status="pending",
        files=files,
        persona=persona,
        document_markdown=None,
        markdown_pages=None,
        cu_raw_result_path=None,
        llm_outputs={},
    )
    save_application_metadata(root, md)
    return md
