"""Local filesystem storage provider implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.storage_providers.base import StorageSettings

logger = logging.getLogger(__name__)


class LocalStorageProvider:
    """Storage provider implementation using local filesystem.
    
    This provider stores application data in the local filesystem:
    
        {storage_root}/applications/{app_id}/files/{filename}
        {storage_root}/applications/{app_id}/metadata.json
        {storage_root}/applications/{app_id}/content_understanding.json
    """
    
    def __init__(
        self, 
        settings: StorageSettings,
        public_base_url: Optional[str] = None
    ) -> None:
        """Initialize the local storage provider.
        
        Args:
            settings: Storage settings containing local root path.
            public_base_url: Optional base URL for public file access.
        """
        self._settings = settings
        self._storage_root = Path(settings.local_root)
        self._public_base_url = public_base_url
        
        # Ensure storage root exists
        self._storage_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Initialized Local Storage provider at '%s'",
            self._storage_root.absolute()
        )
    
    def _get_application_dir(self, app_id: str) -> Path:
        """Get the directory path for an application.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            Path to the application directory.
        """
        app_dir = self._storage_root / "applications" / app_id
        app_dir.mkdir(parents=True, exist_ok=True)
        return app_dir
    
    def _get_files_dir(self, app_id: str) -> Path:
        """Get the files directory path for an application.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            Path to the application's files directory.
        """
        files_dir = self._get_application_dir(app_id) / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        return files_dir
    
    def save_file(self, app_id: str, filename: str, content: bytes) -> str:
        """Save a file to local filesystem.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            content: File content as bytes.
            
        Returns:
            Local file path where the file was saved.
        """
        files_dir = self._get_files_dir(app_id)
        file_path = files_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.debug("Saved file to local: %s", file_path)
        return str(file_path)
    
    def load_file(self, app_id: str, filename: str) -> Optional[bytes]:
        """Load a file from local filesystem.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            
        Returns:
            File content as bytes, or None if not found.
        """
        files_dir = self._get_files_dir(app_id)
        file_path = files_dir / filename
        
        if not file_path.exists():
            logger.debug("File not found: %s", file_path)
            return None
        
        with open(file_path, "rb") as f:
            return f.read()
    
    def save_metadata(self, app_id: str, metadata: Dict[str, Any]) -> None:
        """Save application metadata JSON to local filesystem.
        
        Args:
            app_id: Application identifier.
            metadata: Metadata dictionary to save.
        """
        app_dir = self._get_application_dir(app_id)
        meta_path = app_dir / "metadata.json"
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug("Saved metadata to local: %s", meta_path)
    
    def load_metadata(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Load application metadata from local filesystem.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            Metadata dictionary, or None if not found.
        """
        app_dir = self._storage_root / "applications" / app_id
        meta_path = app_dir / "metadata.json"
        
        if not meta_path.exists():
            logger.debug("Metadata not found: %s", meta_path)
            return None
        
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_applications(self, persona: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all applications from local filesystem, optionally filtered by persona.
        
        Args:
            persona: Optional persona filter.
            
        Returns:
            List of application summary dictionaries.
        """
        from app.personas import normalize_persona_id
        
        base = self._storage_root / "applications"
        if not base.exists():
            return []
        
        # Normalize the filter persona
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
            app_persona = data.get("persona") or "underwriting"
            app_persona = normalize_persona_id(app_persona)
            
            if persona is not None and app_persona != persona:
                continue
            
            apps.append({
                "id": data.get("id"),
                "created_at": data.get("created_at"),
                "external_reference": data.get("external_reference"),
                "status": data.get("status", "unknown"),
                "persona": app_persona,
                "summary_title": data.get("llm_outputs", {})
                    .get("application_summary", {})
                    .get("customer_profile", {})
                    .get("summary", "") or "",
            })
        
        # Sort by created_at descending
        apps.sort(key=lambda a: a.get("created_at") or "", reverse=True)
        return apps
    
    def save_cu_result(self, app_id: str, payload: Dict[str, Any]) -> str:
        """Save Content Understanding result JSON to local filesystem.
        
        Args:
            app_id: Application identifier.
            payload: Content Understanding result dictionary.
            
        Returns:
            Local file path where the result was saved.
        """
        app_dir = self._get_application_dir(app_id)
        cu_path = app_dir / "content_understanding.json"
        
        with open(cu_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        
        logger.debug("Saved CU result to local: %s", cu_path)
        return str(cu_path)
    
    def application_exists(self, app_id: str) -> bool:
        """Check if application exists in local filesystem.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            True if the application exists (has metadata), False otherwise.
        """
        app_dir = self._storage_root / "applications" / app_id
        meta_path = app_dir / "metadata.json"
        return meta_path.exists()
    
    def get_file_url(self, app_id: str, filename: str) -> Optional[str]:
        """Get the public URL for a file, if available.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            
        Returns:
            Public URL string, or None if not available.
        """
        files_dir = self._get_files_dir(app_id)
        file_path = files_dir / filename
        
        if not file_path.exists():
            return None
        
        if self._public_base_url:
            return f"{self._public_base_url.rstrip('/')}/applications/{app_id}/files/{filename}"
        
        return None
