"""Azure Blob Storage provider implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContainerClient, ExponentialRetry

from app.storage_providers.base import StorageSettings

logger = logging.getLogger(__name__)


class AzureBlobStorageProvider:
    """Storage provider implementation using Azure Blob Storage.
    
    This provider stores application data in Azure Blob Storage containers,
    mirroring the local filesystem path structure:
    
        applications/{app_id}/files/{filename}
        applications/{app_id}/metadata.json
        applications/{app_id}/content_understanding.json
    """
    
    def __init__(self, settings: StorageSettings) -> None:
        """Initialize the Azure Blob Storage provider.
        
        Args:
            settings: Storage settings containing Azure credentials.
            
        Raises:
            ValueError: If Azure credentials are not configured.
        """
        if not settings.azure_account_name or not settings.azure_account_key:
            raise ValueError(
                "Azure Blob Storage requires AZURE_STORAGE_ACCOUNT_NAME and "
                "AZURE_STORAGE_ACCOUNT_KEY to be configured"
            )
        
        self._settings = settings
        self._container_name = settings.azure_container_name
        
        # Configure retry policy per research.md §2
        retry_policy = ExponentialRetry(
            initial_backoff=10,      # First retry after 10 seconds
            increment_base=4,        # Exponential factor
            retry_total=settings.azure_retry_total,
            random_jitter_range=3    # ±3s jitter to prevent thundering herd
        )
        
        # Create blob service client
        account_url = f"https://{settings.azure_account_name}.blob.core.windows.net"
        self._blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=settings.azure_account_key,
            retry_policy=retry_policy,
            connection_timeout=settings.azure_timeout_seconds,
            read_timeout=settings.azure_timeout_seconds,
        )
        
        # Container client (will create container on first use)
        self._container_client: Optional[ContainerClient] = None
        self._container_ensured = False
        
        logger.info(
            "Initialized Azure Blob Storage provider for account '%s', container '%s'",
            settings.azure_account_name,
            self._container_name
        )
    
    def _ensure_container(self) -> ContainerClient:
        """Ensure the container exists and return the container client.
        
        Creates the container if it doesn't exist (idempotent per research.md §4).
        
        Returns:
            ContainerClient for the configured container.
        """
        if self._container_client is None:
            self._container_client = self._blob_service_client.get_container_client(
                self._container_name
            )
        
        if not self._container_ensured:
            try:
                self._container_client.create_container()
                logger.info("Created container '%s'", self._container_name)
            except Exception as e:
                # Container may already exist - that's fine
                if "ContainerAlreadyExists" in str(e) or "409" in str(e):
                    logger.debug("Container '%s' already exists", self._container_name)
                else:
                    raise
            self._container_ensured = True
        
        return self._container_client
    
    def _get_blob_path(self, app_id: str, *path_parts: str) -> str:
        """Construct blob path for an application resource.
        
        Args:
            app_id: Application identifier.
            *path_parts: Additional path components.
            
        Returns:
            Blob path string.
        """
        parts = ["applications", app_id] + list(path_parts)
        return "/".join(parts)
    
    def save_file(self, app_id: str, filename: str, content: bytes) -> str:
        """Save a file to Azure Blob Storage.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            content: File content as bytes.
            
        Returns:
            Blob path where the file was saved.
        """
        container = self._ensure_container()
        blob_path = self._get_blob_path(app_id, "files", filename)
        
        blob_client = container.get_blob_client(blob_path)
        blob_client.upload_blob(content, overwrite=True)
        
        logger.debug("Saved file to blob: %s", blob_path)
        return blob_path
    
    def load_file(self, app_id: str, filename: str) -> Optional[bytes]:
        """Load a file from Azure Blob Storage.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            
        Returns:
            File content as bytes, or None if not found.
        """
        container = self._ensure_container()
        blob_path = self._get_blob_path(app_id, "files", filename)
        
        try:
            blob_client = container.get_blob_client(blob_path)
            download = blob_client.download_blob()
            return download.readall()
        except ResourceNotFoundError:
            logger.debug("File not found: %s", blob_path)
            return None
    
    def save_metadata(self, app_id: str, metadata: Dict[str, Any]) -> None:
        """Save application metadata JSON to Azure Blob Storage.
        
        Args:
            app_id: Application identifier.
            metadata: Metadata dictionary to save.
        """
        container = self._ensure_container()
        blob_path = self._get_blob_path(app_id, "metadata.json")
        
        content = json.dumps(metadata, indent=2).encode("utf-8")
        blob_client = container.get_blob_client(blob_path)
        blob_client.upload_blob(content, overwrite=True)
        
        logger.debug("Saved metadata to blob: %s", blob_path)
    
    def load_metadata(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Load application metadata from Azure Blob Storage.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            Metadata dictionary, or None if not found.
        """
        container = self._ensure_container()
        blob_path = self._get_blob_path(app_id, "metadata.json")
        
        try:
            blob_client = container.get_blob_client(blob_path)
            download = blob_client.download_blob()
            content = download.readall()
            return json.loads(content.decode("utf-8"))
        except ResourceNotFoundError:
            logger.debug("Metadata not found: %s", blob_path)
            return None
    
    def list_applications(self, persona: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all applications from Azure Blob Storage, optionally filtered by persona.
        
        Args:
            persona: Optional persona filter.
            
        Returns:
            List of application summary dictionaries.
        """
        from app.personas import normalize_persona_id
        
        container = self._ensure_container()
        
        # Normalize the filter persona
        if persona is not None:
            persona = normalize_persona_id(persona)
        
        apps: List[Dict[str, Any]] = []
        
        # List all metadata.json blobs under applications/
        prefix = "applications/"
        seen_app_ids = set()
        
        for blob in container.list_blobs(name_starts_with=prefix):
            # Extract app_id from path: applications/{app_id}/metadata.json
            parts = blob.name.split("/")
            if len(parts) >= 3 and parts[2] == "metadata.json":
                app_id = parts[1]
                if app_id in seen_app_ids:
                    continue
                seen_app_ids.add(app_id)
                
                # Load metadata
                metadata = self.load_metadata(app_id)
                if metadata is None:
                    continue
                
                # Filter by persona if specified
                app_persona = metadata.get("persona") or "underwriting"
                app_persona = normalize_persona_id(app_persona)
                
                if persona is not None and app_persona != persona:
                    continue
                
                apps.append({
                    "id": metadata.get("id"),
                    "created_at": metadata.get("created_at"),
                    "external_reference": metadata.get("external_reference"),
                    "status": metadata.get("status", "unknown"),
                    "persona": app_persona,
                    "summary_title": metadata.get("llm_outputs", {})
                        .get("application_summary", {})
                        .get("customer_profile", {})
                        .get("summary", "") or "",
                })
        
        # Sort by created_at descending
        apps.sort(key=lambda a: a.get("created_at") or "", reverse=True)
        return apps
    
    def save_cu_result(self, app_id: str, payload: Dict[str, Any]) -> str:
        """Save Content Understanding result JSON to Azure Blob Storage.
        
        Args:
            app_id: Application identifier.
            payload: Content Understanding result dictionary.
            
        Returns:
            Blob path where the result was saved.
        """
        container = self._ensure_container()
        blob_path = self._get_blob_path(app_id, "content_understanding.json")
        
        content = json.dumps(payload, indent=2).encode("utf-8")
        blob_client = container.get_blob_client(blob_path)
        blob_client.upload_blob(content, overwrite=True)
        
        logger.debug("Saved CU result to blob: %s", blob_path)
        return blob_path
    
    def application_exists(self, app_id: str) -> bool:
        """Check if application exists in Azure Blob Storage.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            True if the application exists (has metadata), False otherwise.
        """
        container = self._ensure_container()
        blob_path = self._get_blob_path(app_id, "metadata.json")
        
        blob_client = container.get_blob_client(blob_path)
        return blob_client.exists()
    
    def get_file_url(self, app_id: str, filename: str) -> Optional[str]:
        """Get the URL for a file in Azure Blob Storage.
        
        Note: This returns the blob URL but the blob is not publicly accessible
        unless the container has public access enabled.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            
        Returns:
            Blob URL string, or None if file doesn't exist.
        """
        container = self._ensure_container()
        blob_path = self._get_blob_path(app_id, "files", filename)
        
        blob_client = container.get_blob_client(blob_path)
        if blob_client.exists():
            return blob_client.url
        return None
