"""Base types and protocols for storage providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class StorageBackend(Enum):
    """Enumeration of supported storage backend types."""
    LOCAL = "local"
    AZURE_BLOB = "azure_blob"
    
    @classmethod
    def from_string(cls, value: str) -> "StorageBackend":
        """Parse a string value to StorageBackend enum.
        
        Args:
            value: String value (case-insensitive).
            
        Returns:
            Matching StorageBackend enum value.
            
        Raises:
            ValueError: If value doesn't match any backend.
        """
        normalized = value.lower().strip()
        for backend in cls:
            if backend.value == normalized:
                return backend
        valid_values = [b.value for b in cls]
        raise ValueError(
            f"Invalid storage backend '{value}'. "
            f"Valid options are: {', '.join(valid_values)}"
        )


@dataclass
class StorageSettings:
    """Configuration container for storage backend settings.
    
    Attributes:
        backend: Selected storage backend (local or azure_blob).
        local_root: Root directory for local storage.
        azure_account_name: Azure Storage account name.
        azure_account_key: Azure Storage account key.
        azure_container_name: Blob container name.
        azure_timeout_seconds: Per-operation timeout in seconds.
        azure_retry_total: Maximum retry attempts.
    """
    backend: StorageBackend = StorageBackend.LOCAL
    local_root: str = "data"
    azure_account_name: Optional[str] = None
    azure_account_key: Optional[str] = None
    azure_container_name: str = "workbenchiq-data"
    azure_timeout_seconds: int = 30
    azure_retry_total: int = 3
    
    def validate(self) -> List[str]:
        """Validate settings and return list of error messages.
        
        Returns:
            List of error messages. Empty if valid.
        """
        errors: List[str] = []
        
        if self.backend == StorageBackend.AZURE_BLOB:
            if not self.azure_account_name:
                errors.append(
                    "AZURE_STORAGE_ACCOUNT_NAME is required when STORAGE_BACKEND=azure_blob"
                )
            if not self.azure_account_key:
                errors.append(
                    "AZURE_STORAGE_ACCOUNT_KEY is required when STORAGE_BACKEND=azure_blob"
                )
            # Validate container name format (Azure requirements)
            if self.azure_container_name:
                name = self.azure_container_name
                if len(name) < 3 or len(name) > 63:
                    errors.append(
                        f"AZURE_STORAGE_CONTAINER_NAME must be 3-63 characters, got {len(name)}"
                    )
                if not name.islower() or not all(c.isalnum() or c == '-' for c in name):
                    errors.append(
                        "AZURE_STORAGE_CONTAINER_NAME must contain only lowercase letters, "
                        "numbers, and hyphens"
                    )
        
        if self.azure_timeout_seconds <= 0:
            errors.append("Azure timeout must be a positive integer")
        
        if self.azure_retry_total < 0:
            errors.append("Azure retry count must be non-negative")
        
        return errors


@runtime_checkable
class StorageProvider(Protocol):
    """Protocol defining storage operations for WorkbenchIQ.
    
    This protocol uses structural typing (duck typing) - any class implementing
    these methods can be used as a StorageProvider without explicit inheritance.
    """
    
    def save_file(self, app_id: str, filename: str, content: bytes) -> str:
        """Save a file and return the storage path/URL.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            content: File content as bytes.
            
        Returns:
            Storage path or URL where the file was saved.
        """
        ...
    
    def load_file(self, app_id: str, filename: str) -> Optional[bytes]:
        """Load file content.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            
        Returns:
            File content as bytes, or None if not found.
        """
        ...
    
    def save_metadata(self, app_id: str, metadata: Dict[str, Any]) -> None:
        """Save application metadata JSON.
        
        Args:
            app_id: Application identifier.
            metadata: Metadata dictionary to save.
        """
        ...
    
    def load_metadata(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Load application metadata.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            Metadata dictionary, or None if not found.
        """
        ...
    
    def list_applications(self, persona: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all applications, optionally filtered by persona.
        
        Args:
            persona: Optional persona filter.
            
        Returns:
            List of application summary dictionaries.
        """
        ...
    
    def save_cu_result(self, app_id: str, payload: Dict[str, Any]) -> str:
        """Save Content Understanding result JSON.
        
        Args:
            app_id: Application identifier.
            payload: Content Understanding result dictionary.
            
        Returns:
            Storage path where the result was saved.
        """
        ...
    
    def application_exists(self, app_id: str) -> bool:
        """Check if application exists.
        
        Args:
            app_id: Application identifier.
            
        Returns:
            True if the application exists, False otherwise.
        """
        ...
    
    def get_file_url(self, app_id: str, filename: str) -> Optional[str]:
        """Get the public URL for a file, if available.
        
        Args:
            app_id: Application identifier.
            filename: Name of the file.
            
        Returns:
            Public URL string, or None if not available.
        """
        ...
