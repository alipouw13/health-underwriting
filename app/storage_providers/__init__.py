"""Storage providers package for WorkbenchIQ.

This package provides pluggable storage backends for application data.
Supported backends:
- local: Local filesystem storage (default)
- azure_blob: Azure Blob Storage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.storage_providers.base import StorageBackend, StorageProvider, StorageSettings

if TYPE_CHECKING:
    from app.config import Settings


# Global storage provider instance (initialized at startup)
_storage_provider: Optional[StorageProvider] = None


def get_storage_provider(
    settings: Optional["Settings"] = None,
    public_base_url: Optional[str] = None,
) -> StorageProvider:
    """Factory function to create or return the storage provider based on settings.
    
    Args:
        settings: Application settings containing storage configuration.
                  If None, returns the cached provider (must be initialized first).
        public_base_url: Optional base URL for public file access (local storage only).
        
    Returns:
        A storage provider instance matching the configured backend.
        
    Raises:
        ValueError: If the configured backend is not supported.
        RuntimeError: If settings is None and no provider has been initialized.
    """
    global _storage_provider
    
    # Return cached provider if no settings provided
    if settings is None:
        if _storage_provider is None:
            raise RuntimeError(
                "Storage provider not initialized. Call get_storage_provider(settings) first."
            )
        return _storage_provider
    
    storage_settings = settings.storage
    
    if storage_settings.backend == StorageBackend.LOCAL:
        from app.storage_providers.local import LocalStorageProvider
        _storage_provider = LocalStorageProvider(
            storage_settings,
            public_base_url=public_base_url or settings.app.public_files_base_url,
        )
    
    elif storage_settings.backend == StorageBackend.AZURE_BLOB:
        from app.storage_providers.azure_blob import AzureBlobStorageProvider
        _storage_provider = AzureBlobStorageProvider(storage_settings)
    
    else:
        raise ValueError(f"Unsupported storage backend: {storage_settings.backend}")
    
    return _storage_provider


def reset_storage_provider() -> None:
    """Reset the cached storage provider. Useful for testing."""
    global _storage_provider
    _storage_provider = None


__all__ = [
    "StorageBackend",
    "StorageProvider",
    "StorageSettings",
    "get_storage_provider",
    "reset_storage_provider",
]
