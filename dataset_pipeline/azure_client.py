"""
Azure Blob Storage client and utilities

Handles authentication, hierarchical listing, and blob downloads for the pipeline.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Iterator
from azure.storage.blob import ContainerClient, BlobClient
from azure.identity import ClientSecretCredential
from azure.core.exceptions import AzureError, ResourceNotFoundError

from .config import AzureConfig

logger = logging.getLogger(__name__)


class AzureBlobClient:
    """
    Client for interacting with Azure Blob Storage.

    Handles authentication and provides methods for listing and downloading blobs.
    """

    def __init__(self, config: AzureConfig):
        """
        Initialize Azure Blob Storage client.

        Args:
            config: Azure configuration with credentials and storage URL
        """
        self.config = config

        # Create credential
        self.credential = ClientSecretCredential(
            tenant_id=config.tenant_id,
            client_id=config.client_id,
            client_secret=config.client_secret
        )

        # Create container client
        self.container_client = ContainerClient(
            account_url=config.storage_url,
            container_name=config.container_name,
            credential=self.credential
        )

        logger.info(
            f"Initialized Azure Blob client for container: {config.container_name}"
        )

    def list_prefixes(
        self,
        prefix: str = "",
        delimiter: str = "/"
    ) -> List[str]:
        """
        List virtual directories at one level using hierarchical listing.

        Uses Azure Blob Storage's delimiter feature to efficiently discover
        the directory structure without listing all blobs.

        Args:
            prefix: Prefix to filter results (e.g., "GRADINA/U/")
            delimiter: Delimiter for hierarchical listing (default: "/")

        Returns:
            List of directory prefixes (virtual directories)

        Examples:
            >>> client = AzureBlobClient(config)
            >>> borders = client.list_prefixes(delimiter="/")
            >>> # Returns: ["GRADINA/", "KELEBIJA/", "DJALA/", ...]
            >>> directions = client.list_prefixes(prefix="GRADINA/", delimiter="/")
            >>> # Returns: ["GRADINA/U/", "GRADINA/I/"]
        """
        prefixes = []

        try:
            blob_list = self.container_client.walk_blobs(
                name_starts_with=prefix,
                delimiter=delimiter
            )

            for item in blob_list:
                # Check if it's a virtual directory (has prefix attribute)
                if hasattr(item, "prefix"):
                    prefixes.append(item.prefix)

            logger.debug(
                f"Found {len(prefixes)} prefixes with prefix='{prefix}', "
                f"delimiter='{delimiter}'"
            )

        except AzureError as e:
            logger.error(f"Error listing prefixes: {e}")
            raise

        return prefixes

    def list_blobs(
        self,
        prefix: str = "",
        max_results: Optional[int] = None
    ) -> List[str]:
        """
        List blob names matching a prefix.

        Args:
            prefix: Prefix to filter results (e.g., "GRADINA/U/2024/07/15/")
            max_results: Maximum number of results to return (optional)

        Returns:
            List of blob names

        Examples:
            >>> client = AzureBlobClient(config)
            >>> images = client.list_blobs(prefix="GRADINA/U/2024/07/15/")
            >>> # Returns: ["GRADINA/U/2024/07/15/10-30-45.jpg", ...]
        """
        blobs = []

        try:
            blob_iter = self.container_client.list_blobs(
                name_starts_with=prefix
            )

            for i, blob in enumerate(blob_iter):
                if max_results and i >= max_results:
                    break

                if blob.name.endswith(".jpg"):
                    blobs.append(blob.name)

            logger.debug(f"Found {len(blobs)} blobs with prefix='{prefix}'")

        except AzureError as e:
            logger.error(f"Error listing blobs: {e}")
            raise

        return blobs

    def download_blob(
        self,
        blob_name: str,
        local_path: Path | str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> bool:
        """
        Download a blob to local file with retry logic.

        Args:
            blob_name: Name of blob to download (e.g., "GRADINA/U/2024/07/15/16-20-58.jpg")
            local_path: Local path where to save the file
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)

        Returns:
            True if download successful, False otherwise

        Examples:
            >>> client = AzureBlobClient(config)
            >>> success = client.download_blob(
            ...     "GRADINA/U/2024/07/15/16-20-58.jpg",
            ...     "traffic_dataset/raw/GRADINA_U/2024-07-15_16-20-58.jpg"
            ... )
        """
        local_path = Path(local_path)

        # Create parent directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Retry loop
        for attempt in range(max_retries):
            try:
                # Get blob client
                blob_client = self.container_client.get_blob_client(blob_name)

                # Download to file
                with open(local_path, "wb") as f:
                    download_stream = blob_client.download_blob()
                    f.write(download_stream.readall())

                # Verify download
                if not local_path.exists():
                    raise Exception("Downloaded file does not exist")

                file_size = local_path.stat().st_size
                if file_size < 1000:  # Suspiciously small
                    raise Exception(f"Downloaded file too small: {file_size} bytes")

                logger.debug(f"Downloaded {blob_name} to {local_path}")
                return True

            except ResourceNotFoundError:
                logger.error(f"Blob not found: {blob_name}")
                return False

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Download attempt {attempt + 1} failed for {blob_name}: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay *= 2
                else:
                    logger.error(
                        f"Download failed after {max_retries} attempts for {blob_name}: {e}"
                    )
                    return False

        return False

    def get_blob_properties(self, blob_name: str) -> Optional[dict]:
        """
        Get properties of a blob.

        Args:
            blob_name: Name of blob

        Returns:
            Dictionary with blob properties, or None if blob not found

        Examples:
            >>> client = AzureBlobClient(config)
            >>> props = client.get_blob_properties("GRADINA/U/2024/07/15/16-20-58.jpg")
            >>> print(props["size"])
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()

            return {
                "name": blob_name,
                "size": properties.size,
                "last_modified": properties.last_modified,
                "content_type": properties.content_type
            }

        except ResourceNotFoundError:
            logger.warning(f"Blob not found: {blob_name}")
            return None

        except AzureError as e:
            logger.error(f"Error getting blob properties: {e}")
            return None

    def blob_exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists.

        Args:
            blob_name: Name of blob to check

        Returns:
            True if blob exists, False otherwise

        Examples:
            >>> client = AzureBlobClient(config)
            >>> if client.blob_exists("GRADINA/U/2024/07/15/16-20-58.jpg"):
            ...     print("Blob exists!")
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.exists()

        except AzureError as e:
            logger.error(f"Error checking blob existence: {e}")
            return False

    def close(self):
        """Close the Azure client and clean up resources."""
        try:
            self.container_client.close()
            logger.debug("Closed Azure Blob client")
        except Exception as e:
            logger.warning(f"Error closing Azure client: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_azure_client(config: Optional[AzureConfig] = None) -> AzureBlobClient:
    """
    Factory function to create Azure Blob client.

    Args:
        config: Azure configuration. If None, loads from environment variables.

    Returns:
        Configured AzureBlobClient instance

    Examples:
        >>> # Create client with config from environment
        >>> client = create_azure_client()
        >>>
        >>> # Create client with explicit config
        >>> config = AzureConfig.from_env()
        >>> client = create_azure_client(config)
        >>>
        >>> # Use as context manager
        >>> with create_azure_client() as client:
        ...     blobs = client.list_blobs("GRADINA/U/")
    """
    if config is None:
        config = AzureConfig.from_env()

    return AzureBlobClient(config)
