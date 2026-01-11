"""
Phase 1c: Parallel Download

Downloads all selected images from Azure Blob Storage with resume capability,
progress tracking, and error handling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base import PipelinePhase
from .config import PipelineConfig, AzureConfig
from .azure_client import AzureBlobClient
from .utils import (
    load_json,
    save_json,
    create_progress_tracker,
    validate_image_path,
    is_image_corrupt
)

logger = logging.getLogger(__name__)


class ParallelDownloadPhase(PipelinePhase):
    """
    Phase 1c: Download selected images from Azure Blob Storage.

    Downloads all images in the sample manifest with:
    - Resume capability (skip already downloaded files)
    - Progress tracking
    - Parallel downloads for efficiency
    - Error handling and retry logic
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        azure_config: AzureConfig,
        max_workers: int = 4
    ):
        """
        Initialize parallel download phase.

        Args:
            pipeline_config: Pipeline configuration
            azure_config: Azure configuration for blob access
            max_workers: Maximum number of parallel download threads (default: 4)
        """
        super().__init__(
            config=pipeline_config,
            phase_name="download",
            description="Downloading images from Azure Blob Storage"
        )

        self.azure_config = azure_config
        self.max_workers = max_workers

    def download_single_image(
        self,
        azure_client: AzureBlobClient,
        sample: Dict[str, Any],
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Download a single image.

        Args:
            azure_client: Azure Blob Storage client
            sample: Sample dictionary with blob_name and local_path
            skip_existing: If True, skip if file already exists and is valid

        Returns:
            Result dictionary with status and details
        """
        blob_name = sample["blob_name"]
        local_path = self.config.base_dir / sample["local_path"]

        # Check if file already exists and is valid
        if skip_existing and validate_image_path(local_path, min_size_bytes=1000):
            return {
                "blob_name": blob_name,
                "local_path": str(local_path),
                "status": "skipped",
                "reason": "already_exists"
            }

        # Download
        try:
            success = azure_client.download_blob(
                blob_name=blob_name,
                local_path=local_path,
                max_retries=3,
                retry_delay=1.0
            )

            if success:
                # Check for corruption
                if is_image_corrupt(local_path):
                    self.logger.warning(f"Corrupt image detected, deleting: {local_path}")
                    local_path.unlink()  # Delete corrupt download
                    return {
                        "blob_name": blob_name,
                        "local_path": str(local_path),
                        "status": "failed",
                        "reason": "corrupt_image"
                    }

                return {
                    "blob_name": blob_name,
                    "local_path": str(local_path),
                    "status": "success"
                }
            else:
                return {
                    "blob_name": blob_name,
                    "local_path": str(local_path),
                    "status": "failed",
                    "reason": "download_failed"
                }

        except Exception as e:
            self.logger.error(f"Error downloading {blob_name}: {e}")
            return {
                "blob_name": blob_name,
                "local_path": str(local_path),
                "status": "failed",
                "reason": str(e)
            }

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute parallel download phase.

        Args:
            resume: If True, skip already downloaded files

        Returns:
            Dictionary with download results and statistics
        """
        manifest_file = self.config.get_path(self.config.sample_manifest_file)

        # Load manifest
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_file}. "
                "Run Phase 1b (sample) first."
            )

        manifest = load_json(manifest_file)
        if not manifest or not manifest.get("samples"):
            raise ValueError("Manifest is empty or has no samples")

        samples = manifest["samples"]
        total_samples = len(samples)

        self.logger.info(f"Starting download of {total_samples} images...")
        if resume:
            self.logger.info("Resume mode: skipping existing valid files")

        # Initialize progress tracking
        progress = self.initialize_progress(total=total_samples)

        # Results tracking
        results = {
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "failed_items": []
        }

        # Create Azure client pool (one per worker thread)
        def create_client_for_worker():
            return AzureBlobClient(self.azure_config)

        # Parallel download with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create one client per worker
            clients = {i: create_client_for_worker() for i in range(self.max_workers)}
            client_pool = list(clients.values())
            current_client_idx = 0

            # Submit all download tasks
            futures = {}
            for sample in samples:
                # Round-robin client assignment
                client = client_pool[current_client_idx]
                current_client_idx = (current_client_idx + 1) % len(client_pool)

                future = executor.submit(
                    self.download_single_image,
                    client,
                    sample,
                    skip_existing=resume
                )
                futures[future] = sample

            # Process completed downloads with progress bar
            with tqdm(total=total_samples, desc="Downloading") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    # Update results
                    status = result["status"]
                    if status == "success":
                        results["success"] += 1
                    elif status == "skipped":
                        results["skipped"] += 1
                    else:  # failed
                        results["failed"] += 1
                        results["failed_items"].append({
                            "blob_name": result["blob_name"],
                            "reason": result.get("reason", "unknown")
                        })

                    # Update progress
                    progress["completed"] = (
                        results["success"] +
                        results["skipped"] +
                        results["failed"]
                    )

                    # Save progress every 100 items
                    if progress["completed"] % 100 == 0:
                        self.save_progress()

                    pbar.update(1)
                    pbar.set_postfix({
                        "success": results["success"],
                        "skipped": results["skipped"],
                        "failed": results["failed"]
                    })

            # Close all clients
            for client in client_pool:
                client.close()

        # Final progress save
        progress.update(results)
        self.save_progress()

        self.logger.info(
            f"Download complete: {results['success']} succeeded, "
            f"{results['skipped']} skipped, {results['failed']} failed"
        )

        return {
            "total": total_samples,
            "results": results,
            "progress": progress
        }

    def validate(self) -> bool:
        """
        Validate that download completed successfully.

        Returns:
            True if most images were downloaded successfully
        """
        # Load progress
        self.load_progress()

        if not self.progress:
            self.logger.error("No progress data found")
            return False

        # Check success rate
        total = self.progress.get("total", 0)
        success = self.progress.get("success", 0)
        skipped = self.progress.get("skipped", 0)
        failed = self.progress.get("failed", 0)

        if total == 0:
            self.logger.error("No downloads attempted")
            return False

        # Success rate (including skipped as they were already downloaded)
        success_rate = (success + skipped) / total

        if success_rate < 0.95:  # At least 95% should be downloaded
            self.logger.warning(
                f"Success rate too low: {success_rate:.1%} "
                f"({success + skipped}/{total})"
            )
            return False

        self.logger.info(
            f"Validation passed: {success} downloaded, {skipped} skipped, "
            f"{failed} failed (success rate: {success_rate:.1%})"
        )

        return True


def main():
    """
    CLI entry point for Phase 1c: Parallel Download.

    Usage:
        python -m dataset_pipeline.phase1_download [--resume] [--workers N]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1c: Download images from Azure Blob Storage"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume download, skipping existing files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest file (default: sample_manifest.json)"
    )

    args = parser.parse_args()

    # Load configurations
    pipeline_config = PipelineConfig()

    # Override manifest file if specified
    if args.manifest:
        pipeline_config.sample_manifest_file = args.manifest

    azure_config = AzureConfig.from_env()

    # Ensure directories exist
    pipeline_config.ensure_directories()

    # Create phase
    phase = ParallelDownloadPhase(
        pipeline_config,
        azure_config,
        max_workers=args.workers
    )

    if args.validate_only:
        # Only run validation
        print("Running validation...")
        if phase.validate():
            print("✓ Validation passed")
            exit(0)
        else:
            print("✗ Validation failed")
            exit(1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        # Print results
        print("\n" + "=" * 60)
        print("Phase 1c: Parallel Download - Complete")
        print("=" * 60)

        if result["status"] == "completed":
            data = result["data"]
            total = data.get("total", 0)
            results = data.get("results", {})

            print(f"Status: ✓ Success")
            print(f"Duration: {result['duration_seconds']:.1f} seconds")
            print(f"\nResults:")
            print(f"  Total: {total}")
            print(f"  Downloaded: {results.get('success', 0)}")
            print(f"  Skipped (existing): {results.get('skipped', 0)}")
            print(f"  Failed: {results.get('failed', 0)}")

            if results.get('failed', 0) > 0:
                print(f"\nFailed downloads:")
                for item in results.get('failed_items', [])[:10]:
                    print(f"    {item['blob_name']}: {item['reason']}")
                if results['failed'] > 10:
                    print(f"    ... and {results['failed'] - 10} more")

            print("=" * 60)
            exit(0)
        else:
            print(f"Status: ✗ Failed")
            print(f"Reason: {result.get('reason', 'Unknown')}")
            print("=" * 60)
            exit(1)


if __name__ == "__main__":
    main()
