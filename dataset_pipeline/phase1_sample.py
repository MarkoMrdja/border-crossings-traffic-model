"""
Phase 1b: Stratified Sampling

Selects images from the discovered structure with stratified distribution
across time-of-day and season buckets.
"""

import random
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from .base import PipelinePhase
from .config import (
    PipelineConfig,
    AzureConfig,
    TIME_BUCKETS,
    SEASON_BUCKETS,
    get_time_bucket,
    get_season
)
from .azure_client import AzureBlobClient
from .utils import (
    load_json,
    save_json,
    parse_blob_name,
    construct_local_path
)

logger = logging.getLogger(__name__)


class StratifiedSamplingPhase(PipelinePhase):
    """
    Phase 1b: Select images with stratified sampling across time and season.

    Target: 700 images per camera (35 per time-season cell)
    - 5 time buckets (night, morning, midday, afternoon, evening)
    - 4 season buckets (winter, spring, summer, autumn)
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        azure_config: AzureConfig
    ):
        """
        Initialize stratified sampling phase.

        Args:
            pipeline_config: Pipeline configuration
            azure_config: Azure configuration for blob access
        """
        super().__init__(
            config=pipeline_config,
            phase_name="sample",
            description="Selecting stratified sample of images"
        )

        self.azure_config = azure_config
        self.azure_client: AzureBlobClient = None

    def parse_time_from_blob(self, blob_name: str) -> Tuple[int, str]:
        """
        Extract hour and time string from blob name.

        Args:
            blob_name: Blob name (e.g., "GRADINA/U/2024/07/15/16-20-58.jpg")

        Returns:
            Tuple of (hour, time_string) or (None, None) if invalid
        """
        parsed = parse_blob_name(blob_name)
        if not parsed:
            return None, None

        time_str = parsed["time"]
        try:
            # Parse hour from "HH-MM-SS" format
            hour = int(time_str.split("-")[0])
            return hour, time_str
        except (ValueError, IndexError):
            return None, None

    def list_images_in_day(
        self,
        camera_id: str,
        year: str,
        month: str,
        day: str
    ) -> List[Dict[str, Any]]:
        """
        List all images for a specific camera/year/month/day.

        Args:
            camera_id: Camera identifier (e.g., "GRADINA_U")
            year: Year string
            month: Month string (padded or non-padded)
            day: Day string (padded or non-padded)

        Returns:
            List of image info dictionaries
        """
        from .utils import parse_camera_id, construct_blob_path

        border, direction = parse_camera_id(camera_id)
        prefix = f"{border}/{direction}/{year}/{month}/{day}/"

        images = []

        try:
            blob_names = self.azure_client.list_blobs(prefix=prefix)

            for blob_name in blob_names:
                hour, time_str = self.parse_time_from_blob(blob_name)

                if hour is not None:
                    images.append({
                        "blob_name": blob_name,
                        "hour": hour,
                        "time_str": time_str
                    })

        except Exception as e:
            self.logger.warning(
                f"Error listing images for {camera_id}/{year}/{month}/{day}: {e}"
            )

        return images

    def select_samples_for_camera(
        self,
        camera_id: str,
        camera_inventory: Dict[str, Any],
        target_per_cell: int = 35
    ) -> List[Dict[str, Any]]:
        """
        Select stratified samples for a single camera.

        Args:
            camera_id: Camera identifier
            camera_inventory: Inventory for this camera (year → month → [days])
            target_per_cell: Target samples per time-season cell (default: 35)

        Returns:
            List of selected sample dictionaries
        """
        samples = []

        # Create stratification matrix
        for season_name, season_info in SEASON_BUCKETS.items():
            season_months = season_info["months"]

            for time_bucket_name, time_bucket_info in TIME_BUCKETS.items():
                start_hour, end_hour = time_bucket_info["hours"]

                # Collect candidate images for this cell
                candidates = []

                # Search through inventory for matching months
                for year, months in camera_inventory.items():
                    for month, days in months.items():
                        # Check if month matches season
                        month_padded = month.zfill(2)
                        if month_padded not in season_months:
                            continue

                        # Sample days to search (not all days)
                        sample_days = random.sample(
                            days,
                            min(len(days), 10)  # Search max 10 days per month
                        )

                        for day in sample_days:
                            # List images for this day
                            images = self.list_images_in_day(
                                camera_id, year, month, day
                            )

                            # Filter by time bucket
                            for img in images:
                                hour = img["hour"]

                                # Check if hour is in time bucket
                                in_bucket = False
                                if start_hour > end_hour:  # Wrap-around (night)
                                    in_bucket = hour >= start_hour or hour < end_hour
                                else:
                                    in_bucket = start_hour <= hour < end_hour

                                if in_bucket:
                                    candidates.append({
                                        "camera_id": camera_id,
                                        "blob_name": img["blob_name"],
                                        "year": year,
                                        "month": month_padded,
                                        "day": day.zfill(2),
                                        "time": img["time_str"],
                                        "hour": hour,
                                        "time_bucket": time_bucket_name,
                                        "season": season_name
                                    })

                # Select from candidates
                if len(candidates) >= target_per_cell:
                    selected = random.sample(candidates, target_per_cell)
                else:
                    selected = candidates
                    if len(selected) > 0:
                        self.logger.warning(
                            f"{camera_id} {season_name}/{time_bucket_name}: "
                            f"Only {len(selected)}/{target_per_cell} samples available"
                        )

                samples.extend(selected)

        return samples

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute stratified sampling phase.

        Args:
            resume: If True, skip sampling if manifest already exists

        Returns:
            Dictionary with manifest and statistics
        """
        inventory_file = self.config.get_path(self.config.inventory_file)
        manifest_file = self.config.get_path(self.config.sample_manifest_file)

        # Check if we can resume
        if resume and manifest_file.exists():
            self.logger.info(f"Manifest already exists at {manifest_file}")
            manifest = load_json(manifest_file)

            if manifest and manifest.get("samples"):
                self.logger.info("Resuming with existing manifest")
                return {
                    "manifest": manifest,
                    "resumed": True
                }

        # Load inventory
        if not inventory_file.exists():
            raise FileNotFoundError(
                f"Inventory file not found: {inventory_file}. "
                "Run Phase 1a (discover) first."
            )

        inventory = load_json(inventory_file)
        if not inventory:
            raise ValueError("Inventory is empty")

        # Initialize Azure client
        self.azure_client = AzureBlobClient(self.azure_config)

        try:
            all_samples = []

            # Process each camera
            cameras = list(inventory.keys())
            self.logger.info(f"Sampling from {len(cameras)} cameras...")

            for camera_id in tqdm(cameras, desc="Sampling cameras"):
                camera_inventory = inventory[camera_id]

                # Select samples for this camera
                camera_samples = self.select_samples_for_camera(
                    camera_id,
                    camera_inventory,
                    target_per_cell=self.config.samples_per_season
                )

                self.logger.info(
                    f"{camera_id}: Selected {len(camera_samples)} samples "
                    f"(target: {self.config.initial_samples_per_camera})"
                )

                # Add local path to each sample
                for sample in camera_samples:
                    sample["local_path"] = construct_local_path(
                        sample["camera_id"],
                        sample["year"],
                        sample["month"],
                        sample["day"],
                        sample["time"]
                    )

                all_samples.extend(camera_samples)

            # Create manifest
            manifest = {
                "total_samples": len(all_samples),
                "cameras": len(cameras),
                "per_camera": self.config.initial_samples_per_camera,
                "target_per_cell": self.config.samples_per_season,
                "time_buckets": list(TIME_BUCKETS.keys()),
                "season_buckets": list(SEASON_BUCKETS.keys()),
                "created_at": datetime.now().isoformat(),
                "samples": all_samples
            }

            # Save manifest
            self.logger.info(f"Saving manifest to {manifest_file}")
            save_json(manifest, manifest_file)

            # Calculate distribution statistics
            distribution = self._calculate_distribution(all_samples)

            result = {
                "manifest": manifest,
                "distribution": distribution,
                "resumed": False
            }

            self.logger.info(
                f"Manifest saved: {len(all_samples)} total samples "
                f"from {len(cameras)} cameras"
            )

            return result

        finally:
            # Clean up Azure client
            if self.azure_client:
                self.azure_client.close()

    def _calculate_distribution(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate distribution statistics for samples."""
        distribution = {
            "by_time_bucket": {},
            "by_season": {},
            "by_camera": {}
        }

        for sample in samples:
            # Count by time bucket
            time_bucket = sample["time_bucket"]
            distribution["by_time_bucket"][time_bucket] = \
                distribution["by_time_bucket"].get(time_bucket, 0) + 1

            # Count by season
            season = sample["season"]
            distribution["by_season"][season] = \
                distribution["by_season"].get(season, 0) + 1

            # Count by camera
            camera = sample["camera_id"]
            distribution["by_camera"][camera] = \
                distribution["by_camera"].get(camera, 0) + 1

        return distribution

    def validate(self) -> bool:
        """
        Validate that sampling completed successfully.

        Returns:
            True if manifest file exists and contains reasonable data
        """
        manifest_file = self.config.get_path(self.config.sample_manifest_file)

        if not manifest_file.exists():
            self.logger.error(f"Manifest file not found: {manifest_file}")
            return False

        manifest = load_json(manifest_file)

        if not manifest or not manifest.get("samples"):
            self.logger.error("Manifest file is empty or has no samples")
            return False

        # Check total samples
        total_samples = len(manifest["samples"])
        expected_min = 10000  # At least this many samples

        if total_samples < expected_min:
            self.logger.warning(
                f"Only {total_samples} samples, expected at least {expected_min}"
            )
            return False

        self.logger.info(f"Validation passed: {total_samples} samples in manifest")
        return True


def main():
    """
    CLI entry point for Phase 1b: Stratified Sampling.

    Usage:
        python -m dataset_pipeline.phase1_sample [--resume]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1b: Stratified sampling of images"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest if available"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation"
    )

    args = parser.parse_args()

    # Load configurations
    pipeline_config = PipelineConfig()
    azure_config = AzureConfig.from_env()

    # Ensure directories exist
    pipeline_config.ensure_directories()

    # Create phase
    phase = StratifiedSamplingPhase(pipeline_config, azure_config)

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
        print("Phase 1b: Stratified Sampling - Complete")
        print("=" * 60)

        if result["status"] == "completed":
            data = result["data"]
            manifest = data.get("manifest", {})
            distribution = data.get("distribution", {})

            print(f"Status: ✓ Success")
            print(f"Duration: {result['duration_seconds']:.1f} seconds")
            print(f"\nStatistics:")
            print(f"  Total samples: {manifest.get('total_samples', 0)}")
            print(f"  Cameras: {manifest.get('cameras', 0)}")

            if distribution:
                print(f"\nDistribution by time bucket:")
                for bucket, count in distribution.get("by_time_bucket", {}).items():
                    print(f"    {bucket}: {count}")

                print(f"\nDistribution by season:")
                for season, count in distribution.get("by_season", {}).items():
                    print(f"    {season}: {count}")

            if data.get("resumed"):
                print(f"\n(Resumed from existing manifest)")

            print("=" * 60)
            exit(0)
        else:
            print(f"Status: ✗ Failed")
            print(f"Reason: {result.get('reason', 'Unknown')}")
            print("=" * 60)
            exit(1)


if __name__ == "__main__":
    main()
