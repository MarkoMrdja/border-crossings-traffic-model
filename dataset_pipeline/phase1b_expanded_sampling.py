"""
Phase 1b: Expanded Stratified Sampling

Expands on the original stratified sampling by sampling more days per month.
Original sampled 10 days/month, this samples 20-25 days/month to get more coverage.

Maintains temporal diversity while increasing dataset size.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
from collections import defaultdict

from .base import PipelinePhase
from .config import PipelineConfig, AzureConfig
from .utils import load_json, save_json
from .azure_client import AzureBlobClient

logger = logging.getLogger(__name__)


class ExpandedSamplingPhase(PipelinePhase):
    """
    Phase 1b: Expanded stratified sampling with more days per month.

    Samples 20-25 days per month (vs original 10) to increase coverage
    while maintaining temporal diversity.
    """

    def __init__(self, pipeline_config: PipelineConfig, azure_config: AzureConfig):
        """Initialize expanded sampling phase."""
        super().__init__(
            config=pipeline_config,
            phase_name="expanded_sampling",
            description="Expanded stratified sampling (20-25 days/month)"
        )

        self.azure_config = azure_config
        self.azure_client = None
        self.recommendations = None

        # Load resampling recommendations
        rec_path = self.config.base_dir / "resampling_recommendations.json"
        if rec_path.exists():
            self.recommendations = load_json(rec_path)
            self.logger.info("Loaded resampling recommendations")
        else:
            self.logger.warning("No resampling recommendations found")

    def run(self, days_per_month: int = 20) -> Dict[str, Any]:
        """
        Execute expanded sampling.

        Args:
            days_per_month: Number of days to sample per month (default: 20)

        Returns:
            Manifest dictionary with sampled images
        """
        if not self.recommendations:
            raise ValueError("No resampling recommendations. Run analyze_dataset_for_resampling.py first.")

        self.logger.info(f"\nExpanded stratified sampling ({days_per_month} days/month)")
        self.logger.info(f"  Target total: {self.recommendations['target_total']}")
        self.logger.info(f"  Gap to fill: +{self.recommendations['gap_present']} present, "
                        f"+{self.recommendations['gap_absent']} absent")

        # Initialize Azure client
        self.logger.info("\nInitializing Azure Blob Storage client...")
        self.azure_client = AzureBlobClient(self.azure_config)
        self.logger.info("✓ Azure client initialized")

        # Load inventory
        inventory_path = self.config.base_dir / "inventory.json"
        inventory = load_json(inventory_path)
        self.logger.info(f"\nLoaded inventory with {len(inventory)} cameras")

        # Load existing images to exclude
        existing_images = self._load_existing_images()
        self.logger.info(f"Found {len(existing_images)} existing images to exclude")

        # Sample for each camera
        recommendations = self.recommendations['recommendations']
        all_samples = []

        # Group by priority
        by_priority = defaultdict(list)
        for camera_id, rec in recommendations.items():
            if rec['needed_present'] > 0 or rec['needed_absent'] > 0:
                priority = rec.get('priority', 'low')
                by_priority[priority].append((camera_id, rec))

        # Process by priority
        for priority in ['critical', 'high', 'medium', 'low']:
            cameras = by_priority.get(priority, [])
            if not cameras:
                continue

            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Processing {priority.upper()} priority cameras ({len(cameras)})")
            self.logger.info(f"{'='*70}")

            for camera_id, rec in cameras:
                needed = rec['needed_present'] + rec['needed_absent']
                self.logger.info(f"\n{camera_id}: Need {needed} total "
                               f"(+{rec['needed_present']} present, +{rec['needed_absent']} absent)")

                camera_inventory = inventory.get(camera_id, {})
                if not camera_inventory:
                    self.logger.warning(f"  No inventory for {camera_id}")
                    continue

                # Sample from this camera
                samples = self._stratified_sample(
                    camera_id=camera_id,
                    camera_inventory=camera_inventory,
                    target_samples=needed,
                    days_per_month=days_per_month,
                    existing_images=existing_images
                )

                self.logger.info(f"  ✓ Sampled {len(samples)} new images")
                all_samples.extend(samples)

        # Build manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "strategy": "expanded_stratified",
            "days_per_month": days_per_month,
            "samples": all_samples,
            "statistics": self._calculate_statistics(all_samples)
        }

        # Save manifest
        output_path = self.config.base_dir / "expanded_sampling_manifest.json"
        save_json(manifest, output_path)
        self.logger.info(f"\n✓ Manifest saved to {output_path}")

        # Log summary
        stats = manifest['statistics']
        self.logger.info(f"\nSampling completed:")
        self.logger.info(f"  Total sampled: {stats['total_samples']}")
        self.logger.info(f"  Cameras covered: {len(stats['by_camera'])}")
        self.logger.info(f"  By month: {stats['by_month']}")

        return manifest

    def parse_time_from_blob(self, blob_name: str) -> tuple[int, str]:
        """
        Parse hour and time string from blob name.

        Args:
            blob_name: Full blob name (e.g., "GRADINA/U/2024/07/15/16-20-58.jpg")

        Returns:
            Tuple of (hour, time_str) or (None, None) if parsing fails
        """
        try:
            # Extract filename: "16-20-58.jpg"
            filename = blob_name.split("/")[-1]
            # Remove extension: "16-20-58"
            time_str = filename.replace(".jpg", "")
            # Parse hour
            hour = int(time_str.split("-")[0])
            return hour, time_str
        except (IndexError, ValueError):
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
            List of image info dictionaries with blob_name, hour, time_str
        """
        from .utils import parse_camera_id

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

    def _load_existing_images(self) -> Set[str]:
        """
        Load existing images to exclude them.

        Returns:
            Set of "camera_year_month_day" keys
        """
        existing = set()

        # Load from binary_selection.json
        selection_path = self.config.base_dir / "binary_selection.json"
        if selection_path.exists():
            selection = load_json(selection_path)
            for item in selection.get('selection', []):
                local_path = item.get('local_path')
                if local_path:
                    # raw/CAMERA/YYYY-MM-DD_HH-MM-SS.jpg
                    filename = Path(local_path).name
                    camera_id = item['camera_id']

                    try:
                        # Parse: YYYY-MM-DD_HH-MM-SS.jpg
                        parts = filename.replace('.jpg', '').split('_')
                        date_str = parts[0]  # YYYY-MM-DD
                        year, month, day = date_str.split('-')

                        # Store as camera_year_month_day to track which days we've sampled
                        key = f"{camera_id}_{year}_{month}_{day}"
                        existing.add(key)
                    except (IndexError, ValueError):
                        continue

        return existing

    def _stratified_sample(
        self,
        camera_id: str,
        camera_inventory: Dict[str, Any],
        target_samples: int,
        days_per_month: int,
        existing_images: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform stratified sampling across time periods.

        Lists actual blobs from Azure and samples more days per month than original.

        Args:
            camera_id: Camera identifier
            camera_inventory: Inventory for this camera
            target_samples: Target number of samples
            days_per_month: Number of days to sample per month
            existing_images: Set of existing day keys to exclude

        Returns:
            List of sample dictionaries
        """
        # Define time buckets (same as original stratified sampling)
        time_buckets = [
            (0, 6),    # Night: 00:00-06:00
            (6, 10),   # Morning: 06:00-10:00
            (10, 14),  # Midday: 10:00-14:00
            (14, 18),  # Afternoon: 14:00-18:00
            (18, 24),  # Evening: 18:00-24:00
        ]

        # Define seasons
        seasons = {
            'winter': ['12', '01', '02'],
            'spring': ['03', '04', '05'],
            'summer': ['06', '07', '08'],
            'autumn': ['09', '10', '11']
        }

        # Collect all candidate images by listing actual blobs
        all_candidates = []

        for year, months in camera_inventory.items():
            for month, days in months.items():
                # Determine season
                month_padded = month.zfill(2)
                season_name = None
                for s_name, s_months in seasons.items():
                    if month_padded in s_months:
                        season_name = s_name
                        break

                if not season_name:
                    continue

                # Sample more days than original (20-25 vs 10)
                available_days = [d for d in days
                                 if f"{camera_id}_{year}_{month}_{d}" not in existing_images]

                if not available_days:
                    continue

                sample_size = min(len(available_days), days_per_month)
                sampled_days = random.sample(available_days, sample_size)

                # For each sampled day, list actual images from Azure
                for day in sampled_days:
                    images = self.list_images_in_day(camera_id, year, month, day)

                    # Categorize by time bucket
                    for img in images:
                        hour = img["hour"]

                        # Determine time bucket
                        for time_start, time_end in time_buckets:
                            in_bucket = False
                            if time_start > time_end:  # Wrap-around (night)
                                in_bucket = hour >= time_start or hour < time_end
                            else:
                                in_bucket = time_start <= hour < time_end

                            if in_bucket:
                                # Generate local path
                                day_padded = day.zfill(2)
                                local_path = f"raw/{camera_id}/{year}-{month_padded}-{day_padded}_{img['time_str']}.jpg"

                                all_candidates.append({
                                    "camera_id": camera_id,
                                    "blob_name": img["blob_name"],
                                    "local_path": local_path,
                                    "year": year,
                                    "month": month_padded,
                                    "day": day_padded,
                                    "time": img["time_str"],
                                    "hour": hour,
                                    "season": season_name,
                                    "source": "expanded_stratified"
                                })
                                break  # Only add to one time bucket

        # Randomly select target number of samples
        if len(all_candidates) >= target_samples:
            samples = random.sample(all_candidates, target_samples)
        else:
            samples = all_candidates
            self.logger.warning(
                f"{camera_id}: Only found {len(samples)}/{target_samples} candidate images"
            )

        return samples

    def _calculate_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for sampled data."""
        stats = {
            'total_samples': len(samples),
            'by_camera': defaultdict(int),
            'by_month': defaultdict(int),
            'by_hour': defaultdict(int)
        }

        for sample in samples:
            camera_id = sample['camera_id']
            month = sample.get('month', 'unknown')
            hour = sample.get('hour', -1)

            stats['by_camera'][camera_id] += 1
            stats['by_month'][month] += 1
            if hour >= 0:
                stats['by_hour'][hour] += 1

        stats['by_camera'] = dict(stats['by_camera'])
        stats['by_month'] = dict(stats['by_month'])
        stats['by_hour'] = dict(stats['by_hour'])

        return stats

    def validate(self) -> bool:
        """Validate that sampling completed successfully."""
        manifest_path = self.config.base_dir / 'expanded_sampling_manifest.json'

        if not manifest_path.exists():
            self.logger.error("Manifest file not found")
            return False

        try:
            manifest = load_json(manifest_path)

            if 'samples' not in manifest:
                self.logger.error("Missing 'samples' in manifest")
                return False

            samples_count = len(manifest.get('samples', []))

            if samples_count == 0:
                self.logger.error("No samples in manifest")
                return False

            self.logger.info(f"Validation passed: {samples_count} samples in manifest")
            return True

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False


def main():
    """CLI entry point for Phase 1b."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1b: Expanded stratified sampling"
    )
    parser.add_argument(
        "--days-per-month",
        type=int,
        default=20,
        help="Number of days to sample per month (default: 20)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation"
    )

    args = parser.parse_args()

    # Initialize configurations
    pipeline_config = PipelineConfig()
    azure_config = AzureConfig.from_env()

    # Create phase instance
    phase = ExpandedSamplingPhase(pipeline_config, azure_config)

    if args.validate_only:
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Run sampling
        manifest = phase.run(days_per_month=args.days_per_month)

        print(f"\n✓ Expanded sampling completed")
        stats = manifest['statistics']
        print(f"  Total sampled: {stats['total_samples']}")
        print(f"  Cameras covered: {len(stats['by_camera'])}")

        print(f"\nNext step: Download samples with phase1_download")
        print(f"  python3 -m dataset_pipeline.phase1_download --manifest expanded_sampling_manifest.json")

        exit(0)


if __name__ == "__main__":
    main()
