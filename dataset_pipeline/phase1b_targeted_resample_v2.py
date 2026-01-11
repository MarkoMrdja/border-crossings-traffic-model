"""
Phase 1b: Targeted Resampling (v2 - Simplified)

Lists images around existing traffic_present images within ±1 hour window.
No YOLO verification - user will review manually in Phase 4.

Strategy:
1. Load existing traffic_present images from binary_selection.json
2. For each image, list ALL images from that day in Azure
3. Filter to ±1 hour time window
4. Exclude already-downloaded images
5. Add to manifest for download
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict

from .base import PipelinePhase
from .config import PipelineConfig, AzureConfig
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class TargetedResamplingPhaseV2(PipelinePhase):
    """
    Phase 1b: Targeted resampling by listing neighbors of existing traffic images.

    For each traffic_present image, lists all images from that day within ±1 hour window.
    """

    def __init__(self, pipeline_config: PipelineConfig, azure_config: AzureConfig):
        """Initialize targeted resampling phase."""
        super().__init__(
            config=pipeline_config,
            phase_name="targeted_resample",
            description="Targeted resampling via neighbor listing"
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

    def run(self) -> Dict[str, Any]:
        """
        Execute targeted resampling.

        Returns:
            Manifest dictionary with resampled images
        """
        if not self.recommendations:
            raise ValueError("No resampling recommendations. Run analyze_dataset_for_resampling.py first.")

        self.logger.info("\nLoading resampling recommendations...")
        self.logger.info(f"  Target total: {self.recommendations['target_total']}")
        self.logger.info(f"  Gap to fill: +{self.recommendations['gap_present']} present, "
                        f"+{self.recommendations['gap_absent']} absent")

        # Initialize Azure client
        self.logger.info("\nInitializing Azure Blob Storage client...")
        from .azure_client import AzureBlobClient
        self.azure_client = AzureBlobClient(self.azure_config)
        self.logger.info("✓ Azure client initialized")

        # Load existing images to exclude
        existing_images = self._load_all_existing_images()
        self.logger.info(f"\nFound {len(existing_images)} existing images to exclude")

        # Load traffic_present seeds
        seeds_by_camera = self._load_traffic_present_seeds()
        total_seeds = sum(len(seeds) for seeds in seeds_by_camera.values())
        self.logger.info(f"Found {total_seeds} traffic_present seeds across {len(seeds_by_camera)} cameras")

        # Sample neighbors for cameras that need more
        recommendations = self.recommendations['recommendations']
        all_samples = []

        for camera_id, rec in recommendations.items():
            if rec['needed_present'] <= 0:
                continue  # Camera doesn't need more traffic_present samples

            self.logger.info(f"\n{camera_id}: Need +{rec['needed_present']} present")

            seeds = seeds_by_camera.get(camera_id, [])
            if not seeds:
                self.logger.warning(f"  No traffic_present seeds for {camera_id}, skipping")
                continue

            self.logger.info(f"  Using {len(seeds)} traffic_present seeds")

            # List neighbors around each seed
            neighbors = self._list_neighbors_around_seeds(
                camera_id=camera_id,
                seeds=seeds,
                existing_images=existing_images,
                target_count=rec['needed_present']
            )

            self.logger.info(f"  ✓ Found {len(neighbors)} new neighbor images")
            all_samples.extend(neighbors)

        # Build manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "strategy": "neighbor_listing",
            "time_window_hours": 1,
            "samples": all_samples,
            "statistics": self._calculate_statistics(all_samples)
        }

        # Save manifest
        output_path = self.config.base_dir / "targeted_resample_manifest.json"
        save_json(manifest, output_path)
        self.logger.info(f"\n✓ Manifest saved to {output_path}")

        # Log summary
        stats = manifest['statistics']
        self.logger.info(f"\nResampling completed:")
        self.logger.info(f"  Total sampled: {stats['total_samples']}")
        self.logger.info(f"  Cameras covered: {len(stats['by_camera'])}")

        return manifest

    def _load_all_existing_images(self) -> Set[str]:
        """
        Load all existing image blob names to exclude them.

        Returns:
            Set of blob names we already have
        """
        existing = set()

        # Load from binary_selection.json
        selection_path = self.config.base_dir / "binary_selection.json"
        if selection_path.exists():
            selection = load_json(selection_path)
            for item in selection.get('selection', []):
                local_path = item.get('local_path')
                if local_path:
                    # Convert local path to blob name
                    # raw/CAMERA/YYYY-MM-DD_HH-MM-SS.jpg → not-processed-imgs/CAMERA/YYYY/MM/DD/CAMERA_YYYY-MM-DD_HH-MM-SS.jpg
                    filename = Path(local_path).name
                    camera_id = item['camera_id']

                    try:
                        # Parse: YYYY-MM-DD_HH-MM-SS.jpg
                        parts = filename.replace('.jpg', '').split('_')
                        date_str = parts[0]  # YYYY-MM-DD
                        time_str = parts[1]  # HH-MM-SS

                        year, month, day = date_str.split('-')

                        # Split camera_id: KELEBIJA_U -> KELEBIJA/U
                        if '_' in camera_id:
                            camera, direction = camera_id.rsplit('_', 1)
                            blob_name = f"{camera}/{direction}/{year}/{month}/{day}/{time_str}.jpg"
                        else:
                            # Fallback if camera_id doesn't have underscore
                            blob_name = f"{camera_id}/{year}/{month}/{day}/{time_str}.jpg"

                        existing.add(blob_name)
                    except (IndexError, ValueError):
                        continue

        return existing

    def _load_traffic_present_seeds(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load traffic_present images as seeds for neighbor listing.

        Returns:
            Dict mapping camera_id to list of seed info dicts
        """
        seeds_by_camera = defaultdict(list)

        selection_path = self.config.base_dir / "binary_selection.json"
        if not selection_path.exists():
            return seeds_by_camera

        selection = load_json(selection_path)

        for item in selection.get('selection', []):
            if item['binary_label'] != 'traffic_present':
                continue

            camera_id = item['camera_id']
            local_path = item['local_path']
            filename = Path(local_path).name

            try:
                # Parse: YYYY-MM-DD_HH-MM-SS.jpg
                parts = filename.replace('.jpg', '').split('_')
                date_str = parts[0]  # YYYY-MM-DD
                time_str = parts[1]  # HH-MM-SS

                timestamp = datetime.strptime(
                    f"{date_str} {time_str}",
                    "%Y-%m-%d %H-%M-%S"
                )

                year, month, day = date_str.split('-')

                seeds_by_camera[camera_id].append({
                    'timestamp': timestamp,
                    'year': year,
                    'month': month,
                    'day': day,
                    'filename': filename
                })

            except (IndexError, ValueError) as e:
                self.logger.debug(f"Could not parse timestamp from {filename}: {e}")
                continue

        return seeds_by_camera

    def _list_neighbors_around_seeds(
        self,
        camera_id: str,
        seeds: List[Dict[str, Any]],
        existing_images: Set[str],
        target_count: int
    ) -> List[Dict[str, Any]]:
        """
        List neighbor images around seed images within ±1 hour window.

        Args:
            camera_id: Camera identifier
            seeds: List of seed image info dicts
            existing_images: Set of blob names we already have
            target_count: Target number of images to collect

        Returns:
            List of neighbor image sample dicts
        """
        neighbors = []
        seen_blobs = set()

        # Process each seed
        for seed in seeds:
            if len(neighbors) >= target_count:
                break  # Have enough

            timestamp = seed['timestamp']
            year = seed['year']
            month = seed['month']
            day = seed['day']

            # Time window: ±1 hour
            window_start = timestamp - timedelta(hours=1)
            window_end = timestamp + timedelta(hours=1)

            # List all blobs for this day
            # Azure uses unpadded month/day: 2024/8/9 not 2024/08/09
            month_unpadded = str(int(month))
            day_unpadded = str(int(day))

            # Split camera_id: BATROVCI_I -> BATROVCI/I
            if '_' in camera_id:
                camera, direction = camera_id.rsplit('_', 1)
                prefix = f"{camera}/{direction}/{year}/{month_unpadded}/{day_unpadded}/"
            else:
                prefix = f"{camera_id}/{year}/{month_unpadded}/{day_unpadded}/"

            try:
                blobs = self.azure_client.list_blobs(prefix=prefix)

                for blob_name in blobs:
                    if len(neighbors) >= target_count:
                        break

                    # Skip if already have this image
                    if blob_name in existing_images or blob_name in seen_blobs:
                        continue

                    # Parse timestamp from blob name
                    # Format: BATROVCI/I/2024/8/9/09-47-42.jpg
                    # Filename is just the time: 09-47-42.jpg
                    try:
                        filename = blob_name.split('/')[-1]
                        blob_time_str = filename.replace('.jpg', '')  # HH-MM-SS

                        # Reconstruct full date from seed's year/month/day
                        blob_date_str = f"{year}-{month}-{day}"

                        blob_timestamp = datetime.strptime(
                            f"{blob_date_str} {blob_time_str}",
                            "%Y-%m-%d %H-%M-%S"
                        )

                        # Check if within ±1 hour window
                        if window_start <= blob_timestamp <= window_end:
                            # Generate local path: raw/CAMERA_ID/YYYY-MM-DD_HH-MM-SS.jpg
                            local_path = f"raw/{camera_id}/{blob_date_str}_{blob_time_str}.jpg"

                            neighbors.append({
                                "camera_id": camera_id,
                                "blob_name": blob_name,
                                "local_path": local_path,
                                "year": year,
                                "month": month,
                                "day": day,
                                "time": blob_time_str,
                                "source": "neighbor_listing",
                                "seed_image": seed['filename']
                            })
                            seen_blobs.add(blob_name)

                    except (IndexError, ValueError) as e:
                        self.logger.debug(f"Could not parse timestamp from {blob_name}: {e}")
                        continue

            except Exception as e:
                self.logger.warning(f"Error listing blobs for {camera_id}/{year}/{month}/{day}: {e}")
                continue

        return neighbors

    def _calculate_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for sampled data."""
        stats = {
            'total_samples': len(samples),
            'by_camera': defaultdict(int),
            'by_source': defaultdict(int)
        }

        for sample in samples:
            camera_id = sample['camera_id']
            source = sample.get('source', 'unknown')

            stats['by_camera'][camera_id] += 1
            stats['by_source'][source] += 1

        stats['by_camera'] = dict(stats['by_camera'])
        stats['by_source'] = dict(stats['by_source'])

        return stats

    def validate(self) -> bool:
        """Validate that resampling completed successfully."""
        manifest_path = self.config.base_dir / 'targeted_resample_manifest.json'

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
        description="Phase 1b: Targeted resampling via neighbor listing"
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
    phase = TargetedResamplingPhaseV2(pipeline_config, azure_config)

    if args.validate_only:
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Run resampling
        manifest = phase.run()

        print(f"\n✓ Targeted resampling completed")
        stats = manifest['statistics']
        print(f"  Total sampled: {stats['total_samples']}")
        print(f"  Cameras covered: {len(stats['by_camera'])}")

        print(f"\nNext step: Download samples with phase1_download")
        print(f"  python3 -m dataset_pipeline.phase1_download --manifest targeted_resample_manifest.json")

        exit(0)


if __name__ == "__main__":
    main()
