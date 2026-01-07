"""
Phase 7: Train/Val Split

Creates stratified 80/20 train/validation split for binary classification.
Ensures each camera is represented in both train and val sets, and maintains
class balance.

This is the final phase before model training.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from .base import PipelinePhase
from .config import PipelineConfig, BINARY_CLASSIFICATION
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class TrainValSplitPhase(PipelinePhase):
    """
    Phase 7: Create stratified train/validation split for binary classification.

    For each binary label (traffic_present, traffic_absent):
    1. Group crops by camera
    2. Split 80/20 within each camera (stratified by camera)
    3. Maintain overall class balance (50/50)
    4. Copy files to train/ and val/ directories

    Stratification ensures:
    - Each camera appears in both train and val
    - Model learns from diverse camera angles
    - No data leakage (same source image not in both sets)
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize train/val split phase.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="split",
            description="Creating train/val split for binary classification"
        )

        # Input directory
        self.crops_dir = self.config.base_dir / self.config.binary_crops_dir

        # Output directories
        self.final_dir = self.config.base_dir / self.config.binary_final_dir
        self.output_summary_path = self.config.base_dir / "binary_split_summary.json"

        # Train/val ratio
        self.train_ratio = self.config.train_ratio

    def run(self, train_ratio: float = 0.8, resume: bool = False) -> Dict[str, Any]:
        """
        Execute train/val split.

        Args:
            train_ratio: Ratio for training set (default: 0.8)
            resume: Ignored (split is fast, no resume needed)

        Returns:
            Dictionary with split summary
        """
        self.train_ratio = train_ratio
        self.logger.info(f"Train/val ratio: {train_ratio:.0%} / {1-train_ratio:.0%}")

        # Load crops
        self.logger.info("Loading crops...")
        crops_by_label = self._load_crops()

        if not any(crops_by_label.values()):
            raise ValueError("No crops found. Run Phase 5b first.")

        total_crops = sum(len(crops) for crops in crops_by_label.values())
        self.logger.info(f"Found {total_crops} total crops:")
        for label, crops in crops_by_label.items():
            self.logger.info(f"  {label}: {len(crops)} crops")

        # Ensure output directories exist
        self._ensure_split_directories()

        # Perform stratified split for each label
        splits = {}
        for label, crops in crops_by_label.items():
            self.logger.info(f"\nSplitting {label}...")
            train, val = self._stratified_split_by_camera(crops, train_ratio)
            splits[label] = {'train': train, 'val': val}

            self.logger.info(f"  Train: {len(train)} crops")
            self.logger.info(f"  Val: {len(val)} crops")

        # Copy files to final directories
        self.logger.info("\nCopying files to final directories...")
        stats = self._copy_files(splits)

        # Build summary
        summary = {
            'created_at': datetime.now().isoformat(),
            'train_ratio': train_ratio,
            'statistics': stats
        }

        # Save summary
        save_json(summary, self.output_summary_path)
        self.logger.info(f"Saved split summary to {self.output_summary_path}")

        # Log summary
        self.logger.info(f"\nSplit completed:")
        self.logger.info(f"  Train total: {stats['train_total']}")
        for label in BINARY_CLASSIFICATION['classes']:
            self.logger.info(f"    {label}: {stats['by_label'][label]['train']}")
        self.logger.info(f"  Val total: {stats['val_total']}")
        for label in BINARY_CLASSIFICATION['classes']:
            self.logger.info(f"    {label}: {stats['by_label'][label]['val']}")

        return summary

    def _load_crops(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all crops from binary_crops directory.

        Returns:
            Dict with keys 'traffic_present' and 'traffic_absent',
            values are lists of crop info dicts
        """
        crops_by_label = {}

        for label in BINARY_CLASSIFICATION['classes']:
            label_dir = self.crops_dir / label
            if not label_dir.exists():
                self.logger.warning(f"Label directory not found: {label_dir}")
                crops_by_label[label] = []
                continue

            crops = []
            for crop_path in label_dir.glob("*.jpg"):
                # Extract camera_id from filename: CAMERA_TIMESTAMP.jpg
                filename = crop_path.stem
                parts = filename.split('_', 1)
                if len(parts) >= 2:
                    camera_id = parts[0]
                else:
                    camera_id = filename

                crops.append({
                    'path': crop_path,
                    'camera_id': camera_id,
                    'label': label
                })

            crops_by_label[label] = crops

        return crops_by_label

    def _stratified_split_by_camera(
        self,
        crops: List[Dict[str, Any]],
        train_ratio: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform stratified split by camera.

        Ensures each camera is represented in both train and val sets.

        Args:
            crops: List of crop info dicts
            train_ratio: Ratio for training set

        Returns:
            Tuple of (train_crops, val_crops)
        """
        # Group by camera
        by_camera = defaultdict(list)
        for crop in crops:
            camera_id = crop['camera_id']
            by_camera[camera_id].append(crop)

        train_crops = []
        val_crops = []

        # Split within each camera
        for camera_id, camera_crops in by_camera.items():
            # Shuffle crops for this camera
            shuffled = camera_crops.copy()
            random.shuffle(shuffled)

            # Split by ratio
            split_idx = int(len(shuffled) * train_ratio)
            train_crops.extend(shuffled[:split_idx])
            val_crops.extend(shuffled[split_idx:])

            self.logger.debug(
                f"  Camera {camera_id}: {len(shuffled)} total, "
                f"{split_idx} train, {len(shuffled) - split_idx} val"
            )

        return train_crops, val_crops

    def _ensure_split_directories(self):
        """Create split output directories if they don't exist."""
        for split in ['train', 'val']:
            for label in BINARY_CLASSIFICATION['classes']:
                split_dir = self.final_dir / split / label
                split_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {split_dir}")

    def _copy_files(self, splits: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Any]:
        """
        Copy files to train/val directories.

        Args:
            splits: Dict with structure {label: {'train': [...], 'val': [...]}}

        Returns:
            Statistics dictionary
        """
        stats = {
            'train_total': 0,
            'val_total': 0,
            'by_label': {},
            'by_camera': defaultdict(lambda: {
                'train': 0,
                'val': 0
            })
        }

        for label in BINARY_CLASSIFICATION['classes']:
            stats['by_label'][label] = {'train': 0, 'val': 0}

        # Copy files
        total_files = sum(
            len(splits[label]['train']) + len(splits[label]['val'])
            for label in splits
        )

        with tqdm(total=total_files, desc="Copying files") as pbar:
            for label, label_splits in splits.items():
                for split_name in ['train', 'val']:
                    for crop_info in label_splits[split_name]:
                        source_path = crop_info['path']
                        camera_id = crop_info['camera_id']

                        # Destination path
                        dest_dir = self.final_dir / split_name / label
                        dest_path = dest_dir / source_path.name

                        # Copy file
                        shutil.copy2(source_path, dest_path)

                        # Update stats
                        stats[f'{split_name}_total'] += 1
                        stats['by_label'][label][split_name] += 1
                        stats['by_camera'][camera_id][split_name] += 1

                        pbar.update(1)

        # Convert defaultdict to regular dict for JSON serialization
        stats['by_camera'] = dict(stats['by_camera'])

        return stats

    def validate(self) -> bool:
        """
        Validate that split completed successfully.

        Returns:
            True if validation passed, False otherwise
        """
        # Check summary exists
        if not self.output_summary_path.exists():
            self.logger.error("Split summary file not found")
            return False

        # Load summary
        summary = load_json(self.output_summary_path)
        stats = summary.get('statistics', {})

        # Check that files were split
        if stats.get('train_total', 0) == 0 or stats.get('val_total', 0) == 0:
            self.logger.error("No files in train or val set")
            return False

        # Count actual files
        total_files = 0
        for split in ['train', 'val']:
            split_count = 0
            for label in BINARY_CLASSIFICATION['classes']:
                split_dir = self.final_dir / split / label
                if split_dir.exists():
                    files = list(split_dir.glob("*.jpg"))
                    split_count += len(files)
                    self.logger.info(f"  {split}/{label}: {len(files)} files")
            total_files += split_count

        # Verify counts match
        expected_total = stats['train_total'] + stats['val_total']
        if total_files != expected_total:
            self.logger.warning(
                f"File count mismatch: {total_files} files on disk, "
                f"{expected_total} in summary"
            )

        # Check class balance (within 5% tolerance)
        for split in ['train', 'val']:
            split_total = stats[f'{split}_total']
            if split_total > 0:
                for label in BINARY_CLASSIFICATION['classes']:
                    label_count = stats['by_label'][label][split]
                    ratio = label_count / split_total
                    if ratio < 0.45 or ratio > 0.55:  # Allow 45-55% (10% tolerance)
                        self.logger.warning(
                            f"{split} set class imbalance: {label} = {ratio:.1%} "
                            f"(expected ~50%)"
                        )

        self.logger.info(f"Validation passed: {total_files} files split")
        return True


def main():
    """CLI entry point for Phase 7b."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7b: Create train/val split for binary classification"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = PipelineConfig()

    # Create phase instance
    phase = TrainValSplitPhase(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Execute phase
        result = phase.execute(
            validate_after=True,
            train_ratio=args.train_ratio
        )

        if result['status'] == 'completed':
            print(f"\n✓ Phase 7b completed successfully in {result['duration_seconds']:.1f} seconds")
            summary = result['data']
            stats = summary.get('statistics', {})
            print(f"  Train total: {stats.get('train_total', 0)}")
            print(f"  Val total: {stats.get('val_total', 0)}")
            print(f"\nBy label:")
            for label in BINARY_CLASSIFICATION['classes']:
                label_stats = stats.get('by_label', {}).get(label, {})
                print(f"  {label}:")
                print(f"    Train: {label_stats.get('train', 0)}")
                print(f"    Val: {label_stats.get('val', 0)}")
            print(f"\nFinal dataset saved to: {config.binary_final_dir}/")
            print(f"\nDataset is ready for model training!")
            exit(0)
        else:
            print(f"\n✗ Phase 7b failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
