"""
Phase 7: Train/Val Split

Creates an 80/20 train/validation split from labeled images,
stratified by traffic density class to ensure balanced representation.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

from .base import PipelinePhase
from .config import PipelineConfig, TRAFFIC_LEVELS
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class TrainValSplitPhase(PipelinePhase):
    """
    Phase 7: Train/validation split.

    Creates stratified 80/20 train/validation split from labeled images.
    Each traffic level is split independently to maintain class balance.

    Directory structure created:
    final/
    ├── train/
    │   ├── empty/
    │   ├── light/
    │   ├── moderate/
    │   └── heavy/
    └── val/
        ├── empty/
        ├── light/
        ├── moderate/
        └── heavy/
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
            description="Creating train/validation split"
        )

        self.labeled_dir = self.config.base_dir / "labeled"
        self.final_dir = self.config.base_dir / "final"
        self.train_dir = self.final_dir / "train"
        self.val_dir = self.final_dir / "val"
        self.split_manifest_path = self.config.base_dir / "split_manifest.json"

        # Split configuration
        self.train_ratio = self.config.train_ratio
        self.random_seed = 42  # For reproducibility

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute train/val split.

        Args:
            resume: If True, skip if output already exists

        Returns:
            Dictionary with split statistics
        """
        # Check if already completed and resume requested
        if resume and self._check_existing_split():
            self.logger.info("Train/val split already completed, loading existing manifest")
            manifest = load_json(self.split_manifest_path)
            return {
                "manifest": manifest,
                "resumed": True
            }

        # Validate input directory
        if not self.labeled_dir.exists():
            raise FileNotFoundError(
                f"Labeled directory not found at {self.labeled_dir}. "
                "Run Phase 6 (labeling tool) first."
            )

        # Check if we have any labeled images
        total_labeled = self._count_labeled_images()
        if total_labeled == 0:
            raise ValueError(
                f"No labeled images found in {self.labeled_dir}. "
                "Run Phase 6 (labeling tool) to label images first."
            )

        self.logger.info(f"Found {total_labeled} labeled images")

        # Clean up existing split directories
        if self.final_dir.exists() and not resume:
            self.logger.info("Cleaning up existing split directories")
            shutil.rmtree(self.final_dir)

        # Create output directories
        self._create_directories()

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Split each class independently
        split_stats = {}
        file_manifest = {
            "train": {},
            "val": {}
        }

        for level in TRAFFIC_LEVELS:
            self.logger.info(f"Processing class: {level}")

            # Get all images for this class
            level_dir = self.labeled_dir / level
            if not level_dir.exists():
                self.logger.warning(f"Class directory not found: {level_dir}")
                split_stats[level] = {
                    "total": 0,
                    "train": 0,
                    "val": 0
                }
                continue

            images = sorted(list(level_dir.glob("*.jpg")))

            if not images:
                self.logger.warning(f"No images found for class: {level}")
                split_stats[level] = {
                    "total": 0,
                    "train": 0,
                    "val": 0
                }
                continue

            # Shuffle for random split
            random.shuffle(images)

            # Calculate split index
            total_count = len(images)
            train_count = int(total_count * self.train_ratio)
            val_count = total_count - train_count

            # Split images
            train_images = images[:train_count]
            val_images = images[train_count:]

            # Copy to train directory
            train_dest_dir = self.train_dir / level
            file_manifest["train"][level] = []
            for img_path in train_images:
                dest_path = train_dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                file_manifest["train"][level].append(img_path.name)

            # Copy to val directory
            val_dest_dir = self.val_dir / level
            file_manifest["val"][level] = []
            for img_path in val_images:
                dest_path = val_dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                file_manifest["val"][level].append(img_path.name)

            # Record statistics
            split_stats[level] = {
                "total": total_count,
                "train": train_count,
                "val": val_count,
                "train_ratio": train_count / total_count if total_count > 0 else 0
            }

            self.logger.info(
                f"  {level}: {total_count} total → "
                f"{train_count} train ({train_count/total_count*100:.1f}%), "
                f"{val_count} val ({val_count/total_count*100:.1f}%)"
            )

        # Calculate overall statistics
        total_train = sum(stats["train"] for stats in split_stats.values())
        total_val = sum(stats["val"] for stats in split_stats.values())
        total = total_train + total_val

        # Create manifest
        manifest = {
            "split_date": datetime.now().isoformat(),
            "random_seed": self.random_seed,
            "train_ratio": self.train_ratio,
            "source_dir": str(self.labeled_dir),
            "output_dir": str(self.final_dir),
            "total_images": total,
            "train_images": total_train,
            "val_images": total_val,
            "actual_train_ratio": total_train / total if total > 0 else 0,
            "class_stats": split_stats,
            "file_manifest": file_manifest
        }

        # Save manifest
        self.logger.info(f"Saving split manifest to {self.split_manifest_path}")
        save_json(manifest, self.split_manifest_path)

        # Log overall summary
        self._log_summary(manifest)

        return {
            "manifest": manifest,
            "resumed": False
        }

    def validate(self) -> bool:
        """
        Validate that train/val split completed successfully.

        Returns:
            True if validation passed, False otherwise
        """
        # Check output directories exist
        if not self.final_dir.exists():
            self.logger.error(f"Final directory not found: {self.final_dir}")
            return False

        if not self.train_dir.exists():
            self.logger.error(f"Train directory not found: {self.train_dir}")
            return False

        if not self.val_dir.exists():
            self.logger.error(f"Val directory not found: {self.val_dir}")
            return False

        # Check manifest exists
        if not self.split_manifest_path.exists():
            self.logger.error(f"Manifest file not found: {self.split_manifest_path}")
            return False

        # Load and validate manifest
        try:
            manifest = load_json(self.split_manifest_path)
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            return False

        # Validate required fields
        required_fields = [
            "split_date", "train_ratio", "total_images",
            "train_images", "val_images", "class_stats"
        ]
        for field in required_fields:
            if field not in manifest:
                self.logger.error(f"Missing required field in manifest: {field}")
                return False

        # Validate class directories
        for level in TRAFFIC_LEVELS:
            train_class_dir = self.train_dir / level
            val_class_dir = self.val_dir / level

            if not train_class_dir.exists():
                self.logger.error(f"Train class directory not found: {train_class_dir}")
                return False

            if not val_class_dir.exists():
                self.logger.error(f"Val class directory not found: {val_class_dir}")
                return False

        # Count actual files and compare with manifest
        actual_train_count = self._count_images_in_split(self.train_dir)
        actual_val_count = self._count_images_in_split(self.val_dir)
        manifest_train_count = manifest["train_images"]
        manifest_val_count = manifest["val_images"]

        if actual_train_count != manifest_train_count:
            self.logger.error(
                f"Train image count mismatch: found {actual_train_count}, "
                f"manifest says {manifest_train_count}"
            )
            return False

        if actual_val_count != manifest_val_count:
            self.logger.error(
                f"Val image count mismatch: found {actual_val_count}, "
                f"manifest says {manifest_val_count}"
            )
            return False

        # Validate split ratio
        total = actual_train_count + actual_val_count
        if total > 0:
            actual_train_ratio = actual_train_count / total
            expected_train_ratio = self.train_ratio

            # Allow 5% tolerance (e.g., due to rounding with small datasets)
            if abs(actual_train_ratio - expected_train_ratio) > 0.05:
                self.logger.warning(
                    f"Train ratio deviation: actual={actual_train_ratio:.3f}, "
                    f"expected={expected_train_ratio:.3f}"
                )

        self.logger.info("✓ Validation passed")
        self.logger.info(f"  Train: {actual_train_count} images")
        self.logger.info(f"  Val: {actual_val_count} images")
        self.logger.info(f"  Total: {total} images")

        return True

    def _count_labeled_images(self) -> int:
        """Count total labeled images across all classes."""
        count = 0
        for level in TRAFFIC_LEVELS:
            level_dir = self.labeled_dir / level
            if level_dir.exists():
                count += len(list(level_dir.glob("*.jpg")))
        return count

    def _create_directories(self):
        """Create output directory structure."""
        self.logger.info("Creating output directories")

        # Create base directories
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        # Create class subdirectories
        for level in TRAFFIC_LEVELS:
            (self.train_dir / level).mkdir(parents=True, exist_ok=True)
            (self.val_dir / level).mkdir(parents=True, exist_ok=True)

    def _check_existing_split(self) -> bool:
        """
        Check if a valid split already exists.

        Returns:
            True if valid split exists
        """
        if not self.split_manifest_path.exists():
            return False

        if not self.train_dir.exists() or not self.val_dir.exists():
            return False

        # Check if directories have images
        train_count = self._count_images_in_split(self.train_dir)
        val_count = self._count_images_in_split(self.val_dir)

        return train_count > 0 and val_count > 0

    def _count_images_in_split(self, split_dir: Path) -> int:
        """
        Count images in a split directory (train or val).

        Args:
            split_dir: Path to train or val directory

        Returns:
            Total number of images
        """
        count = 0
        for level in TRAFFIC_LEVELS:
            level_dir = split_dir / level
            if level_dir.exists():
                count += len(list(level_dir.glob("*.jpg")))
        return count

    def _log_summary(self, manifest: Dict[str, Any]):
        """
        Log summary of split operation.

        Args:
            manifest: Split manifest dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("Train/Val Split Summary")
        self.logger.info("=" * 60)

        total = manifest["total_images"]
        train = manifest["train_images"]
        val = manifest["val_images"]

        self.logger.info(f"Total images: {total}")
        self.logger.info(f"  Train: {train} ({train/total*100:.1f}%)")
        self.logger.info(f"  Val: {val} ({val/total*100:.1f}%)")
        self.logger.info(f"Random seed: {manifest['random_seed']}")

        self.logger.info("\nPer-class distribution:")
        for level in TRAFFIC_LEVELS:
            stats = manifest["class_stats"][level]
            self.logger.info(
                f"  {level:10s}: {stats['total']:5d} total → "
                f"{stats['train']:5d} train, {stats['val']:5d} val"
            )

        self.logger.info("=" * 60)


def main():
    """CLI entry point for Phase 7."""
    import argparse
    from .config import PipelineConfig

    parser = argparse.ArgumentParser(
        description="Phase 7: Train/Validation Split"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous split (skip if already exists)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing split without running"
    )

    args = parser.parse_args()

    # Initialize
    config = PipelineConfig()
    phase = TrainValSplitPhase(config)

    if args.validate_only:
        print("Validating train/val split...")
        is_valid = phase.validate()
        if is_valid:
            print("✓ Validation passed")
        else:
            print("✗ Validation failed")
        return

    # Run
    print("\n" + "=" * 60)
    print("Phase 7: Train/Validation Split")
    print("=" * 60)
    print(f"\nCreating {int(config.train_ratio*100)}/{int((1-config.train_ratio)*100)} "
          f"train/val split from labeled images\n")

    if args.resume:
        print("Resuming from previous split...\n")

    result = phase.execute(resume=args.resume)

    if result["status"] == "completed":
        print("\n✓ Phase 7 completed successfully")
        print(f"Duration: {result['duration_seconds']:.1f} seconds")

        # Print summary
        manifest = result["data"]["manifest"]
        print(f"\nTotal: {manifest['total_images']} images")
        print(f"  Train: {manifest['train_images']} ({manifest['actual_train_ratio']*100:.1f}%)")
        print(f"  Val: {manifest['val_images']} ({(1-manifest['actual_train_ratio'])*100:.1f}%)")
    else:
        print(f"\n✗ Phase 7 failed: {result.get('reason', 'Unknown error')}")


if __name__ == "__main__":
    main()
