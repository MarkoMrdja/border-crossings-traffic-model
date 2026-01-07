"""
Phase 6: Crop Failure Regions with Label Inheritance

Crops failure regions from manually reviewed images.
Crops inherit binary labels from their source images (no separate review needed).

This phase prepares cropped images for train/val split in Phase 7.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from .base import PipelinePhase
from .config import PipelineConfig, BINARY_CLASSIFICATION
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class CropPhase(PipelinePhase):
    """
    Phase 6: Crop failure regions from reviewed images with label inheritance.

    For each reviewed image:
    1. Get the image's binary label from directory structure
    2. Get the single failure region polygon for the camera
    3. Crop the failure region (128×128 pixels)
    4. Save crop to binary_crops/{label}/ with inherited label

    Label Inheritance:
    - All crops from "traffic_present" images are labeled "traffic_present"
    - All crops from "traffic_absent" images are labeled "traffic_absent"
    - Confidence = 1.0 (human-reviewed at image level)
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize crop phase.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="crop",
            description="Cropping failure regions with label inheritance"
        )

        # Input directories/files
        self.labeled_dir = self.config.base_dir / self.config.binary_labeled_dir
        self.failure_regions_path = self.config.base_dir / "yolo_failure_regions.json"

        # Output directory
        self.crops_dir = self.config.base_dir / self.config.binary_crops_dir
        self.output_metadata_path = self.config.base_dir / "binary_crop_metadata.json"

        # Crop size
        self.crop_size = BINARY_CLASSIFICATION['crop_size']

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute binary cropping with label inheritance.

        Args:
            resume: If True, skip already cropped images

        Returns:
            Dictionary with cropping results
        """
        # Load failure regions
        self.logger.info(f"Loading failure regions from {self.failure_regions_path}")
        if not self.failure_regions_path.exists():
            raise FileNotFoundError(
                f"Failure regions not found at {self.failure_regions_path}. "
                "Run Phase 4d (phase4d_exclusion_zones) first."
            )

        failure_regions = load_json(self.failure_regions_path)
        failure_regions_by_camera = failure_regions.get('cameras', {})

        # Get reviewed images
        self.logger.info("Finding reviewed images...")
        reviewed_images = self._get_reviewed_images()

        if not reviewed_images:
            raise ValueError("No reviewed images found. Run Phase 4 first.")

        self.logger.info(f"Found {len(reviewed_images)} reviewed images")

        # Ensure output directories exist
        self._ensure_crop_directories()

        # Initialize progress tracking
        if resume and self.progress_file.exists():
            self.load_progress()
            processed_files = set(self.progress.get('processed_files', []))
            self.logger.info(f"Resuming: {len(processed_files)} files already processed")
        else:
            processed_files = set()
            self.initialize_progress(
                total=len(reviewed_images),
                processed=0,
                failed=0,
                skipped=0,
                processed_files=[]
            )

        # Statistics
        stats = {
            'total_images': len(reviewed_images),
            'processed': len(processed_files),
            'failed': 0,
            'skipped': 0,
            'skipped_no_failure_region': 0,
            'skipped_multiple_regions': 0,
            'by_label': {
                'traffic_present': 0,
                'traffic_absent': 0
            },
            'by_camera': {}
        }

        # Metadata for all crops
        crops_metadata = []

        # Process each reviewed image
        self.logger.info(f"Processing {len(reviewed_images)} images...")

        with tqdm(total=len(reviewed_images), initial=len(processed_files), desc="Cropping images") as pbar:
            for image_info in reviewed_images:
                image_path = image_info['path']
                label = image_info['label']
                camera_id = image_info['camera_id']

                # Skip if already processed
                if str(image_path) in processed_files:
                    pbar.update(1)
                    continue

                # Get failure region for this camera
                camera_data = failure_regions_by_camera.get(camera_id, {})
                regions = camera_data.get('failure_regions', [])

                if not regions:
                    self.logger.warning(f"No failure region for camera {camera_id}, skipping {image_path.name}")
                    stats['skipped'] += 1
                    stats['skipped_no_failure_region'] += 1
                    processed_files.add(str(image_path))
                    pbar.update(1)
                    continue

                # Enforce single region per camera
                if len(regions) != 1:
                    self.logger.warning(
                        f"Camera {camera_id} has {len(regions)} regions (expected 1), using first region"
                    )
                    stats['skipped_multiple_regions'] += 1

                region = regions[0]
                polygon = region['polygon']

                try:
                    # Crop failure region
                    crop = self._crop_roi(
                        str(image_path),
                        polygon,
                        output_size=self.crop_size
                    )

                    # Generate output filename (no region index - only 1 region per camera)
                    timestamp = image_path.stem
                    filename = f"{camera_id}_{timestamp}.jpg"
                    output_path = self.crops_dir / label / filename

                    # Save crop
                    cv2.imwrite(str(output_path), crop)

                    # Update statistics
                    stats['processed'] += 1
                    stats['by_label'][label] += 1

                    if camera_id not in stats['by_camera']:
                        stats['by_camera'][camera_id] = {
                            'total': 0,
                            'traffic_present': 0,
                            'traffic_absent': 0
                        }
                    stats['by_camera'][camera_id]['total'] += 1
                    stats['by_camera'][camera_id][label] += 1

                    # Track metadata
                    crops_metadata.append({
                        'filename': filename,
                        'source_image': str(image_path.relative_to(self.config.base_dir)),
                        'camera_id': camera_id,
                        'binary_label': label,
                        'confidence': 1.0,  # Human-reviewed at image level
                        'needs_review': False
                    })

                    processed_files.add(str(image_path))

                except Exception as e:
                    self.logger.error(f"Failed to crop {image_path}: {e}")
                    stats['failed'] += 1

                # Update progress periodically
                if len(processed_files) % 100 == 0:
                    self.progress['processed'] = stats['processed']
                    self.progress['failed'] = stats['failed']
                    self.progress['skipped'] = stats['skipped']
                    self.progress['processed_files'] = list(processed_files)
                    self.save_progress()

                pbar.update(1)

        # Final progress save
        self.progress['processed'] = stats['processed']
        self.progress['failed'] = stats['failed']
        self.progress['skipped'] = stats['skipped']
        self.progress['processed_files'] = list(processed_files)
        self.progress['completed_at'] = datetime.now().isoformat()
        self.save_progress()

        # Save metadata
        metadata_output = {
            'created_at': datetime.now().isoformat(),
            'source': 'Cropped from manually reviewed binary_labeled/ images',
            'label_inheritance': 'All crops inherit label from source image',
            'crop_size': self.crop_size,
            'crops': crops_metadata,
            'statistics': stats
        }
        save_json(metadata_output, self.output_metadata_path)

        # Log summary
        self.logger.info(f"\nCropping completed:")
        self.logger.info(f"  Total crops: {stats['processed']}")
        self.logger.info(f"  traffic_present: {stats['by_label']['traffic_present']}")
        self.logger.info(f"  traffic_absent: {stats['by_label']['traffic_absent']}")
        self.logger.info(f"  Failed: {stats['failed']}")
        self.logger.info(f"  Skipped: {stats['skipped']}")
        if stats['skipped_no_failure_region'] > 0:
            self.logger.info(f"    (no failure region: {stats['skipped_no_failure_region']})")

        return metadata_output

    def _get_reviewed_images(self) -> List[Dict[str, Any]]:
        """
        Get all reviewed images from binary_labeled directory.

        Returns:
            List of dicts with 'path', 'label', 'camera_id'
        """
        reviewed = []

        for label in BINARY_CLASSIFICATION['classes']:
            label_dir = self.labeled_dir / label
            if not label_dir.exists():
                continue

            for image_path in label_dir.glob("*.jpg"):
                # Extract camera_id from filename: CAMERA_TIMESTAMP.jpg
                filename = image_path.stem
                parts = filename.split('_', 1)
                if len(parts) >= 2:
                    camera_id = parts[0]
                else:
                    camera_id = filename

                reviewed.append({
                    'path': image_path,
                    'label': label,
                    'camera_id': camera_id
                })

        return reviewed

    def _ensure_crop_directories(self):
        """Create crop output directories if they don't exist."""
        for label in BINARY_CLASSIFICATION['classes']:
            label_dir = self.crops_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {label_dir}")

    def _crop_roi(
        self,
        image_path: str,
        polygon: List[List[int]],
        output_size: int = 64
    ) -> np.ndarray:
        """
        Crop polygon region from image.

        Process:
        1. Calculate bounding box of polygon
        2. Crop bounding box from image
        3. Create mask for polygon
        4. Apply mask (black outside polygon)
        5. Resize to output_size × output_size

        Args:
            image_path: Path to source image
            polygon: List of [x, y] coordinates defining the polygon
            output_size: Target size for output (default: 64)

        Returns:
            Cropped and resized image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert polygon to numpy array
        polygon_np = np.array(polygon, dtype=np.int32)

        # Get bounding box
        x_min, y_min = polygon_np.min(axis=0)
        x_max, y_max = polygon_np.max(axis=0)

        # Clamp to image bounds
        h, w = image.shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        # Crop bounding box
        cropped = image[y_min:y_max, x_min:x_max].copy()

        # Create mask for polygon
        mask_h = y_max - y_min
        mask_w = x_max - x_min
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        # Shift polygon coordinates to cropped image space
        shifted_polygon = polygon_np - [x_min, y_min]

        # Fill polygon in mask
        cv2.fillPoly(mask, [shifted_polygon.astype(np.int32)], 255)

        # Apply mask (black outside polygon)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        cropped = (cropped * mask_3ch).astype(np.uint8)

        # Resize to target size
        resized = cv2.resize(
            cropped,
            (output_size, output_size),
            interpolation=cv2.INTER_AREA
        )

        return resized

    def validate(self) -> bool:
        """
        Validate that cropping completed successfully.

        Returns:
            True if validation passed, False otherwise
        """
        # Check metadata exists
        if not self.output_metadata_path.exists():
            self.logger.error("Crop metadata file not found")
            return False

        # Load metadata
        metadata = load_json(self.output_metadata_path)
        stats = metadata.get('statistics', {})

        # Check that some images were processed
        if stats.get('processed', 0) == 0:
            self.logger.error("No images were processed")
            return False

        # Count actual files in crops directories
        total_files = 0
        for label in BINARY_CLASSIFICATION['classes']:
            label_dir = self.crops_dir / label
            if label_dir.exists():
                files = list(label_dir.glob("*.jpg"))
                total_files += len(files)
                self.logger.info(f"  {label}: {len(files)} crops")

        # Verify file count matches processed count
        if total_files != stats['processed']:
            self.logger.warning(
                f"File count mismatch: {total_files} files on disk, "
                f"{stats['processed']} in metadata"
            )

        self.logger.info(f"Validation passed: {total_files} cropped images found")
        return True


def main():
    """CLI entry point for Phase 5b."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5b: Crop failure regions from reviewed images"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
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
    phase = CropPhase(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        if result['status'] == 'completed':
            print(f"\n✓ Phase 5b completed successfully in {result['duration_seconds']:.1f} seconds")
            metadata = result['data']
            stats = metadata.get('statistics', {})
            print(f"  Total crops: {stats.get('processed', 0)}")
            print(f"  traffic_present: {stats.get('by_label', {}).get('traffic_present', 0)}")
            print(f"  traffic_absent: {stats.get('by_label', {}).get('traffic_absent', 0)}")
            print(f"  Failed: {stats.get('failed', 0)}")
            print(f"  Skipped: {stats.get('skipped', 0)}")
            print(f"\nCropped images saved to: {config.binary_crops_dir}/")
            print(f"Next step: Run Phase 7b (phase7b_split_binary) to create train/val split")
            exit(0)
        else:
            print(f"\n✗ Phase 5b failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
