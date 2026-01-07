"""
Phase 3: Binary Selection

Selects balanced set of whole images for binary classification based on YOLO results.
Images are labeled as traffic_present (moderate/heavy) or traffic_absent (empty/light)
based on vehicle counts in polygon areas.

This phase prepares images for manual review in Phase 4.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

from .base import PipelinePhase
from .config import PipelineConfig, BINARY_LABELING_RULES
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class SelectionPhase(PipelinePhase):
    """
    Phase 3: Select and label images for binary classification.

    For each image:
    1. Check YOLO vehicle count
    2. Assign binary label: traffic_present (>=7 vehicles) or traffic_absent (<7 vehicles)
    3. Assign confidence score based on vehicle count
    4. Flag borderline cases (4-8 vehicles) for priority review

    Outputs:
    - binary_selection.json: Selected images with binary labels
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize selection phase.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="selection",
            description="Selecting and labeling images for binary classification"
        )

        # Input files
        self.yolo_results_path = self.config.base_dir / "yolo_results_filtered.json"
        if not self.yolo_results_path.exists():
            self.yolo_results_path = self.config.base_dir / "yolo_results.json"

        # Output file
        self.output_path = self.config.base_dir / "binary_selection.json"

    def run(self, target_per_class: int = 3000, resume: bool = False) -> Dict[str, Any]:
        """
        Execute binary selection.

        Args:
            target_per_class: Number of images to select per class (default: 3000)
            resume: Ignored (selection is fast, no resume needed)

        Returns:
            Dictionary with selection results
        """
        self.logger.info(f"Target: {target_per_class} images per class ({target_per_class * 2} total)")

        # Load YOLO results
        self.logger.info(f"Loading YOLO results from {self.yolo_results_path}")
        yolo_results = load_json(self.yolo_results_path)

        analyses = yolo_results.get('analyses', [])
        self.logger.info(f"Found {len(analyses)} analyzed images")

        # Group by binary label
        self.logger.info("Grouping images by binary label...")
        images_by_label = self._group_by_binary_label(analyses)

        # Stratified selection by camera
        self.logger.info("Performing stratified selection by camera...")
        selection = self._stratified_selection(
            images_by_label,
            target_per_class=target_per_class
        )

        # Build output
        output = {
            "created_at": datetime.now().isoformat(),
            "target_per_class": target_per_class,
            "yolo_source": str(self.yolo_results_path.name),
            "selection": selection,
            "statistics": self._calculate_statistics(selection)
        }

        # Save output
        save_json(output, self.output_path)
        self.logger.info(f"Saved selection to {self.output_path}")

        # Log statistics
        stats = output['statistics']
        self.logger.info(f"\nSelection completed:")
        self.logger.info(f"  Total selected: {stats['total_selected']}")
        self.logger.info(f"  traffic_present: {stats['traffic_present']}")
        self.logger.info(f"  traffic_absent: {stats['traffic_absent']}")
        self.logger.info(f"  Needs review: {stats['needs_review']}")
        self.logger.info(f"  Cameras covered: {len(stats['by_camera'])}")

        return output

    def _group_by_binary_label(self, analyses: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group images by binary label based on YOLO vehicle count.

        Args:
            analyses: List of YOLO analysis results

        Returns:
            Dict with keys 'traffic_present' and 'traffic_absent'
        """
        grouped = {
            'traffic_present': [],
            'traffic_absent': []
        }

        for analysis in analyses:
            vehicle_count = analysis.get('vehicle_count', 0)
            camera_id = analysis.get('camera_id')
            local_path = analysis.get('local_path')

            if not local_path:
                continue

            # Assign binary label based on vehicle count
            if vehicle_count >= 7:
                binary_label = 'traffic_present'
            else:
                binary_label = 'traffic_absent'

            # Assign confidence score
            confidence, needs_review = self._assign_confidence(vehicle_count)

            # Create record
            record = {
                'camera_id': camera_id,
                'local_path': local_path,
                'yolo_count': vehicle_count,
                'binary_label': binary_label,
                'confidence': confidence,
                'needs_review': needs_review
            }

            grouped[binary_label].append(record)

        self.logger.info(f"  traffic_present: {len(grouped['traffic_present'])} images")
        self.logger.info(f"  traffic_absent: {len(grouped['traffic_absent'])} images")

        return grouped

    def _assign_confidence(self, vehicle_count: int) -> Tuple[float, bool]:
        """
        Assign confidence score based on vehicle count.

        Args:
            vehicle_count: Number of vehicles detected by YOLO

        Returns:
            (confidence_score, needs_review_flag)
        """
        if vehicle_count >= 15:
            return (0.95, False)  # Very confident - heavy traffic
        elif vehicle_count >= 9:
            return (0.85, False)  # Confident - clear moderate
        elif vehicle_count >= 7:
            return (0.75, True)   # Moderate confidence - border of moderate/light
        elif vehicle_count >= 4:
            return (0.70, True)   # Low confidence - borderline
        elif vehicle_count >= 1:
            return (0.85, False)  # Confident - clear light
        else:  # 0 vehicles
            return (0.95, False)  # Very confident - empty

    def _stratified_selection(
        self,
        images_by_label: Dict[str, List[Dict]],
        target_per_class: int
    ) -> List[Dict]:
        """
        Perform stratified selection by camera.

        Ensures each camera is represented in both classes (if possible).

        Args:
            images_by_label: Images grouped by binary label
            target_per_class: Number of images to select per class

        Returns:
            List of selected images
        """
        # Group by camera for each label
        by_camera = {
            'traffic_present': defaultdict(list),
            'traffic_absent': defaultdict(list)
        }

        for label, images in images_by_label.items():
            for img in images:
                camera_id = img['camera_id']
                by_camera[label][camera_id].append(img)

        # Calculate target per camera per class
        # Use set union to get all unique cameras
        all_cameras = set(by_camera['traffic_present'].keys()) | set(by_camera['traffic_absent'].keys())
        num_cameras = len(all_cameras)
        target_per_camera = target_per_class // num_cameras

        self.logger.info(f"  Cameras found: {num_cameras}")
        self.logger.info(f"  Target per camera per class: {target_per_camera}")

        # Select from each camera
        selection = []
        for label in ['traffic_present', 'traffic_absent']:
            camera_images = by_camera[label]

            for camera_id in sorted(all_cameras):
                available = camera_images.get(camera_id, [])

                if not available:
                    self.logger.warning(
                        f"  Camera {camera_id} has no {label} images, skipping"
                    )
                    continue

                # Select target_per_camera or all available (whichever is smaller)
                to_select = min(target_per_camera, len(available))
                selected = random.sample(available, to_select)
                selection.extend(selected)

                if to_select < target_per_camera:
                    self.logger.warning(
                        f"  Camera {camera_id} has only {len(available)} {label} images "
                        f"(target: {target_per_camera})"
                    )

        self.logger.info(f"  Selected {len(selection)} images total")

        return selection

    def _calculate_statistics(self, selection: List[Dict]) -> Dict[str, Any]:
        """
        Calculate statistics for the selection.

        Args:
            selection: List of selected images

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_selected': len(selection),
            'traffic_present': 0,
            'traffic_absent': 0,
            'needs_review': 0,
            'by_camera': defaultdict(lambda: {
                'total': 0,
                'traffic_present': 0,
                'traffic_absent': 0
            })
        }

        for img in selection:
            label = img['binary_label']
            camera_id = img['camera_id']

            if label == 'traffic_present':
                stats['traffic_present'] += 1
            else:
                stats['traffic_absent'] += 1

            if img.get('needs_review', False):
                stats['needs_review'] += 1

            stats['by_camera'][camera_id]['total'] += 1
            stats['by_camera'][camera_id][label] += 1

        # Convert defaultdict to regular dict for JSON serialization
        stats['by_camera'] = dict(stats['by_camera'])

        return stats

    def validate(self) -> bool:
        """
        Validate that binary selection completed successfully.

        Returns:
            True if validation passed, False otherwise
        """
        if not self.output_path.exists():
            self.logger.error("Binary selection file not found")
            return False

        try:
            selection_data = load_json(self.output_path)

            if 'selection' not in selection_data:
                self.logger.error("Missing 'selection' in output")
                return False

            stats = selection_data.get('statistics', {})
            total = stats.get('total_selected', 0)

            if total == 0:
                self.logger.error("No images were selected")
                return False

            self.logger.info(f"Validation passed: {total} images selected")
            return True

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False


def main():
    """CLI entry point for Phase 3."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3: Select and label images for binary classification"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=3000,
        help="Number of images to select per class (default: 3000)"
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
    phase = SelectionPhase(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Execute phase
        result = phase.execute(validate_after=True, target_per_class=args.target)

        if result['status'] == 'completed':
            print(f"\n✓ Phase 3 completed successfully in {result['duration_seconds']:.1f} seconds")
            data = result['data']
            stats = data.get('statistics', {})
            print(f"  Total selected: {stats.get('total_selected', 0)}")
            print(f"  traffic_present: {stats.get('traffic_present', 0)}")
            print(f"  traffic_absent: {stats.get('traffic_absent', 0)}")
            print(f"  Needs review: {stats.get('needs_review', 0)}")
            print(f"\nSelected images saved to: binary_selection.json")
            print(f"Next step: Run Phase 4 to manually review and correct labels")
            exit(0)
        else:
            print(f"\n✗ Phase 3 failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
