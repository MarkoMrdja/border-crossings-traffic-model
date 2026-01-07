"""
Phase 4: Manual Review of Binary Labels

Interactive tool for reviewing and correcting binary labels on whole images.
Shows full images with YOLO annotations and allows user to confirm or correct labels.

This phase prepares images for failure region annotation in Phase 5.
"""

import logging
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import PipelinePhase
from .config import PipelineConfig
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class ReviewTool(PipelinePhase):
    """
    Phase 4: Interactive tool for reviewing and correcting binary labels on whole images.

    Shows:
    - Full image (resized to fit screen)
    - YOLO boxes (green)
    - Current binary label and confidence
    - YOLO vehicle count

    Controls:
    - [1] Label as traffic_present
    - [2] Label as traffic_absent
    - [Enter] Accept current label
    - [U] Mark as uncertain (exclude from dataset)
    - [N] Next image
    - [P] Previous image
    - [Q] Quit and save progress
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize review tool.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="binary_review",
            description="Reviewing and correcting binary labels on images"
        )

        # Input files
        self.selection_path = self.config.base_dir / "binary_selection.json"
        self.yolo_results_path = self.config.base_dir / "yolo_results_filtered.json"
        if not self.yolo_results_path.exists():
            self.yolo_results_path = self.config.base_dir / "yolo_results.json"
        self.lane_polygons_path = self.config.base_dir / "lane_polygons.json"

        # Output directories
        self.labeled_dir = self.config.base_dir / self.config.binary_labeled_dir
        self.output_log_path = self.config.base_dir / "binary_review_log.json"

        # GUI state
        self.current_index = 0
        self.window_name = "Binary Label Review"
        self.display_width = 1200
        self.display_height = 900

        # Review queue
        self.review_queue: List[Dict] = []
        self.yolo_lookup: Dict[str, Dict] = {}
        self.lane_polygons: Dict[str, Any] = {}

        # Review log
        self.review_log = {
            'reviewed': 0,
            'corrections': 0,
            'agreement_rate': 0.0,
            'excluded_uncertain': 0,
            'by_label': {
                'traffic_absent': {
                    'reviewed': 0,
                    'confirmed': 0,
                    'corrected_to_present': 0,
                    'marked_uncertain': 0
                },
                'traffic_present': {
                    'reviewed': 0,
                    'confirmed': 0,
                    'corrected_to_absent': 0,
                    'marked_uncertain': 0
                }
            },
            'correction_patterns': {
                'over_labeled_as_present': 0,
                'under_labeled_as_absent': 0
            }
        }

    def run(self, review_all: bool = True, resume: bool = False) -> Dict[str, Any]:
        """
        Execute binary label review.

        Args:
            review_all: If True, review all images. If False, review only borderline cases
            resume: If True, skip already reviewed images

        Returns:
            Dictionary with review results
        """
        # Load required data
        self.logger.info("Loading input data...")
        selection_data = load_json(self.selection_path)
        self._load_yolo_results()
        self._load_lane_polygons()

        # Build review queue
        self.logger.info("Building review queue...")
        self.review_queue = self._build_review_queue(
            selection_data['selection'],
            review_all=review_all
        )

        # Filter already reviewed if resuming
        if resume and self.labeled_dir.exists():
            self.review_queue = self._filter_reviewed(self.review_queue)
            self.logger.info(f"Resuming: {len(self.review_queue)} images remaining")

        if not self.review_queue:
            self.logger.info("No images to review")
            return self.review_log

        self.logger.info(f"Review queue: {len(self.review_queue)} images")
        self.logger.info("\nStarting interactive review...")

        # Setup OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        # Review loop
        self.current_index = 0
        while self.current_index < len(self.review_queue):
            item = self.review_queue[self.current_index]

            # Display and get user input
            action = self._review_image(item)

            if action == 'next':
                self.current_index += 1
            elif action == 'previous':
                if self.current_index > 0:
                    self.current_index -= 1
            elif action == 'quit':
                self.logger.info("User requested quit")
                break

        # Cleanup
        cv2.destroyAllWindows()

        # Calculate final statistics
        self._calculate_final_statistics()

        # Save review log
        save_json(self.review_log, self.output_log_path)
        self.logger.info(f"\nReview log saved to {self.output_log_path}")

        # Log summary
        self.logger.info(f"\nReview completed:")
        self.logger.info(f"  Reviewed: {self.review_log['reviewed']}")
        self.logger.info(f"  Corrections: {self.review_log['corrections']}")
        self.logger.info(f"  Agreement rate: {self.review_log['agreement_rate']:.1%}")
        self.logger.info(f"  Excluded uncertain: {self.review_log['excluded_uncertain']}")

        return self.review_log

    def _load_yolo_results(self):
        """Load YOLO results and build lookup."""
        yolo_data = load_json(self.yolo_results_path)
        self.yolo_lookup = {}

        for analysis in yolo_data.get('analyses', []):
            local_path = analysis.get('local_path')
            if local_path:
                self.yolo_lookup[local_path] = analysis

        self.logger.info(f"  Loaded {len(self.yolo_lookup)} YOLO results")

    def _load_lane_polygons(self):
        """Load lane polygons."""
        if not self.lane_polygons_path.exists():
            self.logger.warning("Lane polygons not found, continuing without visualization")
            self.lane_polygons = {}
            return

        data = load_json(self.lane_polygons_path)
        self.lane_polygons = data.get('cameras', {})
        self.logger.info(f"  Loaded {len(self.lane_polygons)} lane polygons")

    def _build_review_queue(
        self,
        selection: List[Dict],
        review_all: bool
    ) -> List[Dict]:
        """
        Build review queue prioritizing borderline cases.

        Args:
            selection: Selected images from Phase 3
            review_all: If True, review all. If False, review only borderline

        Returns:
            Sorted list of images to review
        """
        if review_all:
            # Review all images, but prioritize borderline cases
            queue = sorted(
                selection,
                key=lambda x: (not x.get('needs_review', False), x.get('confidence', 1.0))
            )
            self.logger.info(f"  Reviewing all {len(queue)} images")
        else:
            # Review only borderline cases + random sample of confident
            borderline = [img for img in selection if img.get('needs_review', False)]
            confident = [img for img in selection if not img.get('needs_review', False)]

            # Sample 500 random confident cases for quality control
            import random
            confident_sample = random.sample(confident, min(500, len(confident)))

            queue = borderline + confident_sample
            self.logger.info(f"  Reviewing {len(borderline)} borderline + {len(confident_sample)} confident samples")

        return queue

    def _filter_reviewed(self, queue: List[Dict]) -> List[Dict]:
        """Filter out already reviewed images."""
        filtered = []

        for item in queue:
            local_path = item['local_path']
            camera_id = item['camera_id']
            timestamp = Path(local_path).stem

            # Check if any label directory has this file
            found = False
            for label in ['traffic_present', 'traffic_absent', 'uncertain']:
                output_path = self.labeled_dir / label / f"{camera_id}_{timestamp}.jpg"
                if output_path.exists():
                    found = True
                    break

            if not found:
                filtered.append(item)

        return filtered

    def _review_image(self, item: Dict) -> str:
        """
        Display image and get user input.

        Args:
            item: Image item to review

        Returns:
            Action: 'next', 'previous', 'quit'
        """
        # Load image
        image_path = self.config.base_dir / item['local_path']
        if not image_path.exists():
            self.logger.error(f"Image not found: {image_path}")
            return 'next'

        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return 'next'

        # Get YOLO data
        yolo_data = self.yolo_lookup.get(item['local_path'], {})
        boxes = yolo_data.get('boxes', [])

        # Get lane polygon
        camera_id = item['camera_id']
        lane_data = self.lane_polygons.get(camera_id, {})
        polygons = lane_data.get('polygons', [])
        lane_polygon = np.array(polygons[0]['polygon'], dtype=np.int32) if polygons else None

        # Current label
        current_label = item.get('binary_label', 'unknown')
        pending_label = current_label  # User can change this

        while True:
            # Render display
            display = self._render_display(
                image,
                boxes,
                lane_polygon,
                item,
                pending_label
            )

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):  # Label as traffic_present
                pending_label = 'traffic_present'

            elif key == ord('2'):  # Label as traffic_absent
                pending_label = 'traffic_absent'

            elif key == ord('u') or key == ord('U'):  # Mark as uncertain
                self._save_reviewed_image(
                    image_path,
                    camera_id,
                    'uncertain',
                    current_label
                )
                self.review_log['excluded_uncertain'] += 1
                self.logger.info(f"Marked as uncertain: {image_path.name}")
                return 'next'

            elif key == 13:  # Enter - accept current label
                self._save_reviewed_image(
                    image_path,
                    camera_id,
                    pending_label,
                    current_label
                )
                return 'next'

            elif key == ord('n') or key == ord('N'):  # Next
                self._save_reviewed_image(
                    image_path,
                    camera_id,
                    pending_label,
                    current_label
                )
                return 'next'

            elif key == ord('p') or key == ord('P'):  # Previous
                return 'previous'

            elif key == ord('q') or key == ord('Q'):  # Quit
                # Save current before quitting
                self._save_reviewed_image(
                    image_path,
                    camera_id,
                    pending_label,
                    current_label
                )
                return 'quit'

    def _render_display(
        self,
        image: np.ndarray,
        boxes: List[Dict],
        lane_polygon: Optional[np.ndarray],
        item: Dict,
        pending_label: str
    ) -> np.ndarray:
        """Render the review display."""
        # Resize image to fit display
        h, w = image.shape[:2]
        scale = min(self.display_width / w, self.display_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        display = cv2.resize(image, (new_w, new_h))

        # Draw lane polygon
        if lane_polygon is not None:
            scaled_polygon = (lane_polygon * scale).astype(np.int32)
            cv2.polylines(display, [scaled_polygon], True, (255, 255, 0), 2)

        # Draw YOLO boxes
        for box in boxes:
            xyxy = box.get('xyxy', [])
            if len(xyxy) == 4:
                x1, y1, x2, y2 = [int(coord * scale) for coord in xyxy]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw UI overlay
        self._draw_ui_overlay(display, item, pending_label)

        return display

    def _draw_ui_overlay(self, image: np.ndarray, item: Dict, pending_label: str):
        """Draw UI text overlay."""
        # Progress
        progress_text = f"Image {self.current_index + 1}/{len(self.review_queue)}"

        # Info
        camera_id = item.get('camera_id', 'unknown')
        yolo_count = item.get('yolo_count', 0)
        confidence = item.get('confidence', 0.0)
        original_label = item.get('binary_label', 'unknown')

        info_lines = [
            progress_text,
            f"Camera: {camera_id}",
            f"YOLO count: {yolo_count} vehicles",
            f"Original label: {original_label} (conf: {confidence:.2f})",
            f"Current label: {pending_label}",
            "",
            "Controls:",
            "[1] traffic_present  [2] traffic_absent  [U] uncertain",
            "[Enter] Accept  [N] Next  [P] Previous  [Q] Quit"
        ]

        y_offset = 30
        for line in info_lines:
            # Draw shadow
            cv2.putText(
                image,
                line,
                (12, y_offset + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
            # Draw text
            cv2.putText(
                image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset += 30

    def _save_reviewed_image(
        self,
        source_path: Path,
        camera_id: str,
        final_label: str,
        original_label: str
    ):
        """Save reviewed image to appropriate directory."""
        # Create output directory
        output_dir = self.labeled_dir / final_label
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        timestamp = source_path.stem
        output_path = output_dir / f"{camera_id}_{timestamp}.jpg"

        # Copy image
        shutil.copy2(source_path, output_path)

        # Update review log
        self.review_log['reviewed'] += 1

        if final_label != 'uncertain':
            if final_label == original_label:
                self.review_log['by_label'][final_label]['confirmed'] += 1
            else:
                self.review_log['corrections'] += 1
                if original_label == 'traffic_present':
                    self.review_log['by_label']['traffic_present']['corrected_to_absent'] += 1
                    self.review_log['correction_patterns']['over_labeled_as_present'] += 1
                else:
                    self.review_log['by_label']['traffic_absent']['corrected_to_present'] += 1
                    self.review_log['correction_patterns']['under_labeled_as_absent'] += 1

            self.review_log['by_label'][final_label]['reviewed'] += 1

    def _calculate_final_statistics(self):
        """Calculate final statistics."""
        total_reviewed = self.review_log['reviewed']
        corrections = self.review_log['corrections']

        if total_reviewed > 0:
            confirmed = total_reviewed - corrections - self.review_log['excluded_uncertain']
            self.review_log['agreement_rate'] = confirmed / total_reviewed

    def validate(self) -> bool:
        """Validate that review completed successfully."""
        if not self.labeled_dir.exists():
            self.logger.error("Labeled directory not found")
            return False

        # Count labeled images
        total_labeled = 0
        for label in ['traffic_present', 'traffic_absent']:
            label_dir = self.labeled_dir / label
            if label_dir.exists():
                count = len(list(label_dir.glob("*.jpg")))
                total_labeled += count
                self.logger.info(f"  {label}: {count} images")

        if total_labeled == 0:
            self.logger.error("No labeled images found")
            return False

        self.logger.info(f"Validation passed: {total_labeled} images labeled")
        return True


def main():
    """CLI entry point for Phase 4."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4: Review and correct binary labels on whole images"
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        default=True,
        help="Review all images (default: True)"
    )
    parser.add_argument(
        "--borderline-only",
        action="store_true",
        help="Review only borderline cases (overrides --review-all)"
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
    phase = ReviewTool(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Determine review mode
        review_all = not args.borderline_only

        # Execute phase
        result = phase.execute(
            validate_after=True,
            review_all=review_all,
            resume=args.resume
        )

        if result['status'] == 'completed':
            print(f"\n✓ Phase 4 completed successfully in {result['duration_seconds']:.1f} seconds")
            log = result['data']
            print(f"  Reviewed: {log.get('reviewed', 0)}")
            print(f"  Corrections: {log.get('corrections', 0)}")
            print(f"  Agreement rate: {log.get('agreement_rate', 0):.1%}")
            print(f"  Excluded uncertain: {log.get('excluded_uncertain', 0)}")
            print(f"\nReviewed images saved to: {config.binary_labeled_dir}/")
            print(f"Next step: Run Phase 4d (phase4d_exclusion_zones) to annotate failure regions")
            exit(0)
        else:
            print(f"\n✗ Phase 4 failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
