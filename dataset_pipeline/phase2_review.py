"""
Phase 2.5: Label Review & Correction Tool

Interactive GUI for reviewing and correcting YOLO auto-assigned traffic labels
before proceeding to balanced selection.

This phase allows human correction of borderline cases where YOLO's vehicle
count might not accurately reflect true traffic density (e.g., distant queues).
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base import PipelinePhase
from .config import PipelineConfig, TRAFFIC_LEVELS, categorize_traffic
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class LabelReviewTool(PipelinePhase):
    """
    Phase 2.5: Interactive tool for reviewing and correcting YOLO labels.

    Shows full images with YOLO detections and allows human correction
    of auto-assigned traffic levels. Focuses on borderline cases where
    YOLO's vehicle count may be unreliable.

    Controls:
    - 1: Assign to "empty"
    - 2: Assign to "light"
    - 3: Assign to "moderate"
    - 4: Assign to "heavy"
    - Enter: Confirm current label (no change)
    - B: Go back to previous image
    - S: Skip image
    - Q: Save progress and quit
    - R: Reset to auto-assigned label
    """

    # Mapping from number keys to traffic levels
    KEY_TO_LEVEL = {
        ord('1'): 'empty',
        ord('2'): 'light',
        ord('3'): 'moderate',
        ord('4'): 'heavy'
    }

    # Vehicle count ranges that are borderline (for filtering)
    BORDERLINE_RANGES = [
        (2, 3),   # empty/light boundary
        (6, 8),   # light/moderate boundary
        (14, 17)  # moderate/heavy boundary
    ]

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize label review tool.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="review",
            description="Human review and correction of traffic labels"
        )

        self.yolo_results_path = self.config.get_path(self.config.yolo_results_file)
        self.progress_file = self.config.base_dir / "review_progress.json"
        self.backup_file = self.config.base_dir / "yolo_results_backup.json"

        # Display settings
        self.window_name = "Label Review Tool"
        self.display_width = 1200
        self.display_height = 900

        # Colors (BGR format)
        self.color_bg = (40, 40, 40)
        self.color_text = (255, 255, 255)
        self.color_box = (0, 255, 0)  # Green for bounding boxes
        self.color_current = (255, 200, 100)  # Light blue for current label
        self.color_options = (200, 200, 200)  # Gray for other options
        self.color_modified = (0, 165, 255)  # Orange for modified labels

        # State
        self.yolo_results: Dict[str, Any] = None
        self.analyses: List[Dict[str, Any]] = []
        self.current_index: int = 0
        self.modifications: Dict[str, str] = {}  # image_path -> new_label
        self.history: List[int] = []  # For back navigation

    def run(
        self,
        resume: bool = False,
        borderline_only: bool = True,
        specific_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Execute label review tool.

        Args:
            resume: If True, continue from last saved progress
            borderline_only: If True, only show borderline cases
            specific_ranges: Optional custom vehicle count ranges to filter

        Returns:
            Dictionary with review statistics
        """
        self.logger.info("Starting label review tool")

        # Load YOLO results
        if not self.yolo_results_path.exists():
            raise FileNotFoundError(
                f"YOLO results not found: {self.yolo_results_path}. "
                "Run Phase 2 (YOLO analysis) first."
            )

        self.yolo_results = load_json(self.yolo_results_path)
        self.analyses = self.yolo_results.get("analyses", [])

        if not self.analyses:
            raise ValueError("No YOLO analyses found in results file")

        self.logger.info(f"Loaded {len(self.analyses)} YOLO analyses")

        # Filter to borderline cases if requested
        if borderline_only:
            ranges = specific_ranges or self.BORDERLINE_RANGES
            self.analyses = self._filter_borderline(self.analyses, ranges)
            self.logger.info(
                f"Filtered to {len(self.analyses)} borderline cases "
                f"(ranges: {ranges})"
            )

        if not self.analyses:
            self.logger.warning("No images to review after filtering")
            return {"reviewed": 0, "modified": 0}

        # Create backup before starting
        if not self.backup_file.exists():
            save_json(self.yolo_results, self.backup_file)
            self.logger.info(f"Created backup at {self.backup_file}")

        # Load progress if resuming
        if resume and self.progress_file.exists():
            progress = load_json(self.progress_file)
            self.current_index = progress.get("current_index", 0)
            self.modifications = progress.get("modifications", {})
            self.logger.info(f"Resuming from index {self.current_index}")

        # Run interactive review
        stats = self._interactive_review()

        # Apply modifications to YOLO results
        if self.modifications:
            self._apply_modifications()

        return stats

    def _filter_borderline(
        self,
        analyses: List[Dict[str, Any]],
        ranges: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Filter analyses to only borderline vehicle counts.

        Args:
            analyses: List of YOLO analyses
            ranges: List of (min, max) vehicle count ranges

        Returns:
            Filtered list of analyses
        """
        filtered = []
        for analysis in analyses:
            count = analysis.get("vehicle_count", 0)
            for min_count, max_count in ranges:
                if min_count <= count <= max_count:
                    filtered.append(analysis)
                    break
        return filtered

    def _interactive_review(self) -> Dict[str, Any]:
        """
        Run interactive review loop with OpenCV window.

        Returns:
            Statistics about the review session
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        start_time = datetime.now()
        reviewed_count = 0

        while self.current_index < len(self.analyses):
            analysis = self.analyses[self.current_index]

            # Display current image
            display_frame = self._create_display_frame(analysis)
            cv2.imshow(self.window_name, display_frame)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            # Handle input
            if key == ord('q') or key == 27:  # Q or ESC
                self.logger.info("User quit - saving progress")
                self._save_progress()
                break

            elif key in self.KEY_TO_LEVEL:
                # Assign new label
                new_level = f"likely_{self.KEY_TO_LEVEL[key]}"
                old_level = analysis.get("traffic_level")
                if new_level != old_level:
                    image_path = analysis["local_path"]
                    self.modifications[image_path] = new_level
                    self.logger.debug(f"Modified {image_path}: {old_level} -> {new_level}")

                self.history.append(self.current_index)
                self.current_index += 1
                reviewed_count += 1

            elif key == 13:  # Enter - confirm current label
                self.history.append(self.current_index)
                self.current_index += 1
                reviewed_count += 1

            elif key == ord('b') or key == ord('B'):  # Back
                if self.history:
                    self.current_index = self.history.pop()
                    reviewed_count = max(0, reviewed_count - 1)

            elif key == ord('s') or key == ord('S'):  # Skip
                self.history.append(self.current_index)
                self.current_index += 1

            elif key == ord('r') or key == ord('R'):  # Reset
                image_path = analysis["local_path"]
                if image_path in self.modifications:
                    del self.modifications[image_path]
                    self.logger.debug(f"Reset {image_path} to auto-label")

        cv2.destroyAllWindows()

        # Calculate statistics
        duration = (datetime.now() - start_time).total_seconds()
        stats = {
            "reviewed": reviewed_count,
            "modified": len(self.modifications),
            "total_available": len(self.analyses),
            "duration_seconds": duration,
            "avg_seconds_per_image": duration / reviewed_count if reviewed_count > 0 else 0
        }

        self.logger.info(
            f"Review complete: {reviewed_count} reviewed, "
            f"{len(self.modifications)} modified"
        )

        return stats

    def _create_display_frame(self, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Create display frame with image, bounding boxes, and UI.

        Args:
            analysis: YOLO analysis dictionary

        Returns:
            Display frame as numpy array
        """
        # Load image
        image_path = self.config.base_dir / analysis["local_path"]
        if not image_path.exists():
            # Create error frame
            frame = np.full(
                (self.display_height, self.display_width, 3),
                self.color_bg,
                dtype=np.uint8
            )
            cv2.putText(
                frame,
                f"Image not found: {image_path}",
                (50, self.display_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            return frame

        img = cv2.imread(str(image_path))
        if img is None:
            # Create error frame
            frame = np.full(
                (self.display_height, self.display_width, 3),
                self.color_bg,
                dtype=np.uint8
            )
            cv2.putText(
                frame,
                f"Failed to load: {image_path}",
                (50, self.display_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            return frame

        # Draw bounding boxes on image
        for box in analysis.get("boxes", []):
            xyxy = box["xyxy"]
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), self.color_box, 2)

            # Add class label
            label = box.get("class", "vehicle")
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.color_box,
                1
            )

        # Resize image to fit display
        img_height = int(self.display_height * 0.7)  # 70% for image
        img_width = int(img.shape[1] * (img_height / img.shape[0]))
        img_resized = cv2.resize(img, (img_width, img_height))

        # Create full frame with info panel
        frame = np.full(
            (self.display_height, self.display_width, 3),
            self.color_bg,
            dtype=np.uint8
        )

        # Paste image (centered horizontally)
        x_offset = (self.display_width - img_width) // 2
        frame[0:img_height, x_offset:x_offset + img_width] = img_resized

        # Draw info panel
        self._draw_info_panel(frame, analysis, img_height)

        return frame

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        analysis: Dict[str, Any],
        y_start: int
    ):
        """
        Draw information panel below image.

        Args:
            frame: Display frame to draw on
            analysis: YOLO analysis dictionary
            y_start: Y-coordinate to start drawing
        """
        y = y_start + 30
        x_left = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Progress
        progress_text = f"Image {self.current_index + 1} / {len(self.analyses)}"
        cv2.putText(frame, progress_text, (x_left, y), font, font_scale, self.color_text, thickness)
        y += 40

        # Vehicle count
        vehicle_count = analysis.get("vehicle_count", 0)
        count_text = f"Vehicles detected: {vehicle_count}"
        cv2.putText(frame, count_text, (x_left, y), font, font_scale, self.color_text, thickness)
        y += 40

        # Current label
        current_label = analysis.get("traffic_level", "unknown")
        image_path = analysis["local_path"]
        if image_path in self.modifications:
            current_label = self.modifications[image_path]
            label_color = self.color_modified
            label_text = f"Current: {current_label.replace('likely_', '').upper()} (MODIFIED)"
        else:
            label_color = self.color_current
            label_text = f"Current: {current_label.replace('likely_', '').upper()} (auto)"

        cv2.putText(frame, label_text, (x_left, y), font, font_scale, label_color, thickness)
        y += 60

        # Options
        cv2.putText(
            frame,
            "Press number to assign label:",
            (x_left, y),
            font,
            font_scale,
            self.color_options,
            1
        )
        y += 35

        options = [
            "1: Empty",
            "2: Light",
            "3: Moderate",
            "4: Heavy"
        ]
        for option in options:
            cv2.putText(frame, option, (x_left + 30, y), font, 0.6, self.color_options, 1)
            y += 30

        y += 20

        # Controls
        controls = [
            "Enter: Confirm current label",
            "B: Back to previous",
            "S: Skip",
            "R: Reset to auto-label",
            "Q: Save and quit"
        ]
        for control in controls:
            cv2.putText(frame, control, (x_left, y), font, 0.5, self.color_options, 1)
            y += 25

        # Camera ID
        camera = analysis.get("camera_id", "unknown")
        cv2.putText(
            frame,
            f"Camera: {camera}",
            (self.display_width - 300, y_start + 30),
            font,
            0.6,
            self.color_text,
            1
        )

    def _save_progress(self):
        """Save current progress to resume later."""
        progress = {
            "current_index": self.current_index,
            "modifications": self.modifications,
            "saved_at": datetime.now().isoformat()
        }
        save_json(progress, self.progress_file)
        self.logger.info(f"Progress saved to {self.progress_file}")

    def _apply_modifications(self):
        """Apply modifications to YOLO results and save."""
        self.logger.info(f"Applying {len(self.modifications)} modifications")

        modified_count = 0
        for analysis in self.yolo_results["analyses"]:
            image_path = analysis["local_path"]
            if image_path in self.modifications:
                old_label = analysis["traffic_level"]
                new_label = self.modifications[image_path]
                analysis["traffic_level"] = new_label
                analysis["human_reviewed"] = True
                analysis["original_auto_label"] = old_label
                modified_count += 1

        # Save updated results
        save_json(self.yolo_results, self.yolo_results_path)
        self.logger.info(
            f"Updated {modified_count} labels in {self.yolo_results_path}"
        )

        # Clear progress file
        if self.progress_file.exists():
            self.progress_file.unlink()
            self.logger.info("Cleared progress file")


def main():
    """
    CLI entry point for label review tool.

    Usage:
        python -m dataset_pipeline.phase2_review [options]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Review and correct YOLO auto-assigned traffic labels"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved progress"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review all images (not just borderline cases)"
    )
    parser.add_argument(
        "--ranges",
        type=str,
        help="Custom vehicle count ranges to review (e.g., '2-3,6-8,14-17')"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    pipeline_config = PipelineConfig()

    # Parse custom ranges if provided
    specific_ranges = None
    if args.ranges:
        try:
            specific_ranges = []
            for range_str in args.ranges.split(','):
                min_val, max_val = map(int, range_str.split('-'))
                specific_ranges.append((min_val, max_val))
        except ValueError:
            logger.error(f"Invalid ranges format: {args.ranges}")
            logger.error("Expected format: '2-3,6-8,14-17'")
            exit(1)

    # Create review tool
    review_tool = LabelReviewTool(pipeline_config)

    try:
        # Run review
        stats = review_tool.run(
            resume=args.resume,
            borderline_only=not args.all,
            specific_ranges=specific_ranges
        )

        # Print summary
        print("\n" + "=" * 80)
        print("LABEL REVIEW COMPLETE")
        print("=" * 80)
        print(f"\nImages reviewed: {stats['reviewed']}")
        print(f"Labels modified: {stats['modified']}")
        print(f"Total available: {stats['total_available']}")
        print(f"Duration: {stats['duration_seconds']:.1f} seconds")
        if stats['reviewed'] > 0:
            print(f"Average: {stats['avg_seconds_per_image']:.1f} seconds per image")
        print("\n" + "=" * 80)

        exit(0)

    except Exception as e:
        logger.error(f"Review failed: {e}", exc_info=True)
        print(f"\n✗ Review failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
