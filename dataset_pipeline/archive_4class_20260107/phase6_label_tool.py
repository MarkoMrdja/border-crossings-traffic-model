"""
Phase 6: Labeling & Verification Tool

Interactive GUI for human verification of auto-assigned traffic labels.
Shows cropped images with predicted labels and allows confirmation or correction.
"""

import logging
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base import PipelinePhase
from .config import PipelineConfig, TRAFFIC_LEVELS
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class LabelingTool(PipelinePhase):
    """
    Phase 6: Interactive labeling tool for verifying YOLO-assigned labels.

    Shows 64x64 cropped images from likely_{level} directories and allows
    users to confirm or correct the predicted labels.

    Controls:
    - Enter: Confirm predicted label
    - 1: Assign to "empty"
    - 2: Assign to "light"
    - 3: Assign to "moderate"
    - 4: Assign to "heavy"
    - S: Skip image
    - Q: Save progress and quit
    - B: Go back to previous image
    """

    # Mapping from number keys to traffic levels
    KEY_TO_LEVEL = {
        ord('1'): 'empty',
        ord('2'): 'light',
        ord('3'): 'moderate',
        ord('4'): 'heavy'
    }

    # Reverse mapping for display
    LEVEL_TO_NUMBER = {
        'empty': '1',
        'light': '2',
        'moderate': '3',
        'heavy': '4'
    }

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize labeling tool.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="labeling",
            description="Human verification of traffic labels"
        )

        self.crops_dir = self.config.base_dir / "crops"
        self.labeled_dir = self.config.base_dir / "labeled"
        self.progress_file = self.config.base_dir / "labeling_progress.json"

        # Create labeled directories
        for level in TRAFFIC_LEVELS:
            (self.labeled_dir / level).mkdir(parents=True, exist_ok=True)

        # Display settings
        self.window_name = "Traffic Labeling Tool"
        self.display_size = 512  # Scale up from 64x64 for visibility

        # Colors (BGR format)
        self.color_bg = (40, 40, 40)
        self.color_text = (255, 255, 255)
        self.color_predicted = (100, 200, 255)
        self.color_selected = (100, 255, 100)

        # State
        self.current_images: List[Tuple[Path, str]] = []  # (path, predicted_level)
        self.current_index: int = 0
        self.history: List[Tuple[Path, str, str]] = []  # (path, predicted, confirmed) for undo

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute labeling tool.

        Args:
            resume: If True, skip already labeled images

        Returns:
            Dictionary with labeling statistics
        """
        # Load or initialize progress
        progress = self._load_or_initialize_progress(resume)

        # Collect all images to label
        self.logger.info("Collecting images from crops directory...")
        images_by_level = self._collect_images(resume, progress)

        if not any(images_by_level.values()):
            self.logger.info("All images already labeled!")
            return progress

        # Count total
        total_images = sum(len(imgs) for imgs in images_by_level.values())
        self.logger.info(f"Found {total_images} images to label")

        # Setup OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 700)

        # Process images in priority order (heavy → moderate → light → empty)
        # This ensures critical mislabels (missed heavy traffic) are caught first
        for predicted_level in ['likely_heavy', 'likely_moderate', 'likely_light', 'likely_empty']:
            images = images_by_level.get(predicted_level, [])

            if not images:
                continue

            self.logger.info(f"Processing {len(images)} images from '{predicted_level}'")

            # Set current batch
            self.current_images = [(Path(img), predicted_level) for img in images]
            self.current_index = 0

            # Process batch
            quit_requested = self._process_batch(predicted_level, progress)

            if quit_requested:
                self.logger.info("User quit - saving progress")
                break

        # Cleanup
        cv2.destroyAllWindows()

        # Final save
        save_json(progress, self.progress_file)

        # Log summary
        self._log_summary(progress)

        return progress

    def validate(self) -> bool:
        """
        Validate that labeling completed successfully.

        Returns:
            True if labeled directory has images
        """
        labeled_count = 0
        for level in TRAFFIC_LEVELS:
            level_dir = self.labeled_dir / level
            if level_dir.exists():
                labeled_count += len(list(level_dir.glob("*.jpg")))

        self.logger.info(f"Validation: Found {labeled_count} labeled images")
        return labeled_count > 0

    def _load_or_initialize_progress(self, resume: bool) -> Dict[str, Any]:
        """Load existing progress or initialize new."""
        if resume and self.progress_file.exists():
            self.logger.info("Resuming from existing progress")
            progress = load_json(self.progress_file)
        else:
            self.logger.info("Initializing new labeling session")
            progress = {
                "started_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "total": 0,
                "confirmed": 0,
                "corrected": 0,
                "skipped": 0,
                "remaining": 0,
                "corrections": {},  # Track correction patterns
                "labeled_files": []  # Track which files have been labeled
            }
            save_json(progress, self.progress_file)

        return progress

    def _collect_images(
        self,
        resume: bool,
        progress: Dict[str, Any]
    ) -> Dict[str, List[Path]]:
        """
        Collect all images that need labeling.

        Args:
            resume: If True, skip already labeled images
            progress: Progress dictionary

        Returns:
            Dictionary mapping predicted level to list of image paths
        """
        images_by_level = {}
        labeled_files = set(progress.get("labeled_files", []))

        for predicted_level in ['likely_empty', 'likely_light', 'likely_moderate', 'likely_heavy']:
            level_dir = self.crops_dir / predicted_level

            if not level_dir.exists():
                self.logger.warning(f"Directory not found: {level_dir}")
                images_by_level[predicted_level] = []
                continue

            # Get all jpg images
            all_images = sorted(level_dir.glob("*.jpg"))

            # Filter out already labeled if resuming
            if resume:
                images = [
                    img for img in all_images
                    if str(img.relative_to(self.config.base_dir)) not in labeled_files
                ]
            else:
                images = all_images

            images_by_level[predicted_level] = images

            if resume:
                self.logger.info(
                    f"{predicted_level}: {len(images)} to label "
                    f"({len(all_images) - len(images)} already done)"
                )

        return images_by_level

    def _process_batch(
        self,
        predicted_level: str,
        progress: Dict[str, Any]
    ) -> bool:
        """
        Process a batch of images with the same predicted level.

        Args:
            predicted_level: The predicted traffic level
            progress: Progress dictionary

        Returns:
            True if user quit, False otherwise
        """
        while self.current_index < len(self.current_images):
            image_path, pred_level = self.current_images[self.current_index]

            # Display image and get user input
            action = self._display_and_get_input(
                image_path,
                pred_level,
                self.current_index,
                len(self.current_images),
                progress
            )

            if action == "quit":
                return True
            elif action == "skip":
                progress["skipped"] += 1
                progress["labeled_files"].append(str(image_path.relative_to(self.config.base_dir)))
                self.current_index += 1
            elif action == "back":
                if self.history:
                    # Undo last action
                    last_path, last_pred, last_confirmed = self.history.pop()

                    # Move file back from labeled to crops
                    confirmed_level = last_confirmed.replace('likely_', '')
                    source = self.labeled_dir / confirmed_level / last_path.name
                    dest = self.crops_dir / last_pred / last_path.name

                    if source.exists():
                        shutil.move(str(source), str(dest))

                        # Update progress
                        progress["labeled_files"].remove(str(last_path.relative_to(self.config.base_dir)))
                        if last_pred == f"likely_{last_confirmed}":
                            progress["confirmed"] -= 1
                        else:
                            progress["corrected"] -= 1
                            correction_key = f"{last_pred}_to_{last_confirmed}"
                            if correction_key in progress["corrections"]:
                                progress["corrections"][correction_key] -= 1

                        # Go back in both batches
                        if self.current_index > 0:
                            self.current_index -= 1

                        self.logger.info(f"Undid labeling of {last_path.name}")
                else:
                    self.logger.warning("No history to undo")
            elif action in TRAFFIC_LEVELS:
                # User assigned a label
                self._apply_label(image_path, pred_level, action, progress)
                self.current_index += 1

            # Save progress periodically
            if self.current_index % 10 == 0:
                progress["updated_at"] = datetime.now().isoformat()
                save_json(progress, self.progress_file)

        return False

    def _display_and_get_input(
        self,
        image_path: Path,
        predicted_level: str,
        current_idx: int,
        total_in_batch: int,
        progress: Dict[str, Any]
    ) -> str:
        """
        Display image and wait for user input.

        Args:
            image_path: Path to image to display
            predicted_level: Predicted traffic level (with 'likely_' prefix)
            current_idx: Current index in batch
            total_in_batch: Total images in batch
            progress: Progress dictionary

        Returns:
            Action string: 'quit', 'skip', 'back', or a traffic level
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.warning(f"Failed to load image: {image_path}")
            return "skip"

        # Create display
        display = self._create_display(
            img,
            predicted_level,
            current_idx,
            total_in_batch,
            progress
        )

        # Show
        cv2.imshow(self.window_name, display)

        # Wait for key
        while True:
            key = cv2.waitKey(0) & 0xFF

            # Enter = confirm predicted label
            if key == 13:  # Enter
                # Remove 'likely_' prefix
                return predicted_level.replace('likely_', '')

            # Number keys 1-4 = assign specific label
            elif key in self.KEY_TO_LEVEL:
                return self.KEY_TO_LEVEL[key]

            # S = skip
            elif key == ord('s') or key == ord('S'):
                return "skip"

            # Q = quit
            elif key == ord('q') or key == ord('Q'):
                return "quit"

            # B = back
            elif key == ord('b') or key == ord('B'):
                return "back"

            # ESC = quit
            elif key == 27:  # ESC
                return "quit"

    def _create_display(
        self,
        img: np.ndarray,
        predicted_level: str,
        current_idx: int,
        total_in_batch: int,
        progress: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create display image with controls and info.

        Args:
            img: Original 64x64 image
            predicted_level: Predicted traffic level
            current_idx: Current index
            total_in_batch: Total in batch
            progress: Progress dictionary

        Returns:
            Display image
        """
        # Create canvas (wider to accommodate instructions)
        canvas_height = 700
        canvas_width = 1000
        canvas = np.full((canvas_height, canvas_width, 3), self.color_bg, dtype=np.uint8)

        # Scale up image (64x64 → 512x512) for visibility
        img_scaled = cv2.resize(img, (self.display_size, self.display_size), interpolation=cv2.INTER_NEAREST)

        # Place image in center-left
        y_offset = (canvas_height - self.display_size) // 2
        x_offset = 50
        canvas[y_offset:y_offset+self.display_size, x_offset:x_offset+self.display_size] = img_scaled

        # Draw border around image
        cv2.rectangle(
            canvas,
            (x_offset-2, y_offset-2),
            (x_offset+self.display_size+2, y_offset+self.display_size+2),
            self.color_text,
            2
        )

        # Right side: Info and controls
        right_x = x_offset + self.display_size + 50
        y = 50

        # Title
        self._draw_text(canvas, "Traffic Labeling Tool", (right_x, y), scale=0.8, thickness=2)
        y += 50

        # Progress in current batch
        batch_name = predicted_level.replace('likely_', '').upper()
        progress_text = f"Reviewing: {batch_name}"
        self._draw_text(canvas, progress_text, (right_x, y), color=self.color_predicted)
        y += 30

        # Current position
        pos_text = f"Image: {current_idx + 1} / {total_in_batch}"
        self._draw_text(canvas, pos_text, (right_x, y))
        y += 50

        # Overall progress
        total_processed = progress["confirmed"] + progress["corrected"] + progress["skipped"]
        overall_text = f"Total processed: {total_processed}"
        self._draw_text(canvas, overall_text, (right_x, y))
        y += 30

        stats_text = f"  Confirmed: {progress['confirmed']}"
        self._draw_text(canvas, stats_text, (right_x, y), scale=0.5)
        y += 25

        stats_text = f"  Corrected: {progress['corrected']}"
        self._draw_text(canvas, stats_text, (right_x, y), scale=0.5)
        y += 25

        stats_text = f"  Skipped: {progress['skipped']}"
        self._draw_text(canvas, stats_text, (right_x, y), scale=0.5)
        y += 50

        # Separator
        cv2.line(canvas, (right_x, y), (right_x + 350, y), self.color_text, 1)
        y += 40

        # Predicted label
        pred_clean = predicted_level.replace('likely_', '').upper()
        pred_num = self.LEVEL_TO_NUMBER[predicted_level.replace('likely_', '')]
        self._draw_text(
            canvas,
            f"Predicted: [{pred_num}] {pred_clean}",
            (right_x, y),
            color=self.color_predicted,
            scale=0.7,
            thickness=2
        )
        y += 60

        # Separator
        cv2.line(canvas, (right_x, y), (right_x + 350, y), self.color_text, 1)
        y += 40

        # Controls
        self._draw_text(canvas, "Controls:", (right_x, y), scale=0.6, thickness=2)
        y += 35

        controls = [
            ("[Enter]", f"Confirm as {pred_clean}"),
            ("", ""),
            ("[1]", "Label as EMPTY"),
            ("[2]", "Label as LIGHT"),
            ("[3]", "Label as MODERATE"),
            ("[4]", "Label as HEAVY"),
            ("", ""),
            ("[S]", "Skip this image"),
            ("[B]", "Go back (undo)"),
            ("[Q]", "Save & Quit"),
        ]

        for key, desc in controls:
            if not key:
                y += 15
                continue

            # Draw key
            self._draw_text(canvas, key, (right_x, y), scale=0.5, color=self.color_selected)
            # Draw description
            self._draw_text(canvas, desc, (right_x + 100, y), scale=0.5)
            y += 25

        return canvas

    def _draw_text(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        scale: float = 0.6,
        color: Tuple[int, int, int] = None,
        thickness: int = 1
    ):
        """Draw text on image."""
        if color is None:
            color = self.color_text

        cv2.putText(
            img,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def _apply_label(
        self,
        image_path: Path,
        predicted_level: str,
        confirmed_level: str,
        progress: Dict[str, Any]
    ):
        """
        Apply label to image (move to labeled directory).

        Args:
            image_path: Source image path
            predicted_level: Predicted level (with 'likely_' prefix)
            confirmed_level: User-confirmed level (without prefix)
            progress: Progress dictionary
        """
        # Destination
        dest_dir = self.labeled_dir / confirmed_level
        dest_path = dest_dir / image_path.name

        # Move file
        try:
            shutil.move(str(image_path), str(dest_path))
        except Exception as e:
            self.logger.error(f"Failed to move {image_path} to {dest_path}: {e}")
            return

        # Update progress
        progress["labeled_files"].append(str(image_path.relative_to(self.config.base_dir)))

        # Check if this was a confirmation or correction
        predicted_clean = predicted_level.replace('likely_', '')
        if predicted_clean == confirmed_level:
            progress["confirmed"] += 1
        else:
            progress["corrected"] += 1

            # Track correction pattern
            correction_key = f"{predicted_level}_to_{confirmed_level}"
            if correction_key not in progress["corrections"]:
                progress["corrections"][correction_key] = 0
            progress["corrections"][correction_key] += 1

        # Add to history for undo
        self.history.append((image_path, predicted_level, confirmed_level))

        self.logger.info(
            f"Labeled {image_path.name}: {predicted_level} → {confirmed_level}"
        )

    def _log_summary(self, progress: Dict[str, Any]):
        """Log summary of labeling session."""
        self.logger.info("=" * 60)
        self.logger.info("Labeling Summary")
        self.logger.info("=" * 60)

        total = progress["confirmed"] + progress["corrected"] + progress["skipped"]
        self.logger.info(f"Total processed: {total}")
        self.logger.info(f"  Confirmed: {progress['confirmed']}")
        self.logger.info(f"  Corrected: {progress['corrected']}")
        self.logger.info(f"  Skipped: {progress['skipped']}")

        if progress["corrections"]:
            self.logger.info("\nCorrection patterns:")
            for pattern, count in sorted(
                progress["corrections"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                self.logger.info(f"  {pattern}: {count}")

        # Count final labels
        self.logger.info("\nFinal label distribution:")
        for level in TRAFFIC_LEVELS:
            level_dir = self.labeled_dir / level
            if level_dir.exists():
                count = len(list(level_dir.glob("*.jpg")))
                self.logger.info(f"  {level}: {count}")

        self.logger.info("=" * 60)


def main():
    """CLI entry point for Phase 6."""
    import argparse
    from .config import PipelineConfig

    parser = argparse.ArgumentParser(
        description="Phase 6: Interactive Labeling Tool"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous session (skip already labeled images)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing labels without running tool"
    )

    args = parser.parse_args()

    # Initialize
    config = PipelineConfig()
    tool = LabelingTool(config)

    if args.validate_only:
        print("Validating labeled images...")
        is_valid = tool.validate()
        if is_valid:
            print("✓ Validation passed")
        else:
            print("✗ Validation failed")
        return

    # Run
    print("\n" + "=" * 60)
    print("Phase 6: Traffic Labeling Tool")
    print("=" * 60)
    print("\nThis tool will display cropped images for human verification.")
    print("You can confirm or correct the predicted traffic labels.\n")

    if args.resume:
        print("Resuming from previous session...\n")

    result = tool.execute(resume=args.resume)

    if result["status"] == "completed":
        print("\n✓ Phase 6 completed successfully")
        print(f"Duration: {result['duration_seconds']:.1f} seconds")
    else:
        print(f"\n✗ Phase 6 failed: {result.get('reason', 'Unknown error')}")


if __name__ == "__main__":
    main()
