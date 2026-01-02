"""
Phase 4: ROI Definition Tool

Interactive GUI for defining polygon regions of interest (ROI) for each camera.
Shows reference images with YOLO bounding boxes and allows manual polygon drawing.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from .base import PipelinePhase
from .config import PipelineConfig
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class ROIDefinitionTool(PipelinePhase):
    """
    Phase 4: Interactive ROI definition tool.

    For each camera, displays the reference image (highest traffic) with YOLO
    bounding boxes overlaid. User draws a polygon around the region where
    YOLO misses vehicles (typically distant areas).

    Controls:
    - Left-click: Add polygon vertex
    - Right-click: Remove last vertex
    - N: Save polygon and move to next camera
    - R: Reset current polygon
    - S: Skip camera (no ROI defined)
    - Q: Save progress and quit
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize ROI definition tool.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="roi_definition",
            description="Defining ROI polygons for cameras"
        )

        self.roi_references_path = self.config.base_dir / "roi_references.json"
        self.output_path = self.config.base_dir / self.config.roi_config_file

        # GUI state
        self.current_polygon: List[Tuple[int, int]] = []
        self.current_camera_id: Optional[str] = None
        self.image_display: Optional[np.ndarray] = None
        self.image_original: Optional[np.ndarray] = None
        self.window_name = "ROI Definition Tool"

        # Colors (BGR format for OpenCV)
        self.color_yolo_box = (0, 255, 0)      # Green for YOLO boxes
        self.color_polygon = (0, 255, 255)     # Yellow for user polygon
        self.color_vertex = (255, 0, 0)        # Blue for vertices
        self.color_text = (255, 255, 255)      # White for text

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute ROI definition tool.

        Args:
            resume: If True, skip cameras that already have ROIs defined

        Returns:
            Dictionary with ROI configuration
        """
        # Load ROI references (from Phase 3)
        self.logger.info(f"Loading ROI references from {self.roi_references_path}")
        if not self.roi_references_path.exists():
            raise FileNotFoundError(
                f"ROI references not found at {self.roi_references_path}. "
                "Run Phase 3 (balanced selection) first."
            )

        roi_references = load_json(self.roi_references_path)

        # Load existing ROI config if resuming
        roi_config = self._load_or_initialize_config(resume)

        # Get list of cameras to process
        cameras_to_process = self._get_cameras_to_process(
            roi_references,
            roi_config,
            resume
        )

        if not cameras_to_process:
            self.logger.info("All cameras already have ROIs defined")
            return roi_config

        self.logger.info(f"Processing {len(cameras_to_process)} cameras")

        # Setup OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Process each camera
        camera_index = 0
        while camera_index < len(cameras_to_process):
            camera_id = cameras_to_process[camera_index]
            reference = roi_references[camera_id]

            self.logger.info(
                f"Camera {camera_index + 1}/{len(cameras_to_process)}: {camera_id}"
            )

            # Process this camera
            action = self._process_camera(
                camera_id,
                reference,
                camera_index,
                len(cameras_to_process),
                roi_config
            )

            if action == "next":
                camera_index += 1
            elif action == "quit":
                self.logger.info("User quit - saving progress")
                break
            elif action == "skip":
                # Add to skipped list
                if camera_id not in roi_config["skipped_cameras"]:
                    roi_config["skipped_cameras"].append(camera_id)
                camera_index += 1

            # Save progress after each camera
            save_json(roi_config, self.output_path)

        # Cleanup
        cv2.destroyAllWindows()

        # Final save
        self.logger.info(f"Saving final ROI configuration to {self.output_path}")
        save_json(roi_config, self.output_path)

        # Log summary
        self._log_summary(roi_config)

        return roi_config

    def validate(self) -> bool:
        """
        Validate that ROI configuration is valid.

        Returns:
            True if validation passed, False otherwise
        """
        if not self.output_path.exists():
            self.logger.error(f"ROI config file not found: {self.output_path}")
            return False

        try:
            roi_config = load_json(self.output_path)
        except Exception as e:
            self.logger.error(f"Failed to load ROI config: {e}")
            return False

        # Validate structure
        required_fields = ["created_at", "cameras", "skipped_cameras"]
        for field in required_fields:
            if field not in roi_config:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate camera entries
        for camera_id, camera_data in roi_config["cameras"].items():
            # Check required fields
            if "polygon" not in camera_data:
                self.logger.error(f"Camera {camera_id} missing polygon")
                return False

            # Validate polygon
            polygon = camera_data["polygon"]
            if not isinstance(polygon, list) or len(polygon) < 3:
                self.logger.error(
                    f"Camera {camera_id} has invalid polygon (need at least 3 vertices)"
                )
                return False

        self.logger.info(
            f"Validation passed: {len(roi_config['cameras'])} cameras with ROIs, "
            f"{len(roi_config['skipped_cameras'])} skipped"
        )
        return True

    def _load_or_initialize_config(self, resume: bool) -> Dict[str, Any]:
        """
        Load existing ROI config or create new one.

        Args:
            resume: If True, load existing config

        Returns:
            ROI configuration dictionary
        """
        if resume and self.output_path.exists():
            self.logger.info("Loading existing ROI configuration")
            return load_json(self.output_path)
        else:
            self.logger.info("Initializing new ROI configuration")
            return {
                "created_at": datetime.now().isoformat(),
                "cameras": {},
                "skipped_cameras": []
            }

    def _get_cameras_to_process(
        self,
        roi_references: Dict[str, Any],
        roi_config: Dict[str, Any],
        resume: bool
    ) -> List[str]:
        """
        Get list of camera IDs to process.

        Args:
            roi_references: ROI reference images from Phase 3
            roi_config: Current ROI configuration
            resume: If True, skip cameras that already have ROIs

        Returns:
            List of camera IDs to process
        """
        all_cameras = sorted(roi_references.keys())

        if not resume:
            return all_cameras

        # Filter out cameras that already have ROIs
        cameras_to_process = [
            cam for cam in all_cameras
            if cam not in roi_config["cameras"]
        ]

        return cameras_to_process

    def _process_camera(
        self,
        camera_id: str,
        reference: Dict[str, Any],
        camera_index: int,
        total_cameras: int,
        roi_config: Dict[str, Any]
    ) -> str:
        """
        Process a single camera interactively.

        Args:
            camera_id: Camera identifier
            reference: Reference image information
            camera_index: Current camera index (0-based)
            total_cameras: Total number of cameras to process
            roi_config: Current ROI configuration

        Returns:
            Action string: "next", "quit", or "skip"
        """
        # Load reference image
        local_path = reference.get("local_path")
        if not local_path:
            self.logger.warning(f"No reference image for camera {camera_id}, skipping")
            return "skip"

        image_path = self.config.base_dir / local_path
        if not image_path.exists():
            self.logger.warning(f"Image not found: {image_path}, skipping")
            return "skip"

        self.image_original = cv2.imread(str(image_path))
        if self.image_original is None:
            self.logger.warning(f"Failed to load image: {image_path}, skipping")
            return "skip"

        # Initialize state for this camera
        self.current_camera_id = camera_id
        self.current_polygon = []

        # Check if camera already has a polygon (allow re-definition)
        if camera_id in roi_config["cameras"]:
            existing_polygon = roi_config["cameras"][camera_id].get("polygon", [])
            self.current_polygon = [tuple(p) for p in existing_polygon]
            self.logger.info(f"Loaded existing polygon with {len(self.current_polygon)} vertices")

        # Get YOLO boxes for visualization
        boxes = reference.get("boxes", [])
        vehicle_count = reference.get("vehicle_count", 0)

        # Interactive loop
        while True:
            # Render display
            self._render_display(
                camera_id,
                camera_index,
                total_cameras,
                vehicle_count,
                boxes
            )

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):  # Next camera
                if len(self.current_polygon) >= 3:
                    # Save polygon
                    roi_config["cameras"][camera_id] = {
                        "reference_image": local_path,
                        "polygon": list(self.current_polygon),
                        "defined_at": datetime.now().isoformat()
                    }
                    self.logger.info(
                        f"Saved ROI with {len(self.current_polygon)} vertices"
                    )
                    return "next"
                else:
                    self.logger.warning("Polygon needs at least 3 vertices (press S to skip)")

            elif key == ord('r'):  # Reset polygon
                self.current_polygon = []
                self.logger.info("Polygon reset")

            elif key == ord('s'):  # Skip camera
                self.logger.info(f"Skipping camera {camera_id}")
                return "skip"

            elif key == ord('q'):  # Quit
                return "quit"

    def _render_display(
        self,
        camera_id: str,
        camera_index: int,
        total_cameras: int,
        vehicle_count: int,
        boxes: List[Dict[str, Any]]
    ):
        """
        Render the display with image, boxes, polygon, and UI.

        Args:
            camera_id: Camera identifier
            camera_index: Current camera index
            total_cameras: Total cameras to process
            vehicle_count: Number of vehicles detected by YOLO
            boxes: YOLO bounding boxes
        """
        # Copy original image
        display = self.image_original.copy()

        # Draw YOLO bounding boxes in green
        for box in boxes:
            xyxy = box.get("xyxy", [])
            if len(xyxy) == 4:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(display, (x1, y1), (x2, y2), self.color_yolo_box, 2)

        # Draw polygon lines in yellow
        if len(self.current_polygon) > 0:
            pts = np.array(self.current_polygon, dtype=np.int32)

            # Draw lines between vertices
            if len(self.current_polygon) > 1:
                cv2.polylines(display, [pts], False, self.color_polygon, 2)

            # Draw closing line if polygon is complete (3+ vertices)
            if len(self.current_polygon) >= 3:
                cv2.polylines(display, [pts], True, self.color_polygon, 2)

            # Draw vertices as circles
            for vertex in self.current_polygon:
                cv2.circle(display, vertex, 5, self.color_vertex, -1)

        # Add header text
        header_y = 30
        line_height = 35

        cv2.putText(
            display,
            f"ROI Definition - {camera_id} (Camera {camera_index + 1}/{total_cameras})",
            (10, header_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color_text,
            2
        )

        cv2.putText(
            display,
            f"YOLO detected: {vehicle_count} vehicles",
            (10, header_y + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.color_text,
            2
        )

        cv2.putText(
            display,
            f"Polygon vertices: {len(self.current_polygon)}",
            (10, header_y + 2 * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.color_text,
            2
        )

        # Add footer with controls
        footer_y = display.shape[0] - 100
        controls = [
            "Left-click: Add vertex | Right-click: Remove vertex",
            "[N] Next camera (save) | [R] Reset polygon",
            "[S] Skip camera | [Q] Quit (progress saved)"
        ]

        for i, control in enumerate(controls):
            cv2.putText(
                display,
                control,
                (10, footer_y + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color_text,
                1
            )

        # Show display
        cv2.imshow(self.window_name, display)
        self.image_display = display

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for polygon drawing.

        Args:
            event: OpenCV mouse event
            x: Mouse x coordinate
            y: Mouse y coordinate
            flags: Event flags
            param: User data (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - add vertex
            self.current_polygon.append((x, y))
            self.logger.debug(f"Added vertex at ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - remove last vertex
            if self.current_polygon:
                removed = self.current_polygon.pop()
                self.logger.debug(f"Removed vertex at {removed}")

    def _log_summary(self, roi_config: Dict[str, Any]):
        """
        Log summary of ROI definition results.

        Args:
            roi_config: Final ROI configuration
        """
        num_defined = len(roi_config["cameras"])
        num_skipped = len(roi_config["skipped_cameras"])

        self.logger.info("=" * 60)
        self.logger.info("ROI DEFINITION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Cameras with ROIs defined: {num_defined}")
        self.logger.info(f"Cameras skipped: {num_skipped}")

        if roi_config["skipped_cameras"]:
            self.logger.info(f"Skipped cameras: {', '.join(roi_config['skipped_cameras'])}")

        self.logger.info("=" * 60)


def main():
    """Command-line entry point for Phase 4."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4: ROI Definition Tool"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cameras that already have ROIs defined"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation without executing phase"
    )
    args = parser.parse_args()

    # Initialize phase
    config = PipelineConfig()
    phase = ROIDefinitionTool(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        if success:
            print("✓ Validation passed")
            exit(0)
        else:
            print("✗ Validation failed")
            exit(1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume)

        if result["status"] == "completed":
            print(f"✓ Phase completed in {result['duration_seconds']:.1f} seconds")

            # Print summary
            data = result["data"]
            num_defined = len(data["cameras"])
            num_skipped = len(data["skipped_cameras"])

            print(f"\nROIs defined for {num_defined} cameras")
            print(f"Skipped {num_skipped} cameras")
            exit(0)
        else:
            print(f"✗ Phase failed: {result.get('reason', 'Unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
