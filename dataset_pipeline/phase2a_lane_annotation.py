"""
Phase 2a: Interactive Lane Annotation Tool

Semi-automated lane polygon annotation with auto-suggestions and review interface.
Creates lane_polygons.json which Phase 2 uses for automatic polygon filtering.

Run this phase once to create lane polygons. Skip if lane_polygons.json already exists.

Controls:
- [A] Accept auto-suggestion
- [M] Modify suggestion (enter edit mode)
- [D] Reject and draw manual
- [E] Toggle edge detection overlay
- [L] Toggle line detection overlay
- [V] Toggle YOLO coverage visualization
- [N] Next camera (save)
- [R] Reset polygon
- [S] Skip camera
- [Q] Quit and save progress
- Left-click: Add vertex
- Right-click: Remove last vertex
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from .base import PipelinePhase
from .config import PipelineConfig, LANE_DETECTION_CONFIG
from .utils import load_json, save_json
from .phase4a_lane_detection import detect_lane_polygon

logger = logging.getLogger(__name__)


class LaneAnnotationTool(PipelinePhase):
    """
    Phase 2a: Interactive lane annotation tool with auto-detection.

    For each camera:
    1. Run auto-detection algorithm
    2. Display suggestion with confidence
    3. User reviews: Accept / Modify / Reject
    4. Save to lane_polygons.json

    The lane_polygons.json file is used by Phase 2 (YOLO) for automatic
    polygon filtering to exclude parked cars outside border crossing lanes.
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize lane annotation tool.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="lane_annotation",
            description="Annotating lane polygons with auto-suggestions"
        )

        self.roi_references_path = self.config.base_dir / "roi_references.json"
        self.yolo_results_path = self.config.base_dir / self.config.yolo_results_file
        self.mini_dataset_path = self.config.base_dir / "mini_test_dataset"
        self.mini_manifest_path = self.mini_dataset_path / "mini_dataset_manifest.json"
        self.output_path = self.config.base_dir / "lane_polygons.json"

        # GUI state
        self.current_polygon: List[Tuple[int, int]] = []
        self.auto_suggestion: Optional[np.ndarray] = None
        self.suggestion_confidence: float = 0.0
        self.current_camera_id: Optional[str] = None
        self.current_lane_count: int = 2  # Default number of lanes
        self.image_display: Optional[np.ndarray] = None
        self.image_original: Optional[np.ndarray] = None
        self.window_name = "Lane Annotation Tool"

        # Mode tracking
        self.edit_mode: str = "auto"  # "auto", "edit", "manual"
        self.user_modified: bool = False

        # Visualization overlays (toggleable)
        self.show_edges: bool = False
        self.show_lines: bool = False
        self.show_coverage: bool = False
        self.show_empty_reference: bool = False

        # Edge/line detection caches
        self.edges_overlay: Optional[np.ndarray] = None
        self.lines_overlay: Optional[np.ndarray] = None

        # Empty reference image (for seeing clear lane boundaries)
        self.empty_reference: Optional[np.ndarray] = None

        # Current camera data
        self.current_yolo_boxes: List[Dict[str, Any]] = []

        # Colors (BGR format for OpenCV)
        self.color_yolo_box = (0, 255, 0)          # Green for YOLO boxes
        self.color_auto_polygon = (255, 255, 0)    # Cyan for auto-suggestion
        self.color_user_polygon = (0, 255, 255)    # Yellow for user edits
        self.color_vertex = (255, 0, 0)            # Blue for vertices
        self.color_text = (255, 255, 255)          # White for text
        self.color_edges = (0, 0, 255)             # Red for edges
        self.color_lines = (255, 128, 0)           # Light blue for lines

        # Statistics
        self.stats = {
            "total_cameras": 0,
            "auto_accepted": 0,
            "user_modified": 0,
            "manual_drawn": 0,
            "skipped": 0
        }

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute lane annotation tool.

        Args:
            resume: If True, skip cameras that already have lane polygons

        Returns:
            Dictionary with lane polygon configuration
        """
        # Load ROI references
        self.logger.info(f"Loading ROI references from {self.roi_references_path}")
        if not self.roi_references_path.exists():
            raise FileNotFoundError(
                f"ROI references not found at {self.roi_references_path}. "
                "Run Phase 3 (balanced selection) first."
            )

        roi_references = load_json(self.roi_references_path)

        # Load YOLO results
        self.logger.info(f"Loading YOLO results from {self.yolo_results_path}")
        yolo_data = load_json(self.yolo_results_path)

        # Build YOLO lookup by local_path
        yolo_lookup = {a['local_path']: a for a in yolo_data.get('analyses', [])}

        # Load mini dataset manifest for empty reference images
        mini_manifest = None
        if self.mini_manifest_path.exists():
            self.logger.info(f"Loading mini dataset manifest for empty references")
            mini_manifest = load_json(self.mini_manifest_path)
        else:
            self.logger.warning("Mini dataset not found - empty reference toggle will not be available")

        # Load existing config if resuming
        lane_config = self._load_or_initialize_config(resume)

        # Get cameras to process
        cameras_to_process = self._get_cameras_to_process(
            roi_references,
            lane_config,
            resume
        )

        if not cameras_to_process:
            self.logger.info("All cameras already have lane polygons defined")
            return lane_config

        self.logger.info(f"Processing {len(cameras_to_process)} cameras")
        self.stats["total_cameras"] = len(cameras_to_process)

        # Setup OpenCV window - larger for side-by-side layout
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 2000, 1200)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Process each camera
        camera_index = 0
        while camera_index < len(cameras_to_process):
            camera_id = cameras_to_process[camera_index]
            reference = roi_references[camera_id]

            self.logger.info(
                f"Camera {camera_index + 1}/{len(cameras_to_process)}: {camera_id}"
            )

            # Get YOLO boxes for this camera's reference image
            ref_local_path = reference.get('local_path', '')
            yolo_data_for_ref = yolo_lookup.get(ref_local_path, {})
            self.current_yolo_boxes = yolo_data_for_ref.get('boxes', [])

            # Get empty reference image for this camera (from mini dataset)
            empty_ref_path = None
            if mini_manifest and camera_id in mini_manifest.get('cameras', {}):
                empty_rel_path = mini_manifest['cameras'][camera_id]['emptiest'].get('local_path')
                if empty_rel_path:
                    empty_ref_path = self.config.base_dir / empty_rel_path

            # Process this camera
            action = self._process_camera(
                camera_id,
                reference,
                camera_index,
                len(cameras_to_process),
                lane_config,
                empty_ref_path
            )

            if action == "next":
                camera_index += 1
            elif action == "quit":
                self.logger.info("User quit - saving progress")
                break
            elif action == "skip":
                if camera_id not in lane_config["skipped_cameras"]:
                    lane_config["skipped_cameras"].append(camera_id)
                self.stats["skipped"] += 1
                camera_index += 1

            # Save progress after each camera
            save_json(lane_config, self.output_path)

        # Cleanup
        cv2.destroyAllWindows()

        # Final save with statistics
        lane_config["statistics"] = self.stats
        self.logger.info(f"Saving final lane polygon configuration to {self.output_path}")
        save_json(lane_config, self.output_path)

        # Log summary
        self._log_summary(lane_config)

        return lane_config

    def _load_or_initialize_config(self, resume: bool) -> Dict[str, Any]:
        """Load existing config or create new one."""
        if resume and self.output_path.exists():
            self.logger.info("Loading existing lane polygon configuration")
            config = load_json(self.output_path)
            # Load statistics if available
            self.stats = config.get("statistics", self.stats)
            return config
        else:
            self.logger.info("Initializing new lane polygon configuration")
            return {
                "created_at": datetime.now().isoformat(),
                "mode": LANE_DETECTION_CONFIG.get("mode", "single"),
                "cameras": {},
                "skipped_cameras": [],
                "statistics": {}
            }

    def _get_cameras_to_process(
        self,
        roi_references: Dict[str, Any],
        lane_config: Dict[str, Any],
        resume: bool
    ) -> List[str]:
        """Get list of camera IDs to process."""
        all_cameras = sorted(roi_references.keys())

        if not resume:
            return all_cameras

        # Filter out cameras that already have lane polygons
        cameras_to_process = [
            cam for cam in all_cameras
            if cam not in lane_config["cameras"]
        ]

        return cameras_to_process

    def _process_camera(
        self,
        camera_id: str,
        reference: Dict[str, Any],
        camera_index: int,
        total_cameras: int,
        lane_config: Dict[str, Any],
        empty_ref_path: Optional[Path] = None
    ) -> str:
        """
        Process a single camera interactively.

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

        # Load empty reference image if available
        self.empty_reference = None
        if empty_ref_path and empty_ref_path.exists():
            empty_img = cv2.imread(str(empty_ref_path))
            if empty_img is not None:
                # Resize to match original if needed
                if empty_img.shape != self.image_original.shape:
                    empty_img = cv2.resize(empty_img,
                                          (self.image_original.shape[1], self.image_original.shape[0]))
                self.empty_reference = empty_img
                self.logger.info("Loaded empty reference image for toggle")
            else:
                self.logger.warning(f"Failed to load empty reference: {empty_ref_path}")
        else:
            self.logger.info("No empty reference available for this camera")

        # Initialize state for this camera
        self.current_camera_id = camera_id
        self.current_polygon = []
        self.auto_suggestion = None
        self.suggestion_confidence = 0.0
        self.edit_mode = "auto"
        self.user_modified = False
        self.show_edges = False
        self.show_lines = False
        self.show_coverage = False
        self.show_empty_reference = False
        self.current_lane_count = 2  # Default

        # Check if camera already has polygons (allow re-definition)
        if camera_id in lane_config["cameras"]:
            existing_polygons = lane_config["cameras"][camera_id].get("polygons", [])
            if existing_polygons:
                existing_poly = existing_polygons[0]['polygon']
                self.current_polygon = [tuple(p) for p in existing_poly]
                self.edit_mode = "edit"
                self.logger.info(f"Loaded existing polygon with {len(self.current_polygon)} vertices")

            # Load existing lane count
            self.current_lane_count = lane_config["cameras"][camera_id].get("lane_count", 2)

        # Run auto-detection
        if self.edit_mode == "auto":
            self.logger.info("Running auto-detection...")
            suggestions = detect_lane_polygon(
                str(image_path),
                self.current_yolo_boxes,
                mode='single',
                config=LANE_DETECTION_CONFIG
            )

            if suggestions:
                suggestion = suggestions[0]
                self.auto_suggestion = np.array(suggestion['polygon'], dtype=np.int32)
                self.suggestion_confidence = suggestion['confidence']
                self.logger.info(
                    f"Auto-detected polygon: {len(self.auto_suggestion)} vertices, "
                    f"confidence: {self.suggestion_confidence:.3f}"
                )
            else:
                self.logger.warning("Auto-detection failed, falling back to manual mode")
                self.edit_mode = "manual"

        # Pre-compute visualization overlays
        self._compute_visualization_overlays(image_path)

        # Interactive loop
        while True:
            # Render display
            self._render_display(
                camera_id,
                camera_index,
                total_cameras,
                reference.get("vehicle_count", 0)
            )

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):  # Accept suggestion
                if self.edit_mode == "auto" and self.auto_suggestion is not None:
                    self.current_polygon = [(int(x), int(y)) for x, y in self.auto_suggestion]
                    return self._save_polygon(lane_config, camera_id, local_path, user_modified=False)

            elif key == ord('m'):  # Modify suggestion
                if self.edit_mode == "auto" and self.auto_suggestion is not None:
                    # Copy suggestion to editable polygon
                    self.current_polygon = [(int(x), int(y)) for x, y in self.auto_suggestion]
                    self.edit_mode = "edit"
                    self.logger.info("Entering edit mode")

            elif key == ord('d'):  # Draw manual
                self.edit_mode = "manual"
                self.current_polygon = []
                self.auto_suggestion = None
                self.logger.info("Entering manual mode")

            elif key == ord('e'):  # Toggle edges
                self.show_edges = not self.show_edges
                self.logger.debug(f"Edge overlay: {self.show_edges}")

            elif key == ord('l'):  # Toggle lines
                self.show_lines = not self.show_lines
                self.logger.debug(f"Line overlay: {self.show_lines}")

            elif key == ord('v'):  # Toggle YOLO coverage
                self.show_coverage = not self.show_coverage
                self.logger.debug(f"YOLO coverage: {self.show_coverage}")

            elif key == ord('x'):  # Toggle empty reference
                if self.empty_reference is not None:
                    self.show_empty_reference = not self.show_empty_reference
                    self.logger.debug(f"Empty reference: {self.show_empty_reference}")
                else:
                    self.logger.info("No empty reference available for this camera")

            elif key == ord('n'):  # Next (save)
                if len(self.current_polygon) >= 3:
                    user_modified = self.edit_mode in ["edit", "manual"] or self.user_modified
                    return self._save_polygon(lane_config, camera_id, local_path, user_modified)
                else:
                    self.logger.warning("Polygon needs at least 3 vertices (press S to skip)")

            elif key == ord('r'):  # Reset
                if self.edit_mode == "auto" and self.auto_suggestion is not None:
                    # Reset to auto-suggestion
                    self.current_polygon = []
                    self.user_modified = False
                else:
                    # Clear polygon
                    self.current_polygon = []
                self.logger.info("Polygon reset")

            elif key == ord('s'):  # Skip
                self.logger.info(f"Skipping camera {camera_id}")
                return "skip"

            elif key == ord('q'):  # Quit
                return "quit"

            elif key == ord('+') or key == ord('='):  # Increase lane count
                self.current_lane_count = min(10, self.current_lane_count + 1)
                self.logger.info(f"Lane count: {self.current_lane_count}")

            elif key == ord('-') or key == ord('_'):  # Decrease lane count
                self.current_lane_count = max(1, self.current_lane_count - 1)
                self.logger.info(f"Lane count: {self.current_lane_count}")

    def _save_polygon(
        self,
        lane_config: Dict[str, Any],
        camera_id: str,
        local_path: str,
        user_modified: bool
    ) -> str:
        """Save polygon and update statistics."""
        lane_config["cameras"][camera_id] = {
            "reference_image": local_path,
            "lane_count": self.current_lane_count,  # NEW: Save number of lanes
            "polygons": [{
                "id": 0,
                "name": "all_lanes",
                "polygon": list(self.current_polygon),
                "auto_detected": self.edit_mode == "auto",
                "user_modified": user_modified,
                "confidence": self.suggestion_confidence if self.edit_mode == "auto" else 0.0,
                "defined_at": datetime.now().isoformat()
            }]
        }

        # Update statistics
        if self.edit_mode == "auto" and not user_modified:
            self.stats["auto_accepted"] += 1
        elif user_modified:
            self.stats["user_modified"] += 1
        else:
            self.stats["manual_drawn"] += 1

        self.logger.info(
            f"Saved lane polygon: {len(self.current_polygon)} vertices, "
            f"mode: {self.edit_mode}, modified: {user_modified}"
        )
        return "next"

    def _compute_visualization_overlays(self, image_path: Path):
        """Pre-compute edge and line detection overlays for visualization."""
        # Edge detection overlay
        gray = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2GRAY)
        median = np.median(gray)
        lower = int(max(0, median * 0.5))
        upper = int(min(255, median * 1.5))
        edges = cv2.Canny(gray, lower, upper)

        # Create colored edge overlay (red, semi-transparent)
        self.edges_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.edges_overlay = cv2.applyColorMap(self.edges_overlay, cv2.COLORMAP_HOT)

        # Line detection overlay with improved parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,        # Lower threshold to detect more lines
            minLineLength=50,    # Shorter minimum length
            maxLineGap=30        # Smaller gap for better continuity
        )

        self.lines_overlay = np.zeros_like(self.image_original)
        if lines is not None:
            self.logger.debug(f"Detected {len(lines)} lines for visualization")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.lines_overlay, (x1, y1), (x2, y2), self.color_lines, 2)
        else:
            self.logger.debug("No lines detected for visualization")

    def _render_display(
        self,
        camera_id: str,
        camera_index: int,
        total_cameras: int,
        vehicle_count: int
    ):
        """Render the display with side-by-side layout: image on left, info panel on right."""
        # Start with original image or blend with empty reference
        if self.show_empty_reference and self.empty_reference is not None:
            # Blend: 40% current image, 60% empty reference
            image_display = cv2.addWeighted(self.image_original, 0.4, self.empty_reference, 0.6, 0)
        else:
            image_display = self.image_original.copy()

        # Add edge overlay if enabled
        if self.show_edges and self.edges_overlay is not None:
            image_display = cv2.addWeighted(image_display, 0.7, self.edges_overlay, 0.3, 0)

        # Add line overlay if enabled
        if self.show_lines and self.lines_overlay is not None:
            image_display = cv2.addWeighted(image_display, 0.8, self.lines_overlay, 0.2, 0)

        # Draw YOLO boxes in green
        for box in self.current_yolo_boxes:
            xyxy = box.get("xyxy", [])
            if len(xyxy) == 4:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(image_display, (x1, y1), (x2, y2), self.color_yolo_box, 2)

        # Draw auto-suggestion in cyan (if in auto mode)
        if self.edit_mode == "auto" and self.auto_suggestion is not None:
            pts = self.auto_suggestion.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(image_display, [pts], True, self.color_auto_polygon, 3)

            # Draw vertices
            for point in self.auto_suggestion:
                cv2.circle(image_display, tuple(point.astype(int)), 6, self.color_vertex, -1)

        # Draw current polygon in yellow (if editing/manual)
        if len(self.current_polygon) > 0:
            pts = np.array(self.current_polygon, dtype=np.int32)

            # Polygon color depends on mode
            poly_color = self.color_user_polygon if self.edit_mode != "auto" else self.color_auto_polygon

            # Draw lines
            if len(self.current_polygon) > 1:
                cv2.polylines(image_display, [pts], False, poly_color, 3)

            # Draw closing line if complete
            if len(self.current_polygon) >= 3:
                cv2.polylines(image_display, [pts], True, poly_color, 3)

            # Draw vertices
            for vertex in self.current_polygon:
                cv2.circle(image_display, vertex, 6, self.color_vertex, -1)

        # Coverage visualization
        if self.show_coverage and len(self.current_polygon) >= 3:
            polygon = np.array(self.current_polygon, dtype=np.int32).reshape(-1, 1, 2)
            for box in self.current_yolo_boxes:
                xyxy = box.get("xyxy", [])
                if len(xyxy) == 4:
                    x1, y1, x2, y2 = xyxy
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Check if inside polygon
                    inside = cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0

                    # Draw centroid (green if inside, red if outside)
                    color = (0, 255, 0) if inside else (0, 0, 255)
                    cv2.circle(image_display, (cx, cy), 8, color, -1)

        # Create side-by-side display with info panel
        display = self._create_sidebyside_layout(image_display, camera_id, camera_index, total_cameras, vehicle_count)

        # Show display
        cv2.imshow(self.window_name, display)
        self.image_display = display

    def _create_sidebyside_layout(
        self,
        image_display: np.ndarray,
        camera_id: str,
        camera_index: int,
        total_cameras: int,
        vehicle_count: int
    ) -> np.ndarray:
        """Create side-by-side layout: image on left, info panel on right."""
        img_h, img_w = image_display.shape[:2]

        # Info panel width (400px)
        panel_width = 400

        # Create gray info panel
        panel = np.ones((img_h, panel_width, 3), dtype=np.uint8) * 40  # Dark gray

        # Add text to panel
        y = 40
        line_h = 35

        # Title
        cv2.putText(panel, "LANE ANNOTATION TOOL", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += line_h + 10

        # Camera info
        cv2.putText(panel, f"Camera: {camera_id}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += line_h

        cv2.putText(panel, f"Progress: {camera_index + 1}/{total_cameras}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += line_h + 20

        # Status section
        cv2.line(panel, (20, y), (panel_width - 20, y), (100, 100, 100), 1)
        y += 25

        # Mode
        mode_text = f"Mode: {self.edit_mode.upper()}"
        cv2.putText(panel, mode_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        y += line_h

        # Confidence (if auto mode)
        if self.edit_mode == "auto":
            conf_label = "HIGH" if self.suggestion_confidence >= 0.75 else "MEDIUM" if self.suggestion_confidence >= 0.5 else "LOW"
            conf_color = (0, 255, 0) if self.suggestion_confidence >= 0.75 else (0, 255, 255) if self.suggestion_confidence >= 0.5 else (0, 165, 255)
            cv2.putText(panel, f"Confidence: {self.suggestion_confidence:.2f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            y += 30
            cv2.putText(panel, f"({conf_label})", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            y += line_h

        # Polygon info
        vertex_count = len(self.current_polygon) if self.current_polygon else (
            len(self.auto_suggestion) if self.auto_suggestion is not None else 0
        )
        cv2.putText(panel, f"Vertices: {vertex_count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += line_h

        cv2.putText(panel, f"YOLO boxes: {len(self.current_yolo_boxes)}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += line_h

        # Lane count (highlight in yellow)
        cv2.putText(panel, f"Lane count: {self.current_lane_count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += line_h + 20

        # Active overlays
        overlays = []
        if self.show_edges:
            overlays.append("Edges")
        if self.show_lines:
            overlays.append("Lines")
        if self.show_coverage:
            overlays.append("Coverage")
        if self.show_empty_reference:
            overlays.append("Empty Ref")

        if overlays:
            cv2.line(panel, (20, y), (panel_width - 20, y), (100, 100, 100), 1)
            y += 25
            cv2.putText(panel, "ACTIVE OVERLAYS:", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y += 30
            for overlay in overlays:
                cv2.putText(panel, f"• {overlay}", (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
                y += 25
            y += 15

        # Controls section
        cv2.line(panel, (20, y), (panel_width - 20, y), (100, 100, 100), 1)
        y += 25
        cv2.putText(panel, "CONTROLS:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 35

        controls = [
            ("A", "Accept suggestion"),
            ("M", "Modify suggestion"),
            ("D", "Draw manual"),
            ("N", "Next (save)"),
            ("R", "Reset polygon"),
            ("S", "Skip camera"),
            ("Q", "Quit"),
            ("", ""),
            ("+", "Increase lanes"),
            ("-", "Decrease lanes"),
            ("", ""),
            ("X", "Toggle empty ref"),
            ("E", "Toggle edges"),
            ("V", "Toggle coverage"),
            ("", ""),
            ("LClick", "Add vertex"),
            ("RClick", "Remove vertex"),
        ]

        for key, desc in controls:
            if not key:
                y += 10
                continue
            cv2.putText(panel, f"[{key}]", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)
            cv2.putText(panel, desc, (90, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 28

        # Combine image and panel side-by-side
        combined = np.hstack([image_display, panel])

        return combined

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - add vertex (only in edit/manual mode)
            if self.edit_mode in ["edit", "manual"]:
                self.current_polygon.append((x, y))
                self.user_modified = True
                self.logger.debug(f"Added vertex at ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - remove last vertex
            if self.current_polygon:
                removed = self.current_polygon.pop()
                self.user_modified = True
                self.logger.debug(f"Removed vertex at {removed}")

    def _log_summary(self, lane_config: Dict[str, Any]):
        """Log summary of lane annotation results."""
        stats = lane_config.get("statistics", {})

        self.logger.info("=" * 60)
        self.logger.info("LANE ANNOTATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Cameras with lane polygons: {len(lane_config['cameras'])}")
        self.logger.info(f"Cameras skipped: {len(lane_config['skipped_cameras'])}")
        self.logger.info(f"\nStatistics:")
        self.logger.info(f"  Auto-accepted: {stats.get('auto_accepted', 0)}")
        self.logger.info(f"  User-modified: {stats.get('user_modified', 0)}")
        self.logger.info(f"  Manual drawn: {stats.get('manual_drawn', 0)}")
        self.logger.info(f"  Skipped: {stats.get('skipped', 0)}")

        if stats.get('total_cameras', 0) > 0:
            acceptance_rate = stats.get('auto_accepted', 0) / stats['total_cameras'] * 100
            self.logger.info(f"  Auto-acceptance rate: {acceptance_rate:.1f}%")

        self.logger.info("=" * 60)

    def validate(self) -> bool:
        """Validate lane polygon configuration."""
        if not self.output_path.exists():
            self.logger.error(f"Lane polygon config not found: {self.output_path}")
            return False

        try:
            lane_config = load_json(self.output_path)
        except Exception as e:
            self.logger.error(f"Failed to load lane config: {e}")
            return False

        # Validate structure
        required_fields = ["created_at", "mode", "cameras", "skipped_cameras"]
        for field in required_fields:
            if field not in lane_config:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate camera entries
        for camera_id, camera_data in lane_config["cameras"].items():
            if "polygons" not in camera_data:
                self.logger.error(f"Camera {camera_id} missing polygons")
                return False

            for poly in camera_data["polygons"]:
                polygon = poly.get("polygon", [])
                if not isinstance(polygon, list) or len(polygon) < 3:
                    self.logger.error(
                        f"Camera {camera_id} has invalid polygon (need at least 3 vertices)"
                    )
                    return False

        self.logger.info(
            f"Validation passed: {len(lane_config['cameras'])} cameras with lane polygons, "
            f"{len(lane_config['skipped_cameras'])} skipped"
        )
        return True


def main():
    """CLI entry point for Phase 4b."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4b: Interactive Lane Annotation Tool"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cameras that already have lane polygons"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation without executing phase"
    )
    args = parser.parse_args()

    # Initialize phase
    config = PipelineConfig()
    phase = LaneAnnotationTool(config)

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
            stats = data.get("statistics", {})

            print(f"\nLane polygons defined for {len(data['cameras'])} cameras")
            print(f"Skipped {len(data['skipped_cameras'])} cameras")
            print(f"\nStatistics:")
            print(f"  Auto-accepted: {stats.get('auto_accepted', 0)}")
            print(f"  User-modified: {stats.get('user_modified', 0)}")
            print(f"  Manual drawn: {stats.get('manual_drawn', 0)}")
            exit(0)
        else:
            print(f"✗ Phase failed: {result.get('reason', 'Unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
