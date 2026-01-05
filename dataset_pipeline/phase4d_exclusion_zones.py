"""
Phase 4d: YOLO Failure Regions Annotation Tool

Interactive tool to mark areas where YOLO fails to detect vehicles but they are
visible to humans. These regions will be cropped and used to train a secondary
congestion detection model.

Use cases:
- Overflow/queue areas where cars wait during heavy congestion
- Areas with bad lighting, glare, or shadows where YOLO misses vehicles
- Regions with difficult camera angles where YOLO has false negatives
- Any area where visible vehicles are NOT detected by YOLO

Purpose:
- Mark regions that fill with cars during peak congestion
- Crop these regions separately (Phase 5)
- Train a secondary model to detect traffic in YOLO-blind areas
- Supplement YOLO with additional congestion signals

Controls:
- Left-click: Add vertex to failure region polygon
- Right-click: Remove last vertex
- [C] Complete current polygon and start new one
- [D] Delete last completed polygon
- [R] Reset all polygons for current camera
- [N] Next camera (save and continue)
- [S] Skip camera (no failure regions for this camera)
- [P] Previous camera
- [Q] Quit and save progress
- [V] Toggle YOLO detection overlay
- [L] Toggle lane polygon overlay
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


class YOLOFailureRegionsTool(PipelinePhase):
    """
    Phase 4d: Interactive YOLO failure regions annotation tool.

    For each camera:
    1. Display lane polygon boundaries (from Phase 4b)
    2. Show YOLO detection overlay (from Phase 2b)
    3. User draws polygons around areas where:
       - Vehicles are visible to humans
       - But YOLO failed to detect them (false negatives)
    4. Save to yolo_failure_regions.json

    These regions will be cropped separately and used to train a secondary
    congestion detection model for YOLO-blind areas.
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize YOLO failure regions annotation tool.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="yolo_failure_regions",
            description="Annotating YOLO failure regions for secondary model"
        )

        self.roi_references_path = self.config.base_dir / "roi_references.json"
        self.lane_polygons_path = self.config.base_dir / "lane_polygons.json"
        self.yolo_results_path = self.config.base_dir / "yolo_results_filtered.json"  # Use filtered results from Phase 2b
        self.output_path = self.config.base_dir / "yolo_failure_regions.json"

        # GUI state
        self.current_polygon: List[Tuple[int, int]] = []
        self.completed_polygons: List[List[Tuple[int, int]]] = []
        self.current_camera_id: Optional[str] = None
        self.current_lane_polygon: Optional[np.ndarray] = None
        self.image_display: Optional[np.ndarray] = None
        self.image_original: Optional[np.ndarray] = None
        self.window_name = "YOLO Failure Regions Tool"

        # Visualization toggles
        self.show_yolo: bool = True
        self.show_lanes: bool = True

        # Current camera data
        self.current_yolo_boxes: List[Dict[str, Any]] = []

        # Colors (BGR format for OpenCV)
        self.color_yolo_box = (0, 255, 0)          # Green for YOLO detected vehicles
        self.color_lane_polygon = (255, 255, 0)    # Cyan for lane boundaries
        self.color_failure_region = (0, 0, 255)    # Red for YOLO failure regions
        self.color_current = (0, 165, 255)         # Orange for current polygon

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute interactive YOLO failure regions annotation.

        Args:
            resume: If True, load existing progress and continue

        Returns:
            Dictionary with annotation results
        """
        # Load required data
        self.logger.info("Loading reference images, polygons, and YOLO detections...")
        roi_references = self._load_roi_references()
        lane_config = self._load_lane_polygons()
        yolo_results = self._load_yolo_results()

        # Build YOLO lookup: camera_id -> list of boxes
        yolo_lookup = self._build_yolo_lookup(yolo_results)

        # Initialize or load failure regions config
        if resume and self.output_path.exists():
            self.logger.info(f"Resuming from {self.output_path}")
            failure_config = load_json(self.output_path)
        else:
            failure_config = {
                'created_at': datetime.now().isoformat(),
                'description': 'Regions where YOLO fails to detect vehicles but they are visible to humans',
                'purpose': 'Train secondary congestion detection model for YOLO-blind areas',
                'cameras': {},
                'statistics': {
                    'total_cameras': 0,
                    'cameras_with_failure_regions': 0,
                    'cameras_skipped': 0,
                    'total_failure_regions': 0
                }
            }

        # Get list of cameras to annotate (from lane_polygons.json)
        camera_ids = list(lane_config.get('cameras', {}).keys())
        self.logger.info(f"Found {len(camera_ids)} cameras to annotate")

        # Filter to only unannotated cameras if resuming
        if resume:
            camera_ids = [
                cid for cid in camera_ids
                if cid not in failure_config.get('cameras', {})
            ]
            self.logger.info(f"Resuming: {len(camera_ids)} cameras remaining")

        if not camera_ids:
            self.logger.info("All cameras already annotated")
            return failure_config

        # Setup OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 900)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Process each camera
        current_idx = 0
        while current_idx < len(camera_ids):
            camera_id = camera_ids[current_idx]
            self.current_camera_id = camera_id

            self.logger.info(f"\n[{current_idx + 1}/{len(camera_ids)}] Annotating: {camera_id}")

            # Load reference image
            if camera_id not in roi_references:
                self.logger.warning(f"No reference image for {camera_id}, skipping")
                current_idx += 1
                continue

            ref_image_path = self.config.base_dir / roi_references[camera_id].get('image_path', '')
            if not ref_image_path.exists():
                self.logger.warning(f"Reference image not found: {ref_image_path}")
                current_idx += 1
                continue

            # Load image
            self.image_original = cv2.imread(str(ref_image_path))
            if self.image_original is None:
                self.logger.error(f"Failed to load image: {ref_image_path}")
                current_idx += 1
                continue

            # Load lane polygon for this camera
            camera_data = lane_config['cameras'].get(camera_id, {})
            polygons = camera_data.get('polygons', [])
            if polygons:
                self.current_lane_polygon = np.array(polygons[0]['polygon'], dtype=np.int32)
            else:
                self.current_lane_polygon = None

            # Load YOLO boxes for this camera
            self.current_yolo_boxes = yolo_lookup.get(camera_id, [])

            # Reset annotation state
            self.current_polygon = []
            self.completed_polygons = []

            # Main annotation loop for this camera
            action = self._annotate_camera()

            if action == 'next':
                # Save failure regions for this camera
                failure_config['cameras'][camera_id] = {
                    'reference_image': str(ref_image_path.relative_to(self.config.base_dir)),
                    'failure_regions': [
                        {'polygon': poly, 'description': 'YOLO-blind area with potential vehicles'}
                        for poly in self.completed_polygons
                    ],
                    'annotated_at': datetime.now().isoformat()
                }
                self.logger.info(f"Saved {len(self.completed_polygons)} failure regions for {camera_id}")

                # Save progress
                self._save_progress(failure_config)
                current_idx += 1

            elif action == 'skip':
                # No failure regions for this camera
                failure_config['cameras'][camera_id] = {
                    'reference_image': str(ref_image_path.relative_to(self.config.base_dir)),
                    'failure_regions': [],
                    'skipped': True,
                    'annotated_at': datetime.now().isoformat()
                }
                self.logger.info(f"Skipped {camera_id} (no YOLO failure regions)")

                # Save progress
                self._save_progress(failure_config)
                current_idx += 1

            elif action == 'previous':
                if current_idx > 0:
                    current_idx -= 1
                else:
                    self.logger.warning("Already at first camera")

            elif action == 'quit':
                self.logger.info("User requested quit")
                break

        # Cleanup
        cv2.destroyAllWindows()

        # Update statistics
        failure_config['statistics']['total_cameras'] = len(camera_ids)
        failure_config['statistics']['cameras_with_failure_regions'] = sum(
            1 for cam_data in failure_config['cameras'].values()
            if len(cam_data.get('failure_regions', [])) > 0
        )
        failure_config['statistics']['cameras_skipped'] = sum(
            1 for cam_data in failure_config['cameras'].values()
            if cam_data.get('skipped', False)
        )
        failure_config['statistics']['total_failure_regions'] = sum(
            len(cam_data.get('failure_regions', []))
            for cam_data in failure_config['cameras'].values()
        )

        # Final save
        self._save_progress(failure_config)

        self.logger.info(f"\nYOLO failure regions annotation completed!")
        self.logger.info(f"  Cameras with failure regions: {failure_config['statistics']['cameras_with_failure_regions']}")
        self.logger.info(f"  Total failure regions: {failure_config['statistics']['total_failure_regions']}")
        self.logger.info(f"  Cameras skipped: {failure_config['statistics']['cameras_skipped']}")
        self.logger.info(f"  Saved to: {self.output_path}")

        return failure_config

    def _annotate_camera(self) -> str:
        """
        Annotation loop for a single camera.

        Returns:
            Action to take: 'next', 'skip', 'previous', 'quit'
        """
        while True:
            # Render display
            self._render_display()

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):  # Next
                if not self.completed_polygons and not self.current_polygon:
                    self.logger.warning("No failure regions drawn. Use [S] to skip if YOLO works well everywhere.")
                    continue
                if self.current_polygon:
                    self.logger.warning("Complete current polygon first with [C]")
                    continue
                return 'next'

            elif key == ord('s'):  # Skip
                return 'skip'

            elif key == ord('p'):  # Previous
                return 'previous'

            elif key == ord('q'):  # Quit
                return 'quit'

            elif key == ord('c'):  # Complete current polygon
                if len(self.current_polygon) >= 3:
                    self.completed_polygons.append(self.current_polygon.copy())
                    self.current_polygon = []
                    self.logger.info(f"Completed polygon {len(self.completed_polygons)}")
                else:
                    self.logger.warning("Need at least 3 points to complete polygon")

            elif key == ord('d'):  # Delete last completed polygon
                if self.completed_polygons:
                    deleted = self.completed_polygons.pop()
                    self.logger.info(f"Deleted polygon with {len(deleted)} vertices")
                else:
                    self.logger.warning("No completed polygons to delete")

            elif key == ord('r'):  # Reset all
                self.current_polygon = []
                self.completed_polygons = []
                self.logger.info("Reset all polygons")

            elif key == ord('v'):  # Toggle YOLO overlay
                self.show_yolo = not self.show_yolo
                self.logger.info(f"YOLO overlay: {'ON' if self.show_yolo else 'OFF'}")

            elif key == ord('l'):  # Toggle lane overlay
                self.show_lanes = not self.show_lanes
                self.logger.info(f"Lane overlay: {'ON' if self.show_lanes else 'OFF'}")

    def _render_display(self):
        """Render the current annotation display."""
        # Start with original image
        display = self.image_original.copy()

        # Draw lane polygon (cyan, semi-transparent)
        if self.show_lanes and self.current_lane_polygon is not None:
            overlay = display.copy()
            cv2.polylines(overlay, [self.current_lane_polygon], True, self.color_lane_polygon, 3)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Draw YOLO boxes (green)
        if self.show_yolo:
            for box in self.current_yolo_boxes:
                xyxy = box['xyxy']
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(display, (x1, y1), (x2, y2), self.color_yolo_box, 2)

        # Draw completed failure region polygons (red, filled with transparency)
        for poly in self.completed_polygons:
            if poly:
                poly_np = np.array(poly, dtype=np.int32)
                overlay = display.copy()
                cv2.fillPoly(overlay, [poly_np], self.color_failure_region)
                cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
                cv2.polylines(display, [poly_np], True, self.color_failure_region, 2)

        # Draw current polygon (orange)
        if self.current_polygon:
            poly_np = np.array(self.current_polygon, dtype=np.int32)
            cv2.polylines(display, [poly_np], False, self.color_current, 2)

            # Draw vertices
            for point in self.current_polygon:
                cv2.circle(display, point, 5, self.color_current, -1)

        # Add instructions overlay
        self._draw_instructions(display)

        # Show display
        self.image_display = display
        cv2.imshow(self.window_name, display)

    def _draw_instructions(self, image: np.ndarray):
        """Draw instruction text on image."""
        instructions = [
            f"Camera: {self.current_camera_id}",
            f"Failure regions marked: {len(self.completed_polygons)}",
            f"Current polygon vertices: {len(self.current_polygon)}",
            "",
            "Mark areas where: Cars visible but YOLO missed them",
            "Left-click: Add vertex | Right-click: Remove last",
            "[C] Complete polygon | [D] Delete last | [R] Reset all",
            "[N] Next | [S] Skip (YOLO works well) | [Q] Quit",
            "[V] Toggle YOLO boxes | [L] Toggle lane polygon",
        ]

        y_offset = 30
        for line in instructions:
            cv2.putText(
                image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 25

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add vertex to current polygon
            self.current_polygon.append((x, y))
            self.logger.debug(f"Added vertex: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last vertex
            if self.current_polygon:
                removed = self.current_polygon.pop()
                self.logger.debug(f"Removed vertex: {removed}")

    def _load_roi_references(self) -> Dict[str, Any]:
        """Load ROI reference images."""
        if not self.roi_references_path.exists():
            raise FileNotFoundError(
                f"ROI references not found at {self.roi_references_path}. "
                "Run Phase 4b first."
            )
        return load_json(self.roi_references_path)

    def _load_lane_polygons(self) -> Dict[str, Any]:
        """Load lane polygons."""
        if not self.lane_polygons_path.exists():
            raise FileNotFoundError(
                f"Lane polygons not found at {self.lane_polygons_path}. "
                "Run Phase 4b first."
            )
        return load_json(self.lane_polygons_path)

    def _load_yolo_results(self) -> Dict[str, Any]:
        """Load YOLO results."""
        if not self.yolo_results_path.exists():
            self.logger.warning(f"YOLO results not found, continuing without YOLO overlay")
            return {'analyses': []}
        return load_json(self.yolo_results_path)

    def _build_yolo_lookup(self, yolo_results: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Build lookup: camera_id -> YOLO boxes."""
        lookup = {}
        for analysis in yolo_results.get('analyses', []):
            camera_id = analysis.get('camera_id')
            boxes = analysis.get('boxes', [])
            if camera_id not in lookup:
                lookup[camera_id] = []
            lookup[camera_id].extend(boxes)
        return lookup

    def _save_progress(self, failure_config: Dict[str, Any]):
        """Save current progress to file."""
        save_json(failure_config, self.output_path)
        self.logger.debug(f"Saved progress to {self.output_path}")

    def validate(self) -> bool:
        """Validate that YOLO failure regions were annotated successfully."""
        if not self.output_path.exists():
            self.logger.error("YOLO failure regions file not found")
            return False

        try:
            failure_config = load_json(self.output_path)

            if 'cameras' not in failure_config:
                self.logger.error("Missing 'cameras' in failure regions config")
                return False

            num_cameras = len(failure_config['cameras'])
            num_regions = failure_config.get('statistics', {}).get('total_failure_regions', 0)
            self.logger.info(f"Validation passed: {num_cameras} cameras annotated, {num_regions} total failure regions")
            return True

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False


def main():
    """CLI entry point for Phase 4d."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4d: Annotate YOLO failure regions for secondary model training"
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
    phase = YOLOFailureRegionsTool(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        if result['status'] == 'completed':
            print(f"\n✓ Phase 4d completed successfully in {result['duration_seconds']:.1f} seconds")
            data = result['data']
            stats = data.get('statistics', {})
            print(f"  Cameras with failure regions: {stats.get('cameras_with_failure_regions', 0)}")
            print(f"  Total failure regions: {stats.get('total_failure_regions', 0)}")
            print(f"  Cameras skipped: {stats.get('cameras_skipped', 0)}")
            print(f"\nThese regions will be cropped in Phase 5 for secondary model training.")
            exit(0)
        else:
            print(f"\n✗ Phase 4d failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
