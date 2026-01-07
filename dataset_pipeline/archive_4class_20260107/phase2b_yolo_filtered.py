"""
Phase 2b: YOLO Analysis with Polygon Filtering and Active Lane Detection

Re-runs YOLO detection with improvements:
1. Uses YOLO11m (medium model, more accurate than nano)
2. Filters detections to only count vehicles inside lane polygons
3. Active lane detection via X-coordinate clustering
4. Lane-aware traffic classification (normalizes by active lanes, not total)

Active lane detection:
- For crossings with >2 lanes, detects which lanes have vehicles
- Bins vehicles by X-coordinate into lane positions
- Prevents over-normalization when only some lanes are open
- Example: 20 cars in 2 active lanes (of 6 total) = 10 cars/lane (moderate)
  vs. 20 cars / 6 total lanes = 3.3 cars/lane (light) - INCORRECT

Traffic classification formula:
- vehicles_per_lane = total_vehicles / active_lanes
- empty: vehicles_per_lane < 1
- light: 1 <= vehicles_per_lane < 2.5
- moderate: 2.5 <= vehicles_per_lane < 5
- heavy: vehicles_per_lane >= 5
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from .base import PipelinePhase
from .config import PipelineConfig
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class YOLOFilteredAnalysisPhase(PipelinePhase):
    """
    Phase 2b: YOLO analysis with polygon filtering and lane-aware classification.
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize YOLO filtered analysis phase.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="yolo_filtered",
            description="Running YOLO with polygon filtering"
        )

        self.sample_manifest_path = self.config.base_dir / "sample_manifest.json"
        self.lane_polygons_path = self.config.base_dir / "lane_polygons.json"
        self.output_path = self.config.base_dir / "yolo_results_filtered.json"

        # YOLO model (medium instead of nano)
        self.model_name = "yolo11m.pt"
        self.confidence_threshold = 0.25
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute YOLO analysis with polygon filtering.

        Args:
            resume: If True, skip already processed images

        Returns:
            Dictionary with analysis results
        """
        # Load sample manifest
        self.logger.info(f"Loading sample manifest from {self.sample_manifest_path}")
        if not self.sample_manifest_path.exists():
            raise FileNotFoundError(
                f"Sample manifest not found. Run Phase 1 first."
            )

        manifest = load_json(self.sample_manifest_path)
        samples = manifest.get('samples', [])  # Phase 1b stores in 'samples', not 'selection'

        # Load lane polygons
        self.logger.info(f"Loading lane polygons from {self.lane_polygons_path}")
        if not self.lane_polygons_path.exists():
            raise FileNotFoundError(
                f"Lane polygons not found. Run Phase 4b first."
            )

        lane_config = load_json(self.lane_polygons_path)

        # Build lookup: camera_id -> (polygon, lane_count)
        lane_lookup = {}
        for camera_id, camera_data in lane_config.get('cameras', {}).items():
            polygons = camera_data.get('polygons', [])
            lane_count = camera_data.get('lane_count', 2)

            if polygons:
                polygon = np.array(polygons[0]['polygon'], dtype=np.float32)
                lane_lookup[camera_id] = {
                    'polygon': polygon,
                    'lane_count': lane_count
                }

        self.logger.info(f"Loaded {len(lane_lookup)} lane polygons")

        # Load YOLO model
        self.logger.info(f"Loading YOLO model: {self.model_name}")
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_name)
            self.logger.info(f"Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

        # Initialize results
        results = {
            'model': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'vehicle_classes': self.vehicle_classes,
            'polygon_filtering': True,
            'lane_aware_classification': True,
            'created_at': datetime.now().isoformat(),
            'analyses': []
        }

        # Process images
        self.logger.info(f"Processing {len(samples)} images...")

        processed_count = 0
        skipped_no_polygon = 0
        skipped_missing = 0

        with tqdm(total=len(samples), desc="YOLO analysis") as pbar:
            for sample in samples:
                camera_id = sample['camera_id']
                local_path = sample['local_path']

                # Check if camera has polygon
                if camera_id not in lane_lookup:
                    self.logger.warning(f"No polygon for camera {camera_id}, skipping")
                    skipped_no_polygon += 1
                    pbar.update(1)
                    continue

                # Get image path
                image_path = self.config.base_dir / local_path
                if not image_path.exists():
                    self.logger.warning(f"Image not found: {image_path}")
                    skipped_missing += 1
                    pbar.update(1)
                    continue

                try:
                    # Run YOLO detection
                    detection_results = model(
                        str(image_path),
                        conf=self.confidence_threshold,
                        classes=self.vehicle_classes,
                        verbose=False
                    )

                    # Extract boxes
                    all_boxes = []
                    if len(detection_results) > 0:
                        boxes = detection_results[0].boxes
                        for i in range(len(boxes)):
                            all_boxes.append({
                                'xyxy': boxes.xyxy[i].cpu().numpy().tolist(),
                                'confidence': float(boxes.conf[i]),
                                'class': int(boxes.cls[i])
                            })

                    # Filter boxes inside polygon
                    polygon_data = lane_lookup[camera_id]
                    polygon = polygon_data['polygon']
                    total_lanes = polygon_data['lane_count']

                    filtered_boxes = self._filter_boxes_by_polygon(all_boxes, polygon)

                    # Detect active lanes (only count lanes with vehicles)
                    vehicle_count = len(filtered_boxes)
                    active_lanes = self._detect_active_lanes(filtered_boxes, polygon, total_lanes)

                    # Lane-aware traffic classification (uses active lanes, not total)
                    traffic_level = self._classify_traffic_lane_aware(vehicle_count, active_lanes)

                    # Store analysis
                    analysis = {
                        'camera_id': camera_id,
                        'local_path': local_path,
                        'blob_name': sample.get('blob_name', ''),
                        'vehicle_count': vehicle_count,
                        'vehicle_count_total': len(all_boxes),  # Before filtering
                        'filtered_count': len(all_boxes) - vehicle_count,  # Removed by polygon
                        'traffic_level': traffic_level,
                        'lane_count': total_lanes,  # Total lanes at crossing
                        'active_lanes': active_lanes,  # Lanes with vehicles
                        'boxes': filtered_boxes
                    }

                    results['analyses'].append(analysis)
                    processed_count += 1

                except Exception as e:
                    self.logger.error(f"Failed to process {local_path}: {e}")

                pbar.update(1)

        # Save results
        self.logger.info(f"Saving results to {self.output_path}")
        save_json(results, self.output_path)

        # Log summary
        self.logger.info(f"Processed: {processed_count}")
        self.logger.info(f"Skipped (no polygon): {skipped_no_polygon}")
        self.logger.info(f"Skipped (missing): {skipped_missing}")

        return results

    def _filter_boxes_by_polygon(
        self,
        boxes: List[Dict[str, Any]],
        polygon: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Filter YOLO boxes to only keep vehicles inside polygon.

        Args:
            boxes: List of YOLO bounding boxes
            polygon: Polygon vertices (N, 2)

        Returns:
            Filtered list of boxes
        """
        filtered = []
        polygon_cv = polygon.reshape(-1, 1, 2).astype(np.float32)

        for box in boxes:
            xyxy = box['xyxy']

            # Calculate center point
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2

            # Check if inside polygon
            result = cv2.pointPolygonTest(polygon_cv, (float(cx), float(cy)), False)

            if result >= 0:  # Inside or on boundary
                filtered.append(box)

        return filtered

    def _detect_active_lanes(
        self,
        boxes: List[Dict[str, Any]],
        polygon: np.ndarray,
        total_lanes: int
    ) -> int:
        """
        Detect how many lanes are actually active based on vehicle distribution.

        Uses X-coordinate clustering to determine which lanes have vehicles.
        This prevents over-normalization when only some lanes are open at a
        border crossing with many total lanes.

        Args:
            boxes: List of filtered bounding boxes (vehicles inside polygon)
            polygon: Polygon vertices (N, 2) defining the lane area
            total_lanes: Total number of lanes at this crossing

        Returns:
            Number of lanes with at least one vehicle (active lanes)

        Example:
            - 6-lane crossing with only 2 lanes open
            - 20 vehicles detected, all in 2 lanes
            - Returns: 2 (not 6)
            - Classification: 20/2 = 10 cars/lane = moderate (not 20/6 = 3.3 = light)
        """
        # Skip clustering for small border crossings (<=2 lanes)
        if total_lanes <= 2:
            return total_lanes

        # If no vehicles, return total lanes (avoids division by zero later)
        if not boxes:
            return total_lanes

        # Calculate polygon width (min/max x coordinates)
        x_coords = polygon[:, 0]
        polygon_min_x = float(np.min(x_coords))
        polygon_width = float(np.max(x_coords) - polygon_min_x)

        # Avoid division by zero if polygon is degenerate
        if polygon_width == 0:
            return total_lanes

        # Create bins for each lane
        lane_bins = [0] * total_lanes

        # Assign each vehicle to a lane bin based on its center X coordinate
        for box in boxes:
            xyxy = box['xyxy']

            # Calculate center point
            cx = (xyxy[0] + xyxy[2]) / 2

            # Normalize x to polygon-relative coordinate (0 to 1)
            normalized_x = (cx - polygon_min_x) / polygon_width

            # Clamp to [0, 1] range (handle edge cases)
            normalized_x = max(0.0, min(1.0, normalized_x))

            # Assign to lane bin
            lane_idx = min(int(normalized_x * total_lanes), total_lanes - 1)
            lane_bins[lane_idx] += 1

        # Count bins with at least one vehicle
        active_lanes = sum(1 for count in lane_bins if count > 0)

        # Sanity check: should have at least 1 active lane if we have vehicles
        if active_lanes < 1:
            active_lanes = total_lanes

        return active_lanes

    def _classify_traffic_lane_aware(
        self,
        vehicle_count: int,
        active_lanes: int
    ) -> str:
        """
        Classify traffic level using lane-aware thresholds.

        Formula:
        - vehicles_per_lane = vehicle_count / active_lanes
        - empty: < 1 vehicle per lane
        - light: 1-2.5 vehicles per lane
        - moderate: 2.5-5 vehicles per lane
        - heavy: >= 5 vehicles per lane

        Args:
            vehicle_count: Number of vehicles detected (inside polygon)
            active_lanes: Number of lanes with vehicles (detected by clustering)

        Returns:
            Traffic level string with "likely_" prefix

        Note:
            Uses active_lanes (not total_lanes) to avoid over-normalization
            when only some lanes are open at multi-lane crossings.
        """
        if active_lanes == 0:
            active_lanes = 1  # Avoid division by zero

        vehicles_per_lane = vehicle_count / active_lanes

        if vehicles_per_lane < 1.0:
            return "likely_empty"
        elif vehicles_per_lane < 2.5:
            return "likely_light"
        elif vehicles_per_lane < 5.0:
            return "likely_moderate"
        else:
            return "likely_heavy"

    def validate(self) -> bool:
        """Validate that YOLO analysis completed successfully."""
        if not self.output_path.exists():
            self.logger.error("YOLO results file not found")
            return False

        try:
            results = load_json(self.output_path)

            if 'analyses' not in results:
                self.logger.error("Missing 'analyses' in results")
                return False

            num_analyses = len(results['analyses'])
            self.logger.info(f"Validation passed: {num_analyses} images analyzed")
            return True

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False


def main():
    """CLI entry point for Phase 2b."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2b: YOLO with polygon filtering and lane-aware classification"
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
    phase = YOLOFilteredAnalysisPhase(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        if result['status'] == 'completed':
            print(f"\n✓ Phase 2b completed successfully in {result['duration_seconds']:.1f} seconds")
            data = result['data']
            print(f"  Analyzed: {len(data['analyses'])} images")
            print(f"  Model: {data['model']}")
            print(f"  Polygon filtering: {data['polygon_filtering']}")
            print(f"  Lane-aware classification: {data['lane_aware_classification']}")
            exit(0)
        else:
            print(f"\n✗ Phase 2b failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
