"""
Phase 2: YOLO Analysis

Runs YOLO vehicle detection on all downloaded images to count vehicles
and assign preliminary traffic density labels.

Automatic Polygon Filtering:
- If lane_polygons.json exists, automatically filters detections to only
  count vehicles inside lane polygons (excludes parked cars outside lanes)
- Uses active lane detection to normalize vehicle count by lanes in use
- Falls back to basic vehicle counting if no lane polygons
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

from .base import PipelinePhase
from .config import PipelineConfig, YOLO_CONFIG, categorize_traffic
from .utils import load_json, save_json, validate_image_path

logger = logging.getLogger(__name__)


class YOLOAnalysisPhase(PipelinePhase):
    """
    Phase 2: Run YOLO vehicle detection on downloaded images.

    Analyzes each image to:
    - Count vehicles (cars, motorcycles, buses, trucks)
    - Categorize traffic level (likely_empty, likely_light, likely_moderate, likely_heavy)
    - Store bounding boxes for potential visualization
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        model_name: str = None,
        confidence_threshold: float = None,
        device: str = None
    ):
        """
        Initialize YOLO analysis phase.

        Args:
            pipeline_config: Pipeline configuration
            model_name: YOLO model to use (default: from YOLO_CONFIG)
            confidence_threshold: Detection confidence threshold (default: from YOLO_CONFIG)
            device: Device to use ('mps', 'cpu', or 'cuda'). Auto-detect if None
        """
        super().__init__(
            config=pipeline_config,
            phase_name="yolo",
            description="Running YOLO vehicle detection"
        )

        # YOLO configuration
        self.model_name = model_name or YOLO_CONFIG["model"]
        self.confidence_threshold = confidence_threshold or YOLO_CONFIG["confidence_threshold"]
        self.vehicle_classes = YOLO_CONFIG["classes"]  # [2, 3, 5, 7] for car, motorcycle, bus, truck

        # Device selection
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device

        self.model = None

        # Polygon filtering (optional - loaded if lane_polygons.json exists)
        self.lane_polygons_path = self.config.base_dir / "lane_polygons.json"
        self.use_polygon_filtering = False
        self.lane_lookup = {}

    def _detect_device(self) -> str:
        """
        Auto-detect best available device for YOLO inference.

        Returns:
            Device string ('mps', 'cuda', or 'cpu')
        """
        try:
            import torch

            # Check for Apple Silicon MPS
            if torch.backends.mps.is_available():
                self.logger.info("MPS (Metal Performance Shaders) available - using Apple Silicon GPU")
                return "mps"

            # Check for CUDA
            if torch.cuda.is_available():
                self.logger.info("CUDA available - using NVIDIA GPU")
                return "cuda"

            # Fallback to CPU
            self.logger.info("No GPU available - using CPU")
            return "cpu"

        except ImportError:
            self.logger.warning("PyTorch not available - defaulting to CPU")
            return "cpu"

    def load_model(self):
        """
        Load YOLO model with specified configuration.

        Raises:
            ImportError: If ultralytics is not installed
            Exception: If model loading fails
        """
        if self.model is not None:
            return  # Already loaded

        try:
            from ultralytics import YOLO

            self.logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(f"{self.model_name}.pt")

            # Test the model on device
            self.logger.info(f"Initializing model on device: {self.device}")

        except ImportError:
            raise ImportError(
                "ultralytics package not found. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            raise Exception(f"Failed to load YOLO model: {e}")

    def load_lane_polygons(self):
        """
        Load lane polygons if available for polygon filtering.

        Returns:
            True if polygons loaded, False otherwise
        """
        if not self.lane_polygons_path.exists():
            self.logger.info("No lane polygons found - using basic vehicle counting")
            self.use_polygon_filtering = False
            return False

        try:
            self.logger.info(f"Loading lane polygons from {self.lane_polygons_path}")
            lane_config = load_json(self.lane_polygons_path)

            # Build lookup: camera_id -> (polygon, lane_count)
            self.lane_lookup = {}
            for camera_id, camera_data in lane_config.get('cameras', {}).items():
                polygons = camera_data.get('polygons', [])
                lane_count = camera_data.get('lane_count', 2)

                if polygons:
                    polygon = np.array(polygons[0]['polygon'], dtype=np.float32)
                    self.lane_lookup[camera_id] = {
                        'polygon': polygon,
                        'lane_count': lane_count
                    }

            self.logger.info(
                f"Loaded {len(self.lane_lookup)} lane polygons - using polygon filtering"
            )
            self.use_polygon_filtering = True
            return True

        except Exception as e:
            self.logger.warning(f"Failed to load lane polygons: {e}")
            self.use_polygon_filtering = False
            return False

    def analyze_image(
        self,
        image_path: Path,
        camera_id: str = None,
        verbose: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Run YOLO inference on a single image.

        Args:
            image_path: Path to image file
            camera_id: Camera ID for polygon filtering (optional)
            verbose: Whether to show YOLO output

        Returns:
            Analysis result dictionary, or None if analysis failed
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate image exists
        if not validate_image_path(image_path, min_size_bytes=1000):
            self.logger.warning(f"Invalid or missing image: {image_path}")
            return None

        try:
            # Run inference
            results = self.model(
                str(image_path),
                conf=self.confidence_threshold,
                classes=self.vehicle_classes,
                device=self.device,
                verbose=verbose
            )

            # Extract detections
            detections = results[0]
            boxes = detections.boxes

            # Extract bounding box details
            box_data = []
            for box in boxes:
                box_data.append({
                    "xyxy": box.xyxy[0].cpu().tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0].cpu()),
                    "class": int(box.cls[0].cpu())
                })

            # Apply polygon filtering if available
            if self.use_polygon_filtering and camera_id and camera_id in self.lane_lookup:
                polygon_data = self.lane_lookup[camera_id]
                polygon = polygon_data['polygon']
                total_lanes = polygon_data['lane_count']

                # Filter boxes inside polygon
                filtered_boxes = self._filter_boxes_by_polygon(box_data, polygon)
                vehicle_count = len(filtered_boxes)

                # Detect active lanes and classify traffic
                active_lanes = self._detect_active_lanes(filtered_boxes, polygon, total_lanes)
                traffic_level = self._classify_traffic_lane_aware(vehicle_count, active_lanes)

                return {
                    "vehicle_count": vehicle_count,
                    "vehicle_count_total": len(box_data),  # Before filtering
                    "filtered_count": len(box_data) - vehicle_count,  # Removed by polygon
                    "traffic_level": traffic_level,
                    "lane_count": total_lanes,
                    "active_lanes": active_lanes,
                    "boxes": filtered_boxes
                }
            else:
                # Basic vehicle counting (no polygon filtering)
                vehicle_count = len(box_data)
                traffic_level = categorize_traffic(vehicle_count)

                return {
                    "vehicle_count": vehicle_count,
                    "traffic_level": traffic_level,
                    "boxes": box_data
                }

        except Exception as e:
            self.logger.error(f"Error analyzing {image_path}: {e}")
            return None

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute YOLO analysis phase.

        Args:
            resume: If True, skip images that were already analyzed

        Returns:
            Dictionary with analysis results and statistics
        """
        manifest_file = self.config.get_path(self.config.sample_manifest_file)
        results_file = self.config.get_path(self.config.yolo_results_file)

        # Check if we can resume
        existing_results = None
        if resume and results_file.exists():
            self.logger.info(f"Results file already exists at {results_file}")
            existing_results = load_json(results_file)

            if existing_results and existing_results.get("analyses"):
                self.logger.info(
                    f"Found {len(existing_results['analyses'])} existing analyses"
                )

        # Load manifest
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {manifest_file}. "
                "Run Phase 1b (sample) first."
            )

        manifest = load_json(manifest_file)
        if not manifest or not manifest.get("samples"):
            raise ValueError("Manifest is empty or has no samples")

        samples = manifest["samples"]
        total_samples = len(samples)

        self.logger.info(
            f"Starting YOLO analysis of {total_samples} images..."
        )
        self.logger.info(
            f"Model: {self.model_name}, Device: {self.device}, "
            f"Confidence: {self.confidence_threshold}"
        )

        # Load YOLO model
        self.load_model()

        # Load lane polygons if available
        self.load_lane_polygons()

        # Track which images have been analyzed (for resume)
        analyzed_paths = set()
        if existing_results:
            for analysis in existing_results.get("analyses", []):
                analyzed_paths.add(analysis["local_path"])

        # Initialize results
        results = {
            "model": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "vehicle_classes": self.vehicle_classes,
            "polygon_filtering": self.use_polygon_filtering,
            "lane_aware_classification": self.use_polygon_filtering,
            "created_at": datetime.now().isoformat(),
            "analyses": existing_results.get("analyses", []) if existing_results else []
        }

        # Initialize progress tracking
        progress = self.initialize_progress(total=total_samples)

        # Statistics
        stats = {
            "success": len(analyzed_paths) if resume else 0,
            "skipped": 0,
            "failed": 0,
            "by_traffic_level": {
                "likely_empty": 0,
                "likely_light": 0,
                "likely_moderate": 0,
                "likely_heavy": 0
            }
        }

        # Count existing analyses by traffic level
        if existing_results:
            for analysis in existing_results.get("analyses", []):
                level = analysis.get("traffic_level", "")
                if level in stats["by_traffic_level"]:
                    stats["by_traffic_level"][level] += 1

        # Process each sample
        with tqdm(total=total_samples, desc="Analyzing images") as pbar:
            # Update progress bar to account for existing analyses
            if resume:
                pbar.update(len(analyzed_paths))

            for idx, sample in enumerate(samples):
                local_path = sample["local_path"]
                camera_id = sample.get("camera_id")

                # Skip if already analyzed (resume mode)
                if resume and local_path in analyzed_paths:
                    stats["skipped"] += 1
                    continue

                # Construct full image path
                image_path = self.config.base_dir / local_path

                # Analyze image (with camera_id for polygon filtering)
                analysis_result = self.analyze_image(
                    image_path,
                    camera_id=camera_id,
                    verbose=False
                )

                if analysis_result:
                    # Add metadata to result
                    result_entry = {
                        "camera_id": sample["camera_id"],
                        "local_path": local_path,
                        "blob_name": sample.get("blob_name", ""),
                        **analysis_result
                    }

                    results["analyses"].append(result_entry)
                    stats["success"] += 1

                    # Update traffic level stats
                    traffic_level = analysis_result["traffic_level"]
                    stats["by_traffic_level"][traffic_level] += 1
                else:
                    stats["failed"] += 1
                    self.logger.warning(f"Failed to analyze: {local_path}")

                # Update progress
                progress["completed"] = stats["success"] + stats["skipped"] + stats["failed"]

                # Save progress every 100 images
                if progress["completed"] % 100 == 0:
                    self.save_progress()
                    # Also save intermediate results
                    save_json(results, results_file)

                pbar.update(1)
                pbar.set_postfix({
                    "success": stats["success"],
                    "skipped": stats["skipped"],
                    "failed": stats["failed"]
                })

        # Update results with statistics
        results["statistics"] = {
            "total_analyzed": len(results["analyses"]),
            "by_traffic_level": stats["by_traffic_level"]
        }

        # Save final results
        self.logger.info(f"Saving results to {results_file}")
        save_json(results, results_file)

        # Final progress save
        progress.update(stats)
        self.save_progress()

        self.logger.info(
            f"Analysis complete: {stats['success']} succeeded, "
            f"{stats['skipped']} skipped, {stats['failed']} failed"
        )

        return {
            "results": results,
            "statistics": stats,
            "resumed": resume and len(analyzed_paths) > 0
        }

    def validate(self) -> bool:
        """
        Validate that YOLO analysis completed successfully.

        Returns:
            True if results file exists and has reasonable data
        """
        results_file = self.config.get_path(self.config.yolo_results_file)

        if not results_file.exists():
            self.logger.error(f"Results file not found: {results_file}")
            return False

        results = load_json(results_file)

        if not results or not results.get("analyses"):
            self.logger.error("Results file is empty or has no analyses")
            return False

        # Check that we have reasonable number of analyses
        num_analyses = len(results["analyses"])

        # Load manifest to check expected count
        manifest_file = self.config.get_path(self.config.sample_manifest_file)
        if manifest_file.exists():
            manifest = load_json(manifest_file)
            expected_count = len(manifest.get("samples", []))

            # Should have at least 90% of expected analyses
            if num_analyses < 0.9 * expected_count:
                self.logger.warning(
                    f"Only {num_analyses}/{expected_count} images analyzed "
                    f"({num_analyses/expected_count*100:.1f}%)"
                )
                return False

        # Check traffic level distribution
        stats = results.get("statistics", {})
        by_level = stats.get("by_traffic_level", {})

        self.logger.info(f"Validation passed: {num_analyses} images analyzed")
        self.logger.info(f"Traffic level distribution: {by_level}")

        return True

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


def main():
    """
    CLI entry point for Phase 2: YOLO Analysis.

    Usage:
        python -m dataset_pipeline.phase2_yolo [--resume] [--device DEVICE]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2: YOLO vehicle detection analysis"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume analysis, skipping already analyzed images"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cpu", "cuda"],
        default=None,
        help="Device to use for inference (default: auto-detect)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"YOLO model to use (default: {YOLO_CONFIG['model']})"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=f"Confidence threshold (default: {YOLO_CONFIG['confidence_threshold']})"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation"
    )

    args = parser.parse_args()

    # Load configuration
    pipeline_config = PipelineConfig()

    # Ensure directories exist
    pipeline_config.ensure_directories()

    # Create phase
    phase = YOLOAnalysisPhase(
        pipeline_config,
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )

    if args.validate_only:
        # Only run validation
        print("Running validation...")
        if phase.validate():
            print("✓ Validation passed")
            exit(0)
        else:
            print("✗ Validation failed")
            exit(1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        # Print results
        print("\n" + "=" * 60)
        print("Phase 2: YOLO Analysis - Complete")
        print("=" * 60)

        if result["status"] == "completed":
            data = result["data"]
            stats = data.get("statistics", {})

            print(f"Status: ✓ Success")
            print(f"Duration: {result['duration_seconds']:.1f} seconds")
            print(f"\nResults:")
            print(f"  Total analyzed: {stats.get('success', 0)}")
            print(f"  Skipped (existing): {stats.get('skipped', 0)}")
            print(f"  Failed: {stats.get('failed', 0)}")

            # Traffic level distribution
            by_level = stats.get("by_traffic_level", {})
            if by_level:
                print(f"\nTraffic Level Distribution:")
                for level, count in by_level.items():
                    print(f"    {level}: {count}")

            if data.get("resumed"):
                print(f"\n(Resumed from existing results)")

            print("=" * 60)
            exit(0)
        else:
            print(f"Status: ✗ Failed")
            print(f"Reason: {result.get('reason', 'Unknown')}")
            print("=" * 60)
            exit(1)


if __name__ == "__main__":
    main()
