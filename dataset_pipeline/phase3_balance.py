"""
Phase 3: Traffic-Balanced Selection

Selects final 500 images per camera from the YOLO-analyzed samples,
ensuring balanced distribution across 4 traffic density levels.
Also identifies reference images for ROI definition.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime

from .base import PipelinePhase
from .config import PipelineConfig, TRAFFIC_LEVELS
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class BalancedSelectionPhase(PipelinePhase):
    """
    Phase 3: Select balanced samples across traffic levels.

    For each camera:
    - Select 500 images total
    - Target 125 images per traffic level (empty, light, moderate, heavy)
    - Prioritize balance over quantity if insufficient samples
    - Identify highest-traffic image as ROI reference
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize balanced selection phase.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="balance",
            description="Selecting traffic-balanced samples"
        )

        self.yolo_results_path = self.config.base_dir / self.config.yolo_results_file
        self.output_path = self.config.base_dir / "balanced_selection.json"
        self.roi_references_path = self.config.base_dir / "roi_references.json"

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute balanced selection.

        Args:
            resume: If True, skip if output already exists

        Returns:
            Dictionary with selection results and statistics
        """
        # Check if already completed and resume requested
        if resume and self.output_path.exists():
            self.logger.info("Balanced selection already completed, loading existing results")
            selection = load_json(self.output_path)
            roi_references = load_json(self.roi_references_path)
            return {
                "selection": selection,
                "roi_references": roi_references,
                "resumed": True
            }

        # Load YOLO results
        self.logger.info(f"Loading YOLO results from {self.yolo_results_path}")
        if not self.yolo_results_path.exists():
            raise FileNotFoundError(
                f"YOLO results not found at {self.yolo_results_path}. "
                "Run Phase 2 (YOLO analysis) first."
            )

        yolo_results = load_json(self.yolo_results_path)
        analyses = yolo_results.get("analyses", [])

        if not analyses:
            raise ValueError("No YOLO analyses found in results file")

        self.logger.info(f"Loaded {len(analyses)} YOLO analyses")

        # Group analyses by camera
        by_camera = self._group_by_camera(analyses)
        self.logger.info(f"Found {len(by_camera)} cameras")

        # Select balanced samples for each camera
        final_selection = []
        camera_stats = {}
        roi_references = {}

        for camera_id, camera_analyses in by_camera.items():
            self.logger.info(f"Processing camera: {camera_id}")

            # Select balanced samples
            selected = self._select_balanced_for_camera(camera_id, camera_analyses)
            final_selection.extend(selected)

            # Get statistics for this camera
            stats = self._calculate_camera_stats(camera_id, camera_analyses, selected)
            camera_stats[camera_id] = stats

            # Find ROI reference image (highest traffic)
            roi_ref = self._find_roi_reference(camera_id, camera_analyses)
            roi_references[camera_id] = roi_ref

            # Log summary
            self.logger.info(
                f"  Selected {len(selected)}/{len(camera_analyses)} samples - "
                f"Empty: {stats['selected_by_level']['likely_empty']}, "
                f"Light: {stats['selected_by_level']['likely_light']}, "
                f"Moderate: {stats['selected_by_level']['likely_moderate']}, "
                f"Heavy: {stats['selected_by_level']['likely_heavy']}"
            )

        # Create output structure
        output = {
            "selection_date": datetime.now().isoformat(),
            "yolo_source": str(self.yolo_results_path),
            "target_per_camera": self.config.final_samples_per_camera,
            "target_per_level": self.config.target_per_traffic_level,
            "total_selected": len(final_selection),
            "num_cameras": len(by_camera),
            "camera_stats": camera_stats,
            "selected_samples": final_selection
        }

        # Save results
        self.logger.info(f"Saving balanced selection to {self.output_path}")
        save_json(output, self.output_path)

        self.logger.info(f"Saving ROI references to {self.roi_references_path}")
        save_json(roi_references, self.roi_references_path)

        # Log overall statistics
        self._log_overall_stats(camera_stats)

        return {
            "selection": output,
            "roi_references": roi_references,
            "resumed": False
        }

    def validate(self) -> bool:
        """
        Validate that balanced selection completed successfully.

        Returns:
            True if validation passed, False otherwise
        """
        # Check output files exist
        if not self.output_path.exists():
            self.logger.error(f"Output file not found: {self.output_path}")
            return False

        if not self.roi_references_path.exists():
            self.logger.error(f"ROI references file not found: {self.roi_references_path}")
            return False

        # Load and validate structure
        try:
            selection = load_json(self.output_path)
            roi_references = load_json(self.roi_references_path)
        except Exception as e:
            self.logger.error(f"Failed to load output files: {e}")
            return False

        # Validate expected fields
        required_fields = ["selected_samples", "camera_stats", "total_selected"]
        for field in required_fields:
            if field not in selection:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate sample count
        total_selected = len(selection["selected_samples"])
        if total_selected != selection["total_selected"]:
            self.logger.error(
                f"Sample count mismatch: {total_selected} != {selection['total_selected']}"
            )
            return False

        # Validate per-camera targets (allow some flexibility for cameras with insufficient samples)
        num_cameras = selection["num_cameras"]
        expected_total = num_cameras * self.config.final_samples_per_camera

        if total_selected < expected_total * 0.9:  # Allow 10% tolerance
            self.logger.warning(
                f"Total selected ({total_selected}) is significantly less than "
                f"expected ({expected_total}). Some cameras may have insufficient samples."
            )

        # Validate ROI references
        camera_ids = set(selection["camera_stats"].keys())
        roi_camera_ids = set(roi_references.keys())

        if camera_ids != roi_camera_ids:
            self.logger.error(
                f"Camera ID mismatch between selection and ROI references: "
                f"{camera_ids} != {roi_camera_ids}"
            )
            return False

        self.logger.info("Validation passed")
        return True

    def _group_by_camera(self, analyses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group YOLO analyses by camera ID.

        Args:
            analyses: List of YOLO analysis results

        Returns:
            Dictionary mapping camera_id to list of analyses
        """
        by_camera = defaultdict(list)

        for analysis in analyses:
            camera_id = analysis.get("camera_id")
            if not camera_id:
                self.logger.warning(f"Analysis missing camera_id: {analysis.get('local_path')}")
                continue

            by_camera[camera_id].append(analysis)

        return dict(by_camera)

    def _select_balanced_for_camera(
        self,
        camera_id: str,
        analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select balanced samples for a single camera.

        Strategy:
        1. Group analyses by traffic level
        2. Try to select target_per_level (125) from each level
        3. If a level has fewer samples, take all available
        4. Fill remaining quota from other levels if needed
        5. Prioritize heavy/moderate over light/empty for filling

        Args:
            camera_id: Camera identifier
            analyses: List of YOLO analyses for this camera

        Returns:
            List of selected analyses
        """
        # Group by traffic level
        by_level = {level: [] for level in [f"likely_{l}" for l in TRAFFIC_LEVELS]}

        for analysis in analyses:
            traffic_level = analysis.get("traffic_level")
            if traffic_level in by_level:
                by_level[traffic_level].append(analysis)
            else:
                self.logger.warning(
                    f"Unknown traffic level '{traffic_level}' for {analysis.get('local_path')}"
                )

        selected = []
        target_per_level = self.config.target_per_traffic_level
        target_total = self.config.final_samples_per_camera

        # First pass: try to get target_per_level from each level
        # Process in priority order: heavy, moderate, light, empty
        for level in ["likely_heavy", "likely_moderate", "likely_light", "likely_empty"]:
            available = by_level[level]
            target = min(target_per_level, len(available))

            if len(available) <= target:
                # Take all available
                selected_from_level = available
            else:
                # Random sample
                selected_from_level = random.sample(available, target)

            selected.extend(selected_from_level)

            # Mark as used
            by_level[level] = [a for a in available if a not in selected_from_level]

        # Second pass: fill remaining quota if needed
        remaining_quota = target_total - len(selected)

        if remaining_quota > 0:
            # Collect all remaining samples
            all_remaining = []
            for level in ["likely_heavy", "likely_moderate", "likely_light", "likely_empty"]:
                all_remaining.extend(by_level[level])

            if all_remaining:
                # Take up to remaining_quota
                additional = random.sample(
                    all_remaining,
                    min(remaining_quota, len(all_remaining))
                )
                selected.extend(additional)

        # Ensure we don't exceed target
        if len(selected) > target_total:
            selected = random.sample(selected, target_total)

        return selected

    def _calculate_camera_stats(
        self,
        camera_id: str,
        all_analyses: List[Dict[str, Any]],
        selected: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for camera selection.

        Args:
            camera_id: Camera identifier
            all_analyses: All available analyses for this camera
            selected: Selected analyses for this camera

        Returns:
            Statistics dictionary
        """
        # Count by level in available pool
        available_by_level = defaultdict(int)
        for analysis in all_analyses:
            level = analysis.get("traffic_level", "unknown")
            available_by_level[level] += 1

        # Count by level in selection
        selected_by_level = defaultdict(int)
        for analysis in selected:
            level = analysis.get("traffic_level", "unknown")
            selected_by_level[level] += 1

        return {
            "camera_id": camera_id,
            "total_available": len(all_analyses),
            "total_selected": len(selected),
            "available_by_level": dict(available_by_level),
            "selected_by_level": dict(selected_by_level),
            "selection_rate": len(selected) / len(all_analyses) if all_analyses else 0.0
        }

    def _find_roi_reference(
        self,
        camera_id: str,
        analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find the best reference image for ROI definition.

        Selects the image with the highest vehicle count, as this is most likely
        to show the full range of the camera view including distant vehicles
        where YOLO may fail.

        Args:
            camera_id: Camera identifier
            analyses: List of YOLO analyses for this camera

        Returns:
            Reference image information
        """
        if not analyses:
            return {
                "camera_id": camera_id,
                "local_path": None,
                "vehicle_count": 0,
                "traffic_level": None,
                "boxes": []
            }

        # Sort by vehicle count (descending)
        sorted_analyses = sorted(
            analyses,
            key=lambda x: x.get("vehicle_count", 0),
            reverse=True
        )

        best = sorted_analyses[0]

        return {
            "camera_id": camera_id,
            "local_path": best.get("local_path"),
            "vehicle_count": best.get("vehicle_count", 0),
            "traffic_level": best.get("traffic_level"),
            "boxes": best.get("boxes", [])
        }

    def _log_overall_stats(self, camera_stats: Dict[str, Dict[str, Any]]):
        """
        Log overall selection statistics.

        Args:
            camera_stats: Statistics for all cameras
        """
        total_available = sum(stats["total_available"] for stats in camera_stats.values())
        total_selected = sum(stats["total_selected"] for stats in camera_stats.values())

        # Aggregate by level
        total_by_level = defaultdict(int)
        for stats in camera_stats.values():
            for level, count in stats["selected_by_level"].items():
                total_by_level[level] += count

        self.logger.info("=" * 60)
        self.logger.info("BALANCED SELECTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total cameras: {len(camera_stats)}")
        self.logger.info(f"Total available: {total_available}")
        self.logger.info(f"Total selected: {total_selected}")
        self.logger.info(f"Overall selection rate: {total_selected/total_available:.1%}")
        self.logger.info("")
        self.logger.info("Selected by traffic level:")
        for level in ["likely_empty", "likely_light", "likely_moderate", "likely_heavy"]:
            count = total_by_level.get(level, 0)
            target = len(camera_stats) * self.config.target_per_traffic_level
            self.logger.info(f"  {level:20s}: {count:5d} / {target:5d} ({count/target:.1%})")
        self.logger.info("=" * 60)


def main():
    """Command-line entry point for Phase 3."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3: Traffic-Balanced Selection"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip if already completed"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation without executing phase"
    )
    args = parser.parse_args()

    # Initialize phase
    config = PipelineConfig()
    phase = BalancedSelectionPhase(config)

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
            selection = data["selection"]
            print(f"\nSelected {selection['total_selected']} images across {selection['num_cameras']} cameras")
            print(f"Target: {selection['target_per_camera']} per camera, {selection['target_per_level']} per traffic level")
            exit(0)
        else:
            print(f"✗ Phase failed: {result.get('reason', 'Unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
