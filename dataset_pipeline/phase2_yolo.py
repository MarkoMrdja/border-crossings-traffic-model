"""
Phase 2: YOLO Analysis

Runs YOLO vehicle detection on all downloaded images to count vehicles
and assign preliminary traffic density labels.
"""

import logging
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

    def analyze_image(
        self,
        image_path: Path,
        verbose: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Run YOLO inference on a single image.

        Args:
            image_path: Path to image file
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

            # Count vehicles
            vehicle_count = len(boxes)

            # Extract bounding box details
            box_data = []
            for box in boxes:
                box_data.append({
                    "xyxy": box.xyxy[0].cpu().tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0].cpu()),
                    "class": int(box.cls[0].cpu())
                })

            # Categorize traffic level
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

                # Skip if already analyzed (resume mode)
                if resume and local_path in analyzed_paths:
                    stats["skipped"] += 1
                    continue

                # Construct full image path
                image_path = self.config.base_dir / local_path

                # Analyze image
                analysis_result = self.analyze_image(image_path, verbose=False)

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
