"""
YOLO Results Verification Script

Analyzes YOLO detection results to verify data quality before proceeding to Phase 3.
Generates distribution statistics, visualizations, and quality reports.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from .config import PipelineConfig, TRAFFIC_LEVELS, TRAFFIC_THRESHOLDS
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class YOLOVerification:
    """
    Verification and analysis of YOLO detection results.

    Provides distribution analysis, visualization, and quality reporting
    to ensure YOLO results are suitable for balanced sampling.
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize YOLO verification.

        Args:
            pipeline_config: Pipeline configuration
        """
        self.config = pipeline_config
        self.results_file = self.config.get_path(self.config.yolo_results_file)
        self.output_dir = self.config.base_dir / "yolo_verification"

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

        self.yolo_results = None
        self.analyses = []

    def load_results(self) -> Dict[str, Any]:
        """
        Load YOLO results from file.

        Returns:
            YOLO results dictionary

        Raises:
            FileNotFoundError: If results file doesn't exist
            ValueError: If results are empty or invalid
        """
        if not self.results_file.exists():
            raise FileNotFoundError(
                f"YOLO results file not found: {self.results_file}. "
                "Run Phase 2 (YOLO analysis) first."
            )

        self.yolo_results = load_json(self.results_file)

        if not self.yolo_results or not self.yolo_results.get("analyses"):
            raise ValueError("YOLO results file is empty or has no analyses")

        self.analyses = self.yolo_results["analyses"]

        logger.info(f"Loaded {len(self.analyses)} YOLO analyses")
        return self.yolo_results

    def analyze_distribution(self) -> Dict[str, Any]:
        """
        Analyze traffic level distribution across all images and per camera.

        Returns:
            Dictionary with distribution statistics
        """
        logger.info("Analyzing distribution...")

        # Overall distribution
        overall_counts = Counter(a["traffic_level"] for a in self.analyses)
        total = len(self.analyses)

        # Per-camera distribution
        by_camera = defaultdict(lambda: defaultdict(int))
        for analysis in self.analyses:
            camera = analysis["camera_id"]
            level = analysis["traffic_level"]
            by_camera[camera][level] += 1

        # Vehicle count statistics
        vehicle_counts = [a["vehicle_count"] for a in self.analyses]

        statistics = {
            "total_images": total,
            "by_traffic_level": dict(overall_counts),
            "by_traffic_level_pct": {
                level: (count / total * 100) for level, count in overall_counts.items()
            },
            "by_camera": {
                camera: dict(levels) for camera, levels in by_camera.items()
            },
            "vehicle_count_stats": {
                "mean": float(np.mean(vehicle_counts)),
                "median": float(np.median(vehicle_counts)),
                "std": float(np.std(vehicle_counts)),
                "min": int(np.min(vehicle_counts)),
                "max": int(np.max(vehicle_counts)),
                "percentile_25": float(np.percentile(vehicle_counts, 25)),
                "percentile_75": float(np.percentile(vehicle_counts, 75)),
            }
        }

        logger.info(f"Distribution: {dict(overall_counts)}")
        logger.info(f"Vehicle stats: mean={statistics['vehicle_count_stats']['mean']:.1f}, "
                   f"median={statistics['vehicle_count_stats']['median']:.0f}")

        return statistics

    def detect_issues(self, statistics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect potential issues in YOLO results.

        Args:
            statistics: Distribution statistics from analyze_distribution()

        Returns:
            List of issue dictionaries
        """
        logger.info("Detecting issues...")

        issues = []
        target_per_level = self.config.final_samples_per_camera // 4  # 125

        # Check per-camera distribution for insufficient samples
        for camera, levels in statistics["by_camera"].items():
            for level in TRAFFIC_LEVELS:
                level_key = f"likely_{level}"
                count = levels.get(level_key, 0)

                if count < target_per_level:
                    issues.append({
                        "type": "insufficient_samples",
                        "severity": "warning" if count >= target_per_level * 0.5 else "critical",
                        "camera": camera,
                        "traffic_level": level_key,
                        "count": count,
                        "needed": target_per_level,
                        "message": f"{camera} has only {count} {level_key} images (need {target_per_level})"
                    })

        # Check for outliers (very high vehicle counts)
        max_reasonable = 50  # More than 50 vehicles is likely a false positive
        for analysis in self.analyses:
            if analysis["vehicle_count"] > max_reasonable:
                issues.append({
                    "type": "outlier",
                    "severity": "warning",
                    "camera": analysis["camera_id"],
                    "local_path": analysis["local_path"],
                    "vehicle_count": analysis["vehicle_count"],
                    "message": f"Image has {analysis['vehicle_count']} detections (likely error)"
                })

        # Check for cameras with very imbalanced distribution
        for camera, levels in statistics["by_camera"].items():
            total_camera = sum(levels.values())
            for level in TRAFFIC_LEVELS:
                level_key = f"likely_{level}"
                count = levels.get(level_key, 0)
                pct = (count / total_camera * 100) if total_camera > 0 else 0

                if pct > 60:  # One level dominates
                    issues.append({
                        "type": "imbalanced_distribution",
                        "severity": "info",
                        "camera": camera,
                        "traffic_level": level_key,
                        "percentage": pct,
                        "message": f"{camera} is {pct:.1f}% {level_key}"
                    })

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        issues.sort(key=lambda x: severity_order.get(x["severity"], 3))

        logger.info(f"Found {len(issues)} potential issues")
        return issues

    def create_distribution_histogram(
        self,
        statistics: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create histogram of vehicle counts.

        Args:
            statistics: Distribution statistics
            output_path: Optional custom output path

        Returns:
            Path to saved histogram image
        """
        if output_path is None:
            output_path = self.output_dir / "distribution_histogram.png"

        logger.info("Creating distribution histogram...")

        vehicle_counts = [a["vehicle_count"] for a in self.analyses]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Histogram
        ax.hist(vehicle_counts, bins=50, edgecolor='black', alpha=0.7)

        # Add vertical lines for threshold boundaries
        thresholds = [2.5, 6.5, 15.5]  # Boundaries between categories
        colors = ['red', 'orange', 'yellow']
        labels = ['Empty|Light', 'Light|Moderate', 'Moderate|Heavy']

        for threshold, color, label in zip(thresholds, colors, labels):
            ax.axvline(threshold, color=color, linestyle='--', linewidth=2, label=label)

        ax.set_xlabel('Vehicle Count', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Distribution of Vehicle Counts Across All Images', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = (
            f"Mean: {statistics['vehicle_count_stats']['mean']:.1f}\n"
            f"Median: {statistics['vehicle_count_stats']['median']:.0f}\n"
            f"Std Dev: {statistics['vehicle_count_stats']['std']:.1f}\n"
            f"Max: {statistics['vehicle_count_stats']['max']}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Histogram saved to {output_path}")
        return output_path

    def create_traffic_level_chart(
        self,
        statistics: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create bar chart of traffic levels by camera.

        Args:
            statistics: Distribution statistics
            output_path: Optional custom output path

        Returns:
            Path to saved chart image
        """
        if output_path is None:
            output_path = self.output_dir / "traffic_levels_by_camera.png"

        logger.info("Creating traffic level chart...")

        cameras = sorted(statistics["by_camera"].keys())

        # Prepare data for stacked bar chart
        data = {level: [] for level in TRAFFIC_LEVELS}

        for camera in cameras:
            levels = statistics["by_camera"][camera]
            for level in TRAFFIC_LEVELS:
                level_key = f"likely_{level}"
                data[level].append(levels.get(level_key, 0))

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(16, 8))

        x = np.arange(len(cameras))
        width = 0.8

        colors = ['#90EE90', '#FFD700', '#FFA500', '#FF6347']  # Light green, gold, orange, tomato
        bottoms = np.zeros(len(cameras))

        for level, color in zip(TRAFFIC_LEVELS, colors):
            values = data[level]
            ax.bar(x, values, width, label=level.capitalize(),
                   bottom=bottoms, color=color, edgecolor='black')
            bottoms += values

        # Add target line
        target_total = self.config.initial_samples_per_camera  # 700
        ax.axhline(target_total, color='red', linestyle='--', linewidth=2,
                   label=f'Target ({target_total})', alpha=0.7)

        ax.set_xlabel('Camera', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Traffic Level Distribution by Camera', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cameras, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Traffic level chart saved to {output_path}")
        return output_path

    def create_sample_visualizations(
        self,
        samples_per_level: int = 6,
        output_prefix: Optional[str] = None
    ) -> List[Path]:
        """
        Create sample images with YOLO bounding boxes for each traffic level.

        Args:
            samples_per_level: Number of sample images per traffic level
            output_prefix: Optional prefix for output filenames

        Returns:
            List of paths to saved visualization images
        """
        logger.info("Creating sample visualizations...")

        output_paths = []

        for level in TRAFFIC_LEVELS:
            level_key = f"likely_{level}"

            # Find samples for this level
            level_analyses = [a for a in self.analyses if a["traffic_level"] == level_key]

            if not level_analyses:
                logger.warning(f"No samples found for {level_key}")
                continue

            # Select random samples
            samples = random.sample(level_analyses, min(samples_per_level, len(level_analyses)))

            # Create grid of images
            output_path = self.output_dir / f"{output_prefix or 'sample_detections'}_{level}.jpg"
            self._create_sample_grid(samples, output_path, level_key)
            output_paths.append(output_path)

        logger.info(f"Created {len(output_paths)} sample visualization grids")
        return output_paths

    def _create_sample_grid(
        self,
        samples: List[Dict[str, Any]],
        output_path: Path,
        level_name: str
    ):
        """
        Create a grid of sample images with bounding boxes.

        Args:
            samples: List of analysis dictionaries
            output_path: Path to save the grid image
            level_name: Traffic level name for title
        """
        # Grid dimensions
        cols = 3
        rows = (len(samples) + cols - 1) // cols

        # Image size
        img_width = 400
        img_height = 300

        # Create canvas
        canvas_width = cols * img_width
        canvas_height = rows * img_height + 60  # Extra space for title
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)

        # Draw title
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except:
            font = ImageFont.load_default()

        title = f"Sample Detections - {level_name.replace('likely_', '').upper()}"
        # Get text bbox for centering
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(((canvas_width - text_width) // 2, 10), title, fill='black', font=font)

        # Process each sample
        for idx, sample in enumerate(samples):
            row = idx // cols
            col = idx % cols

            # Load image
            image_path = self.config.base_dir / sample["local_path"]

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            try:
                img = Image.open(image_path)

                # Resize to fit grid
                img.thumbnail((img_width, img_height))

                # Draw bounding boxes
                img_draw = ImageDraw.Draw(img)
                for box in sample.get("boxes", []):
                    xyxy = box["xyxy"]
                    # Scale boxes if image was resized
                    scale_x = img.width / Image.open(image_path).width
                    scale_y = img.height / Image.open(image_path).height

                    x1 = int(xyxy[0] * scale_x)
                    y1 = int(xyxy[1] * scale_y)
                    x2 = int(xyxy[2] * scale_x)
                    y2 = int(xyxy[3] * scale_y)

                    img_draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

                # Add text label
                label = f"Vehicles: {sample['vehicle_count']}"
                img_draw.text((10, 10), label, fill='yellow', font=None)
                img_draw.rectangle([5, 5, img.width - 5, img.height - 5], outline='black', width=2)

                # Paste onto canvas
                x_offset = col * img_width
                y_offset = row * img_height + 60
                canvas.paste(img, (x_offset, y_offset))

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

        # Save canvas
        canvas.save(output_path, quality=95)
        logger.info(f"Saved sample grid to {output_path}")

    def generate_text_report(
        self,
        statistics: Dict[str, Any],
        issues: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate human-readable text report.

        Args:
            statistics: Distribution statistics
            issues: List of detected issues
            output_path: Optional custom output path

        Returns:
            Path to saved report file
        """
        if output_path is None:
            output_path = self.output_dir / "yolo_verification_report.txt"

        logger.info("Generating text report...")

        lines = []
        lines.append("=" * 80)
        lines.append("YOLO RESULTS VERIFICATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total images analyzed: {statistics['total_images']:,}")
        lines.append(f"Model: {self.yolo_results.get('model', 'N/A')}")
        lines.append(f"Device: {self.yolo_results.get('device', 'N/A')}")
        lines.append(f"Confidence threshold: {self.yolo_results.get('confidence_threshold', 'N/A')}")
        lines.append("")

        # Traffic level distribution
        lines.append("TRAFFIC LEVEL DISTRIBUTION")
        lines.append("-" * 80)
        for level in TRAFFIC_LEVELS:
            level_key = f"likely_{level}"
            count = statistics["by_traffic_level"].get(level_key, 0)
            pct = statistics["by_traffic_level_pct"].get(level_key, 0)
            lines.append(f"  {level_key:20s}: {count:6,} ({pct:5.1f}%)")
        lines.append("")

        # Vehicle count statistics
        lines.append("VEHICLE COUNT STATISTICS")
        lines.append("-" * 80)
        stats = statistics["vehicle_count_stats"]
        lines.append(f"  Mean:          {stats['mean']:6.1f}")
        lines.append(f"  Median:        {stats['median']:6.0f}")
        lines.append(f"  Std Dev:       {stats['std']:6.1f}")
        lines.append(f"  Min:           {stats['min']:6}")
        lines.append(f"  Max:           {stats['max']:6}")
        lines.append(f"  25th percentile: {stats['percentile_25']:6.0f}")
        lines.append(f"  75th percentile: {stats['percentile_75']:6.0f}")
        lines.append("")

        # Per-camera summary (top 10)
        lines.append("PER-CAMERA SUMMARY (Top 10 by total count)")
        lines.append("-" * 80)
        lines.append(f"{'Camera':<15} {'Empty':>8} {'Light':>8} {'Moderate':>10} {'Heavy':>8} {'Total':>8}")
        lines.append("-" * 80)

        camera_totals = {
            cam: sum(levels.values())
            for cam, levels in statistics["by_camera"].items()
        }
        top_cameras = sorted(camera_totals.items(), key=lambda x: x[1], reverse=True)[:10]

        for camera, total in top_cameras:
            levels = statistics["by_camera"][camera]
            lines.append(
                f"{camera:<15} "
                f"{levels.get('likely_empty', 0):8} "
                f"{levels.get('likely_light', 0):8} "
                f"{levels.get('likely_moderate', 0):10} "
                f"{levels.get('likely_heavy', 0):8} "
                f"{total:8}"
            )
        lines.append("")

        # Issues
        lines.append("POTENTIAL ISSUES")
        lines.append("-" * 80)

        if not issues:
            lines.append("  No issues detected!")
        else:
            # Group by severity
            by_severity = defaultdict(list)
            for issue in issues:
                by_severity[issue["severity"]].append(issue)

            for severity in ["critical", "warning", "info"]:
                if severity not in by_severity:
                    continue

                lines.append(f"\n{severity.upper()} ({len(by_severity[severity])} issues):")
                lines.append("-" * 80)

                for issue in by_severity[severity][:20]:  # Limit to 20 per severity
                    lines.append(f"  - {issue['message']}")

                if len(by_severity[severity]) > 20:
                    lines.append(f"  ... and {len(by_severity[severity]) - 20} more")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Text report saved to {output_path}")
        return output_path

    def generate_json_report(
        self,
        statistics: Dict[str, Any],
        issues: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate machine-readable JSON report.

        Args:
            statistics: Distribution statistics
            issues: List of detected issues
            output_path: Optional custom output path

        Returns:
            Path to saved JSON file
        """
        if output_path is None:
            output_path = self.output_dir / "yolo_verification.json"

        logger.info("Generating JSON report...")

        report = {
            "yolo_config": {
                "model": self.yolo_results.get("model"),
                "device": self.yolo_results.get("device"),
                "confidence_threshold": self.yolo_results.get("confidence_threshold"),
            },
            "statistics": statistics,
            "issues": issues,
            "summary": {
                "total_issues": len(issues),
                "critical_issues": sum(1 for i in issues if i["severity"] == "critical"),
                "warning_issues": sum(1 for i in issues if i["severity"] == "warning"),
                "info_issues": sum(1 for i in issues if i["severity"] == "info"),
            }
        }

        save_json(report, output_path)
        logger.info(f"JSON report saved to {output_path}")
        return output_path

    def run_full_verification(self) -> Dict[str, Any]:
        """
        Run complete verification workflow.

        Returns:
            Dictionary with all verification results and output paths
        """
        logger.info("Starting full YOLO verification...")

        # Load results
        self.load_results()

        # Analyze distribution
        statistics = self.analyze_distribution()

        # Detect issues
        issues = self.detect_issues(statistics)

        # Create visualizations
        histogram_path = self.create_distribution_histogram(statistics)
        chart_path = self.create_traffic_level_chart(statistics)
        sample_paths = self.create_sample_visualizations(samples_per_level=6)

        # Generate reports
        text_report_path = self.generate_text_report(statistics, issues)
        json_report_path = self.generate_json_report(statistics, issues)

        logger.info("Verification complete!")
        logger.info(f"Output directory: {self.output_dir}")

        return {
            "statistics": statistics,
            "issues": issues,
            "outputs": {
                "histogram": str(histogram_path),
                "chart": str(chart_path),
                "samples": [str(p) for p in sample_paths],
                "text_report": str(text_report_path),
                "json_report": str(json_report_path),
            }
        }


def main():
    """
    CLI entry point for YOLO verification.

    Usage:
        python -m dataset_pipeline.verify_yolo
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify YOLO analysis results with distribution and visualization"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of sample images per traffic level (default: 6)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for verification results"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    pipeline_config = PipelineConfig()

    # Create verifier
    verifier = YOLOVerification(pipeline_config)

    if args.output_dir:
        verifier.output_dir = Path(args.output_dir)
        verifier.output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Run verification
        results = verifier.run_full_verification()

        # Print summary
        print("\n" + "=" * 80)
        print("YOLO VERIFICATION COMPLETE")
        print("=" * 80)
        print(f"\nTotal images analyzed: {results['statistics']['total_images']:,}")
        print(f"\nTraffic level distribution:")
        for level in TRAFFIC_LEVELS:
            level_key = f"likely_{level}"
            count = results['statistics']['by_traffic_level'].get(level_key, 0)
            pct = results['statistics']['by_traffic_level_pct'].get(level_key, 0)
            print(f"  {level_key:20s}: {count:6,} ({pct:5.1f}%)")

        print(f"\nIssues found: {len(results['issues'])}")
        if results['issues']:
            critical = sum(1 for i in results['issues'] if i['severity'] == 'critical')
            warning = sum(1 for i in results['issues'] if i['severity'] == 'warning')
            info = sum(1 for i in results['issues'] if i['severity'] == 'info')
            print(f"  Critical: {critical}")
            print(f"  Warning: {warning}")
            print(f"  Info: {info}")

        print(f"\nOutput directory: {verifier.output_dir}")
        print(f"  - Histogram: {Path(results['outputs']['histogram']).name}")
        print(f"  - Chart: {Path(results['outputs']['chart']).name}")
        print(f"  - Sample images: {len(results['outputs']['samples'])} files")
        print(f"  - Text report: {Path(results['outputs']['text_report']).name}")
        print(f"  - JSON report: {Path(results['outputs']['json_report']).name}")

        print("\n" + "=" * 80)

        # Return exit code based on critical issues
        critical_count = sum(1 for i in results['issues'] if i['severity'] == 'critical')
        if critical_count > 0:
            print(f"⚠️  WARNING: {critical_count} critical issues found!")
            print("Review the report before proceeding to Phase 3.")
            exit(1)
        else:
            print("✓ No critical issues found. Ready to proceed to Phase 3.")
            exit(0)

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        print(f"\n✗ Verification failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
