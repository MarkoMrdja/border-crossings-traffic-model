#!/usr/bin/env python3
"""
Border Crossings Traffic Dataset Pipeline - Unified CLI Orchestrator

This script provides a unified command-line interface to orchestrate all phases
of the dataset pipeline, including status tracking, validation, and reset capabilities.
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Import pipeline components
from dataset_pipeline.config import PipelineConfig, AzureConfig
from dataset_pipeline import (
    StructureDiscoveryPhase,
    StratifiedSamplingPhase,
    ParallelDownloadPhase,
    YOLOAnalysisPhase,
    LaneAnnotationTool,
    SelectionPhase,
    ReviewTool,
    ExclusionZonesTool,
    CropPhase,
    TrainValSplitPhase,
)


# ============================================================================
# PHASE METADATA
# ============================================================================

@dataclass
class PhaseMetadata:
    """Metadata for a single pipeline phase"""
    number: str              # "1a", "1b", "2", etc.
    name: str                # "Structure Discovery"
    description: str         # Brief description
    class_name: str          # "StructureDiscoveryPhase"
    output_files: List[str]  # Required output files (relative to base_dir)
    dependencies: List[str]  # Phase numbers this depends on
    is_interactive: bool     # True for phases 2a, 4, 5 (lane annotation, review, exclusion zones)
    estimated_time: str      # "5-15 minutes"
    requires_azure: bool     # True for phases 1a, 1c
    init_kwargs: List[str]   # Constructor parameter names
    runtime_kwargs: List[str] = field(default_factory=list)  # Optional runtime args


# Registry of all pipeline phases
PHASE_REGISTRY = {
    "1a": PhaseMetadata(
        number="1a",
        name="Structure Discovery",
        description="Discover Azure Blob Storage structure",
        class_name="StructureDiscoveryPhase",
        output_files=["inventory.json"],
        dependencies=[],
        is_interactive=False,
        estimated_time="5-15 minutes",
        requires_azure=True,
        init_kwargs=["pipeline_config", "azure_config"]
    ),
    "1b": PhaseMetadata(
        number="1b",
        name="Stratified Sampling",
        description="Select 700 images per camera",
        class_name="StratifiedSamplingPhase",
        output_files=["sample_manifest.json"],
        dependencies=["1a"],
        is_interactive=False,
        estimated_time="10-30 minutes",
        requires_azure=True,
        init_kwargs=["pipeline_config", "azure_config"]
    ),
    "1c": PhaseMetadata(
        number="1c",
        name="Parallel Download",
        description="Download selected images from Azure",
        class_name="ParallelDownloadPhase",
        output_files=["download_progress.json"],
        dependencies=["1b"],
        is_interactive=False,
        estimated_time="1.5-3 hours",
        requires_azure=True,
        init_kwargs=["pipeline_config", "azure_config"],
        runtime_kwargs=["workers"]
    ),
    "2": PhaseMetadata(
        number="2",
        name="YOLO Analysis",
        description="Run YOLO with polygon filtering to exclude parked cars outside lanes",
        class_name="YOLOAnalysisPhase",
        output_files=["yolo_results.json"],
        dependencies=["1c"],
        is_interactive=False,
        estimated_time="~75 minutes with MPS",
        requires_azure=False,
        init_kwargs=["pipeline_config"],
        runtime_kwargs=["device", "model", "confidence"]
    ),
    "2a": PhaseMetadata(
        number="2a",
        name="Lane Annotation",
        description="Interactive lane polygon annotation (skip if lane_polygons.json exists)",
        class_name="LaneAnnotationTool",
        output_files=["lane_polygons.json"],
        dependencies=["2"],
        is_interactive=True,
        estimated_time="2-3 hours (~3-5 min per camera)",
        requires_azure=False,
        init_kwargs=["pipeline_config"],
        runtime_kwargs=["resume"]
    ),
    "3": PhaseMetadata(
        number="3",
        name="Binary Selection",
        description="Select 6K balanced images (3K traffic_present, 3K traffic_absent)",
        class_name="SelectionPhase",
        output_files=["binary_selection.json"],
        dependencies=["2a"],
        is_interactive=False,
        estimated_time="1-2 minutes",
        requires_azure=False,
        init_kwargs=["pipeline_config"],
        runtime_kwargs=["target_per_class"]
    ),
    "4": PhaseMetadata(
        number="4",
        name="Label Review",
        description="Manual review and correction of binary labels",
        class_name="ReviewTool",
        output_files=["binary_labeled/", "binary_review_log.json"],
        dependencies=["3"],
        is_interactive=True,
        estimated_time="4-6 hours (all) or 1 hour (borderline only)",
        requires_azure=False,
        init_kwargs=["pipeline_config"],
        runtime_kwargs=["review_all", "resume"]
    ),
    "5": PhaseMetadata(
        number="5",
        name="Exclusion Zones",
        description="Mark YOLO failure regions (one polygon per camera)",
        class_name="ExclusionZonesTool",
        output_files=["yolo_failure_regions.json"],
        dependencies=["4"],
        is_interactive=True,
        estimated_time="2-3 hours (~3-5 min per camera)",
        requires_azure=False,
        init_kwargs=["pipeline_config"],
        runtime_kwargs=["resume"]
    ),
    "6": PhaseMetadata(
        number="6",
        name="Crop Failure Regions",
        description="Crop regions with label inheritance (128×128)",
        class_name="CropPhase",
        output_files=["binary_crops/", "binary_crop_metadata.json"],
        dependencies=["5"],
        is_interactive=False,
        estimated_time="15-20 minutes",
        requires_azure=False,
        init_kwargs=["pipeline_config"],
        runtime_kwargs=["resume"]
    ),
    "7": PhaseMetadata(
        number="7",
        name="Train/Val Split",
        description="Create 80/20 stratified train/val split",
        class_name="TrainValSplitPhase",
        output_files=["binary_split_summary.json", "binary_final/train/", "binary_final/val/"],
        dependencies=["6"],
        is_interactive=False,
        estimated_time="1-2 minutes",
        requires_azure=False,
        init_kwargs=["pipeline_config"],
        runtime_kwargs=["train_ratio"]
    )
}

# Ordered list of phases for sequential execution
# Binary classification workflow: 1a -> 1b -> 1c -> 2 -> 2a -> 3 -> 4 -> 5 -> 6 -> 7
# Phase 2a (lane annotation) is optional if lane_polygons.json already exists
PHASE_ORDER = ["1a", "1b", "1c", "2", "2a", "3", "4", "5", "6", "7"]


# ============================================================================
# PHASE ORCHESTRATOR
# ============================================================================

class PhaseOrchestrator:
    """Orchestrates pipeline execution with dependency management"""

    def __init__(self, config: PipelineConfig, azure_config: Optional[AzureConfig]):
        self.config = config
        self.azure_config = azure_config
        self.phases = PHASE_REGISTRY

    def _instantiate_phase(self, phase_num: str, **runtime_kwargs):
        """Dynamically instantiate phase class"""
        metadata = self.phases[phase_num]

        # Get class
        phase_class = globals()[metadata.class_name]

        # Build init kwargs
        init_kwargs = {}
        if "pipeline_config" in metadata.init_kwargs:
            init_kwargs["pipeline_config"] = self.config
        if "azure_config" in metadata.init_kwargs:
            if not self.azure_config:
                raise ValueError(
                    f"Phase {phase_num} requires Azure credentials. "
                    "Please configure .env file with:\n"
                    "  AZURE_CLIENT_ID=...\n"
                    "  AZURE_TENANT_ID=...\n"
                    "  AZURE_CLIENT_SECRET=...\n"
                    "  AZURE_STORAGE_URL=..."
                )
            init_kwargs["azure_config"] = self.azure_config

        # Instantiate phase
        phase = phase_class(**init_kwargs)

        return phase

    def get_phase_status(self, phase_num: str) -> dict:
        """
        Get status of a specific phase

        Returns: {
            "complete": bool,
            "partial": bool,
            "validated": bool,
            "output_files": [...],
            "missing_files": [...]
        }
        """
        metadata = self.phases[phase_num]
        output_files = []
        missing_files = []

        # Check each output file
        for file_path in metadata.output_files:
            full_path = self.config.base_dir / file_path
            if full_path.exists():
                output_files.append(file_path)
            else:
                missing_files.append(file_path)

        # Determine status
        complete = len(missing_files) == 0
        partial = len(output_files) > 0 and not complete

        # Try to validate if complete
        validated = False
        if complete:
            try:
                phase = self._instantiate_phase(phase_num)
                validated = phase.validate()
            except Exception:
                validated = False

        return {
            "complete": complete,
            "partial": partial,
            "validated": validated,
            "output_files": output_files,
            "missing_files": missing_files
        }

    def get_all_status(self) -> dict:
        """Get status of all phases"""
        status = {}
        for phase_num in PHASE_ORDER:
            status[phase_num] = self.get_phase_status(phase_num)

        # Calculate summary
        complete_count = sum(1 for s in status.values() if s["complete"])
        partial_count = sum(1 for s in status.values() if s["partial"])
        not_started_count = len(PHASE_ORDER) - complete_count - partial_count

        # Find next runnable phase
        next_runnable = None
        for phase_num in PHASE_ORDER:
            if not status[phase_num]["complete"]:
                can_run, _ = self.can_run_phase(phase_num)
                if can_run:
                    next_runnable = phase_num
                    break

        return {
            "phases": status,
            "summary": {
                "total": len(PHASE_ORDER),
                "complete": complete_count,
                "partial": partial_count,
                "not_started": not_started_count,
                "next_runnable": next_runnable
            }
        }

    def validate_phase(self, phase_num: str) -> Tuple[bool, str]:
        """Validate phase output using phase.validate()"""
        status = self.get_phase_status(phase_num)

        if not status["complete"]:
            return False, "Phase not complete"

        try:
            phase = self._instantiate_phase(phase_num)
            result = phase.validate()
            if result:
                return True, "Validation passed"
            else:
                return False, "Validation failed"
        except Exception as e:
            return False, f"Validation error: {e}"

    def can_run_phase(self, phase_num: str) -> Tuple[bool, str]:
        """Check if phase dependencies are satisfied"""
        metadata = self.phases[phase_num]

        # Check dependencies
        for dep_phase in metadata.dependencies:
            dep_status = self.get_phase_status(dep_phase)
            if not dep_status["complete"]:
                return False, f"Dependency not satisfied: Phase {dep_phase} ({self.phases[dep_phase].name})"

        # Check Azure credentials if required
        if metadata.requires_azure and not self.azure_config:
            return False, "Azure credentials not configured"

        return True, "All dependencies satisfied"

    def run_phase(self, phase_num: str, resume: bool = False, dry_run: bool = False, **kwargs) -> dict:
        """Execute a single phase with error handling"""
        metadata = self.phases[phase_num]

        # Check if can run
        can_run, reason = self.can_run_phase(phase_num)
        if not can_run:
            return {
                "phase": phase_num,
                "status": "blocked",
                "reason": reason
            }

        # Check if already complete and resume requested
        if resume:
            status = self.get_phase_status(phase_num)
            if status["complete"] and status["validated"]:
                return {
                    "phase": phase_num,
                    "status": "skipped",
                    "reason": "Already complete and validated"
                }

        # Dry run - just report what would happen
        if dry_run:
            return {
                "phase": phase_num,
                "status": "dry_run",
                "would_execute": metadata.description,
                "estimated_time": metadata.estimated_time
            }

        # Handle interactive phases
        if metadata.is_interactive:
            print(f"\n{'='*60}")
            print(f"Phase {phase_num}: {metadata.name}")
            print(f"{'='*60}\n")
            print("⚠️  This phase is INTERACTIVE and requires human input\n")
            print(f"What it does:")
            print(f"  {metadata.description}")
            print(f"  Estimated time: {metadata.estimated_time}\n")

            if phase_num == "4":
                print("Controls:")
                print("  - Left-click: Add polygon vertex")
                print("  - Right-click: Remove last vertex")
                print("  - N: Save and move to next camera")
                print("  - R: Reset current polygon")
                print("  - S: Skip camera")
                print("  - Q: Save progress and quit\n")
            elif phase_num == "4-binary":
                print("Controls:")
                print("  - 1: Label as traffic_present (moderate/heavy congestion)")
                print("  - 2: Label as traffic_absent (empty/light traffic)")
                print("  - Enter: Accept current label (if confident)")
                print("  - U: Mark as uncertain (exclude from dataset)")
                print("  - N: Next image")
                print("  - P: Previous image")
                print("  - Q: Quit and save progress\n")
                print("Review strategy:")
                print("  - Use --borderline-only to review only borderline cases (~1,300 images, 1 hour)")
                print("  - Or review all 6,000 images for highest quality (4-6 hours)\n")
            elif phase_num == "6":
                print("Controls:")
                print("  - Enter: Confirm predicted label")
                print("  - 1-4: Assign specific level (1=empty, 2=light, 3=moderate, 4=heavy)")
                print("  - S: Skip image")
                print("  - B: Go back (undo last label)")
                print("  - Q: Save progress and quit\n")

            print(f"Progress will be saved. Resume with:")
            print(f"  python main.py run --phase {phase_num} --resume\n")

            response = input("Ready to start? [y/N]: ")
            if response.lower() != 'y':
                return {
                    "phase": phase_num,
                    "status": "cancelled",
                    "reason": "User declined to start interactive phase"
                }

        # Instantiate and execute phase
        try:
            phase = self._instantiate_phase(phase_num, **kwargs)
            result = phase.execute(resume=resume, validate_after=True)
            return result
        except Exception as e:
            return {
                "phase": phase_num,
                "status": "failed",
                "reason": str(e)
            }

    def run_pipeline(self, from_phase: str = "1a", to_phase: str = "7",
                     resume: bool = True, dry_run: bool = False, **kwargs) -> List[dict]:
        """Execute multiple phases sequentially"""
        # Get phase range
        start_idx = PHASE_ORDER.index(from_phase)
        end_idx = PHASE_ORDER.index(to_phase)
        phase_range = PHASE_ORDER[start_idx:end_idx + 1]

        results = []

        for phase_num in phase_range:
            result = self.run_phase(phase_num, resume=resume, dry_run=dry_run, **kwargs)
            results.append(result)

            # Stop on failure (unless dry run)
            if not dry_run and result["status"] in ["failed", "blocked", "cancelled"]:
                break

        return results

    def reset_phase(self, phase_num: str, confirm: bool = False) -> dict:
        """Reset phase by moving outputs to backup directory"""
        metadata = self.phases[phase_num]
        files_to_remove = []

        # Collect files to remove
        for file_path in metadata.output_files:
            full_path = self.config.base_dir / file_path
            if full_path.exists():
                files_to_remove.append(full_path)

        # Also include progress file
        progress_file = self.config.base_dir / f"{metadata.class_name.lower()}_progress.json"
        if progress_file.exists():
            files_to_remove.append(progress_file)

        if not files_to_remove:
            return {
                "phase": phase_num,
                "status": "nothing_to_reset",
                "message": "No output files found"
            }

        # Show preview
        print(f"\nThis will reset Phase {phase_num} ({metadata.name}):")
        for f in files_to_remove:
            print(f"  - {f.relative_to(self.config.base_dir)}")

        # Require confirmation
        if not confirm:
            response = input("\nType 'yes' to confirm: ")
            if response != 'yes':
                return {
                    "phase": phase_num,
                    "status": "cancelled",
                    "message": "User cancelled reset"
                }

        # Create backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config.base_dir / f"_backup_{phase_num}_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Move files to backup
        moved_count = 0
        for file_path in files_to_remove:
            try:
                if file_path.is_dir():
                    shutil.move(str(file_path), str(backup_dir / file_path.name))
                else:
                    shutil.move(str(file_path), str(backup_dir / file_path.name))
                moved_count += 1
            except Exception as e:
                print(f"Warning: Failed to move {file_path}: {e}")

        return {
            "phase": phase_num,
            "status": "reset",
            "files_moved": moved_count,
            "backup_location": str(backup_dir)
        }


# ============================================================================
# CLI FORMATTER
# ============================================================================

class CLIFormatter:
    """Format output for CLI display"""

    @staticmethod
    def print_status_table(status_dict: dict):
        """Print formatted table with phase status"""
        print("\nBorder Crossings Traffic Dataset Pipeline - Status")
        print("=" * 60)
        print()

        for phase_num in PHASE_ORDER:
            metadata = PHASE_REGISTRY[phase_num]
            phase_status = status_dict["phases"][phase_num]

            # Status indicator
            if phase_status["complete"]:
                if phase_status["validated"]:
                    indicator = "✓"
                else:
                    indicator = "?"
            elif phase_status["partial"]:
                indicator = "⚠"
            else:
                indicator = "○"

            print(f"Phase {phase_num}: {metadata.name}")
            print(f"  Status: {indicator} {'Complete' if phase_status['complete'] else 'Partial' if phase_status['partial'] else 'Not started'}")

            if phase_status["output_files"]:
                print(f"  Output: {', '.join(phase_status['output_files'])}")

            print()

        print("=" * 60)
        summary = status_dict["summary"]
        print(f"Progress: {summary['complete']}/{summary['total']} phases complete")
        if summary['next_runnable']:
            next_meta = PHASE_REGISTRY[summary['next_runnable']]
            print(f"Next: Phase {summary['next_runnable']} ({next_meta.name})")
        print()

    @staticmethod
    def print_phase_summary(phase_num: str, result: dict):
        """Print summary after phase execution"""
        metadata = PHASE_REGISTRY[phase_num]
        print(f"\n{'='*60}")
        print(f"Phase {phase_num}: {metadata.name}")
        print(f"{'='*60}")
        print(f"Status: {result['status']}")

        if result["status"] == "completed":
            print(f"Duration: {result.get('duration_seconds', 0):.1f} seconds")
        elif result["status"] in ["failed", "blocked", "cancelled"]:
            print(f"Reason: {result.get('reason', 'Unknown')}")

        print()

    @staticmethod
    def print_pipeline_summary(results: List[dict]):
        """Print summary of full pipeline run"""
        print(f"\n{'='*60}")
        print("Pipeline Execution Summary")
        print(f"{'='*60}\n")

        total = len(results)
        completed = sum(1 for r in results if r["status"] == "completed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "failed")

        print(f"Total phases: {total}")
        print(f"Completed: {completed}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed phases:")
            for r in results:
                if r["status"] == "failed":
                    print(f"  - Phase {r['phase']}: {r.get('reason', 'Unknown error')}")

        print()


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def handle_run_command(orchestrator: PhaseOrchestrator, args):
    """Handle run command"""
    # Determine what to run
    if args.all:
        results = orchestrator.run_pipeline(
            from_phase="1a",
            to_phase="7",
            resume=args.resume,
            dry_run=args.dry_run,
            workers=args.workers,
            device=args.device,
            confidence=args.confidence,
            model=args.model
        )

        if args.dry_run:
            print("\nDRY RUN: The following phases would be executed:\n")
            for result in results:
                phase_num = result["phase"]
                metadata = PHASE_REGISTRY[phase_num]
                print(f"{phase_num}. {metadata.name}")
                print(f"   - {metadata.description}")
                print(f"   - Estimated time: {metadata.estimated_time}")
                if result.get("status") == "skipped":
                    print(f"   - Would skip (already complete)")
                print()
            print("No changes were made (dry run mode).\n")
        else:
            CLIFormatter.print_pipeline_summary(results)

    elif args.from_phase:
        results = orchestrator.run_pipeline(
            from_phase=args.from_phase,
            to_phase="7",
            resume=args.resume,
            dry_run=args.dry_run,
            workers=args.workers,
            device=args.device,
            confidence=args.confidence,
            model=args.model
        )

        if not args.dry_run:
            CLIFormatter.print_pipeline_summary(results)

    elif args.phase:
        result = orchestrator.run_phase(
            args.phase,
            resume=args.resume,
            dry_run=args.dry_run,
            workers=args.workers,
            device=args.device,
            confidence=args.confidence,
            model=args.model
        )

        if not args.dry_run:
            CLIFormatter.print_phase_summary(args.phase, result)

            if result["status"] == "failed":
                print("Check logs: traffic_dataset/pipeline.log")
                print(f"Progress saved - can resume with: python main.py run --phase {args.phase} --resume\n")
                sys.exit(1)
            elif result["status"] == "blocked":
                print(f"To fix: {result['reason']}\n")
                sys.exit(1)


def handle_status_command(orchestrator: PhaseOrchestrator, args):
    """Handle status command"""
    status = orchestrator.get_all_status()

    if args.json:
        # Add timestamp
        status["timestamp"] = datetime.now().isoformat()
        print(json.dumps(status, indent=2))
    else:
        CLIFormatter.print_status_table(status)


def handle_validate_command(orchestrator: PhaseOrchestrator, args):
    """Handle validate command"""
    if args.all:
        # Validate all complete phases
        status = orchestrator.get_all_status()
        results = {}

        for phase_num in PHASE_ORDER:
            if status["phases"][phase_num]["complete"]:
                valid, message = orchestrator.validate_phase(phase_num)
                results[phase_num] = {
                    "validated": valid,
                    "message": message
                }

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\nValidation Results")
            print("=" * 60)
            for phase_num, result in results.items():
                metadata = PHASE_REGISTRY[phase_num]
                indicator = "✓" if result["validated"] else "✗"
                print(f"{indicator} Phase {phase_num}: {metadata.name}")
                if not result["validated"]:
                    print(f"  {result['message']}")
            print()

        # Exit code based on validation
        if all(r["validated"] for r in results.values()):
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.phase:
        valid, message = orchestrator.validate_phase(args.phase)

        if args.json:
            print(json.dumps({
                "phase": args.phase,
                "validated": valid,
                "message": message
            }, indent=2))
        else:
            metadata = PHASE_REGISTRY[args.phase]
            indicator = "✓" if valid else "✗"
            print(f"\n{indicator} Phase {args.phase}: {metadata.name}")
            print(f"  {message}\n")

        sys.exit(0 if valid else 1)


def handle_reset_command(orchestrator: PhaseOrchestrator, args):
    """Handle reset command"""
    if args.all:
        print("\n⚠️  WARNING: This will reset ALL phases!\n")

        if not args.confirm:
            response = input("Type 'RESET ALL' to confirm: ")
            if response != 'RESET ALL':
                print("Reset cancelled.\n")
                return

        # Reset all phases in reverse order
        for phase_num in reversed(PHASE_ORDER):
            result = orchestrator.reset_phase(phase_num, confirm=True)
            if result["status"] == "reset":
                print(f"✓ Reset Phase {phase_num}")

        print("\nAll phases reset.\n")

    elif args.phase:
        result = orchestrator.reset_phase(args.phase, confirm=args.confirm)

        if result["status"] == "reset":
            print(f"\n✓ Phase {args.phase} reset")
            print(f"Backup saved to: {result['backup_location']}\n")
        elif result["status"] == "nothing_to_reset":
            print(f"\nNo output files found for Phase {args.phase}\n")
        elif result["status"] == "cancelled":
            print(f"\nReset cancelled.\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Border Crossings Traffic Dataset Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire pipeline
  python main.py run --all --resume

  # Run specific phase
  python main.py run --phase 2 --device mps

  # Run from phase 3 onwards
  python main.py run --from-phase 3

  # Check status
  python main.py status

  # Validate all completed phases
  python main.py validate --all

  # Reset a phase
  python main.py reset --phase 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run pipeline phases')
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--all', action='store_true',
                          help='Run entire pipeline')
    run_group.add_argument('--phase', type=str, choices=PHASE_ORDER,
                          help='Run specific phase')
    run_group.add_argument('--from-phase', type=str, choices=PHASE_ORDER,
                          help='Run from phase N onwards')
    run_parser.add_argument('--resume', action='store_true',
                          help='Resume from existing progress')
    run_parser.add_argument('--dry-run', action='store_true',
                          help='Show what would run without executing')
    run_parser.add_argument('--workers', type=int, default=4,
                          help='Parallel workers (Phase 1c only)')
    run_parser.add_argument('--device', type=str, choices=['mps', 'cpu', 'cuda'],
                          help='Device for YOLO (Phase 2 only)')
    run_parser.add_argument('--model', type=str, default='yolo11n',
                          help='YOLO model (Phase 2 only)')
    run_parser.add_argument('--confidence', type=float, default=0.25,
                          help='YOLO confidence threshold (Phase 2 only)')

    # Status command
    status_parser = subparsers.add_parser('status',
                                         help='Show pipeline status')
    status_parser.add_argument('--json', action='store_true',
                              help='Output as JSON')
    status_parser.add_argument('--verbose', action='store_true',
                              help='Show detailed status')

    # Validate command
    validate_parser = subparsers.add_parser('validate',
                                           help='Validate phase outputs')
    validate_group = validate_parser.add_mutually_exclusive_group(required=True)
    validate_group.add_argument('--all', action='store_true',
                               help='Validate all completed phases')
    validate_group.add_argument('--phase', type=str, choices=PHASE_ORDER,
                               help='Validate specific phase')
    validate_parser.add_argument('--json', action='store_true',
                                help='Output as JSON')

    # Reset command
    reset_parser = subparsers.add_parser('reset',
                                        help='Reset phase outputs')
    reset_group = reset_parser.add_mutually_exclusive_group(required=True)
    reset_group.add_argument('--all', action='store_true',
                            help='Reset entire pipeline (DANGEROUS)')
    reset_group.add_argument('--phase', type=str, choices=PHASE_ORDER,
                            help='Reset specific phase')
    reset_parser.add_argument('--confirm', action='store_true',
                             help='Skip confirmation prompt')

    args = parser.parse_args()

    # Show help if no command
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load configurations
    try:
        config = PipelineConfig()
        config.ensure_directories()
    except Exception as e:
        print(f"ERROR: Failed to load pipeline configuration: {e}")
        sys.exit(1)

    # Azure config (optional - only needed for Phase 1a, 1c)
    azure_config = None
    try:
        azure_config = AzureConfig.from_env()
    except ValueError:
        # Azure credentials not available - that's OK for some commands
        pass

    # Create orchestrator
    orchestrator = PhaseOrchestrator(config, azure_config)

    # Execute command
    try:
        if args.command == 'run':
            handle_run_command(orchestrator, args)
        elif args.command == 'status':
            handle_status_command(orchestrator, args)
        elif args.command == 'validate':
            handle_validate_command(orchestrator, args)
        elif args.command == 'reset':
            handle_reset_command(orchestrator, args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.\n")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
