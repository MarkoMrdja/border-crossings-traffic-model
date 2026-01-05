#!/usr/bin/env python3
"""
Cleanup script to prepare for Phase 2+ re-run.

This script removes old pipeline results while preserving:
- Raw downloaded images (Phase 1)
- Lane polygons (Phase 4b - your manual annotations)
- Sample manifest and inventory

Run this before starting the new lane-aware pipeline workflow.
"""

import shutil
from pathlib import Path
from datetime import datetime


def main():
    """Clean up old pipeline results."""
    base_dir = Path("traffic_dataset")

    # Files to remove (old results that will be regenerated)
    files_to_remove = [
        "yolo_results.json",
        "yolo_results_filtered.json",
        "balanced_selection.json",
        "roi_references.json",
        "crop_summary.json",
        "split_manifest.json",
    ]

    # Directories to clean (will be regenerated)
    dirs_to_clean = [
        "crops",
        "labeled",
        "final",
        "yolo_verification",
    ]

    print("="*70)
    print("CLEANUP FOR PHASE 2+ RE-RUN")
    print("="*70)
    print()
    print("This will remove old pipeline results to prepare for re-run.")
    print("The following will be PRESERVED:")
    print("  ✓ Raw downloaded images (raw/)")
    print("  ✓ Lane polygons (lane_polygons.json)")
    print("  ✓ Sample manifest and inventory")
    print()

    # Create backup timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = base_dir / f"backup_{timestamp}"

    # Ask for confirmation
    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return

    print()
    print("Creating backup directory:", backup_dir)
    backup_dir.mkdir(exist_ok=True)

    # Remove files
    print("\nRemoving old result files:")
    for file_name in files_to_remove:
        file_path = base_dir / file_name
        if file_path.exists():
            # Backup before removing
            backup_path = backup_dir / file_name
            shutil.copy2(file_path, backup_path)
            file_path.unlink()
            print(f"  ✓ Removed {file_name} (backed up)")
        else:
            print(f"  - {file_name} (not found)")

    # Clean directories
    print("\nCleaning output directories:")
    for dir_name in dirs_to_clean:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            # Backup before removing
            backup_path = backup_dir / dir_name
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(dir_path, backup_path)
            shutil.rmtree(dir_path)
            dir_path.mkdir(exist_ok=True)
            print(f"  ✓ Cleaned {dir_name}/ (backed up)")
        else:
            print(f"  - {dir_name}/ (not found)")

    print()
    print("="*70)
    print("CLEANUP COMPLETE!")
    print("="*70)
    print(f"Backup saved to: {backup_dir}")
    print()
    print("You can now run the new pipeline workflow.")
    print()


if __name__ == "__main__":
    main()
