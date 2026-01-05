"""
Cleanup script to delete old YOLO results before re-running Phase 2b.

This script removes:
1. Old yolo_results.json
2. All cropped images in traffic_dataset/crops/
3. All labeled images in traffic_dataset/labeled/

Run this AFTER you've completed Phase 4b (lane polygon annotation)
and BEFORE running Phase 2b (new YOLO with polygon filtering).
"""

import shutil
from pathlib import Path

def cleanup_old_results():
    """Delete old YOLO results and processed images."""
    base_dir = Path("traffic_dataset")

    print("=" * 60)
    print("CLEANUP: Preparing for Phase 2b Re-run")
    print("=" * 60)

    # 1. Delete old YOLO results
    old_yolo = base_dir / "yolo_results.json"
    if old_yolo.exists():
        print(f"\n✓ Deleting old YOLO results: {old_yolo}")
        old_yolo.unlink()
    else:
        print(f"\n  (Old yolo_results.json not found - OK)")

    # Backup old results if needed
    old_yolo_backup = base_dir / "yolo_results_backup.json"
    if old_yolo_backup.exists():
        print(f"✓ Deleting old YOLO backup: {old_yolo_backup}")
        old_yolo_backup.unlink()

    # 2. Delete all cropped images
    crops_dir = base_dir / "crops"
    if crops_dir.exists():
        total_files = sum(1 for _ in crops_dir.rglob("*.jpg"))
        if total_files > 0:
            print(f"\n✓ Deleting {total_files} cropped images from {crops_dir}")
            shutil.rmtree(crops_dir)
            crops_dir.mkdir(parents=True, exist_ok=True)

            # Recreate subdirectories
            for level in ["likely_empty", "likely_light", "likely_moderate", "likely_heavy"]:
                (crops_dir / level).mkdir(exist_ok=True)
        else:
            print(f"\n  (No cropped images found - OK)")
    else:
        print(f"\n  (Crops directory not found - OK)")

    # 3. Delete all labeled images
    labeled_dir = base_dir / "labeled"
    if labeled_dir.exists():
        total_files = sum(1 for _ in labeled_dir.rglob("*.jpg"))
        if total_files > 0:
            print(f"✓ Deleting {total_files} labeled images from {labeled_dir}")
            shutil.rmtree(labeled_dir)
            labeled_dir.mkdir(parents=True, exist_ok=True)

            # Recreate subdirectories
            for level in ["empty", "light", "moderate", "heavy"]:
                (labeled_dir / level).mkdir(exist_ok=True)
        else:
            print(f"  (No labeled images found - OK)")
    else:
        print(f"  (Labeled directory not found - OK)")

    # 4. Delete crop summary if exists
    crop_summary = base_dir / "crop_summary.json"
    if crop_summary.exists():
        print(f"✓ Deleting crop summary: {crop_summary}")
        crop_summary.unlink()

    # 5. Keep these files (DO NOT DELETE):
    keep_files = [
        "inventory.json",
        "sample_manifest.json",
        "balanced_selection.json",
        "roi_references.json",
        "lane_polygons.json",  # Your new polygon annotations!
        "mini_test_dataset/"    # Your reference images
    ]

    print(f"\n{'=' * 60}")
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print("\nKept important files:")
    for filename in keep_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")

    print("\n✓ Ready to run Phase 2b!")
    print("\nNext step:")
    print("  python -m dataset_pipeline.phase2b_yolo_filtered")
    print()


if __name__ == "__main__":
    # Confirm before deletion
    print("\n⚠️  WARNING: This will delete old YOLO results and processed images!")
    print("\nThis will remove:")
    print("  - yolo_results.json")
    print("  - All images in traffic_dataset/crops/")
    print("  - All images in traffic_dataset/labeled/")
    print("\nThis will KEEP:")
    print("  - Raw images")
    print("  - lane_polygons.json (your annotations)")
    print("  - sample_manifest.json")
    print("  - All other configuration files")

    response = input("\nContinue? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        cleanup_old_results()
    else:
        print("\nCancelled.")
