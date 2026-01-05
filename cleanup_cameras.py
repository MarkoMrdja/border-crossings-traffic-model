#!/usr/bin/env python3
"""
Script to remove problematic cameras from the dataset.

Cameras to remove:
- VATIN_U: Poor camera positioning, not useful
- GOSTUN_U: Duplicate of GOSTUN_I (both show same camera)
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

# Cameras to remove
CAMERAS_TO_REMOVE = ['VATIN_U', 'GOSTUN_U']

# Base directory
BASE_DIR = Path(__file__).parent / 'traffic_dataset'

def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], path: Path, backup: bool = True):
    """Save JSON file with optional backup."""
    if backup and path.exists():
        backup_path = path.with_suffix('.json.backup')
        shutil.copy2(path, backup_path)
        print(f"  Created backup: {backup_path.name}")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Updated: {path.name}")

def clean_lane_polygons():
    """Remove cameras from lane_polygons.json."""
    print("\n1. Cleaning lane_polygons.json...")

    path = BASE_DIR / 'lane_polygons.json'
    if not path.exists():
        print(f"  Skipped: {path.name} not found")
        return

    data = load_json(path)
    cameras_before = len(data.get('cameras', {}))

    # Remove cameras from cameras dict
    for camera_id in CAMERAS_TO_REMOVE:
        if camera_id in data.get('cameras', {}):
            del data['cameras'][camera_id]
            print(f"  Removed from cameras: {camera_id}")

    # Remove cameras from skipped_cameras list
    if 'skipped_cameras' in data:
        original_skipped = data['skipped_cameras'][:]
        data['skipped_cameras'] = [
            cam for cam in data['skipped_cameras']
            if cam not in CAMERAS_TO_REMOVE
        ]
        for camera_id in CAMERAS_TO_REMOVE:
            if camera_id in original_skipped:
                print(f"  Removed from skipped: {camera_id}")

    # Update statistics
    if 'statistics' in data:
        data['statistics']['total_cameras'] = len(data['cameras'])

    cameras_after = len(data['cameras'])
    save_json(data, path)
    print(f"  Cameras: {cameras_before} → {cameras_after}")

def clean_inventory():
    """Remove cameras from inventory.json."""
    print("\n2. Cleaning inventory.json...")

    path = BASE_DIR / 'inventory.json'
    if not path.exists():
        print(f"  Skipped: {path.name} not found")
        return

    data = load_json(path)

    # inventory.json has flat structure with camera_id as top-level keys
    cameras_before = sum(1 for k in data.keys() if k in CAMERAS_TO_REMOVE or k.endswith('_I') or k.endswith('_U'))

    # Remove camera keys
    for camera_id in CAMERAS_TO_REMOVE:
        if camera_id in data:
            del data[camera_id]
            print(f"  Removed camera: {camera_id}")

    cameras_after = sum(1 for k in data.keys() if k.endswith('_I') or k.endswith('_U'))
    save_json(data, path)
    print(f"  Cameras: {cameras_before} → {cameras_after}")

def clean_sample_manifest():
    """Remove cameras from sample_manifest.json."""
    print("\n3. Cleaning sample_manifest.json...")

    path = BASE_DIR / 'sample_manifest.json'
    if not path.exists():
        print(f"  Skipped: {path.name} not found")
        return

    data = load_json(path)

    # Remove from 'cameras' list if it exists
    if 'cameras' in data and isinstance(data['cameras'], list):
        cameras_before = len(data['cameras'])
        data['cameras'] = [c for c in data['cameras'] if c not in CAMERAS_TO_REMOVE]
        print(f"  Cameras list: {cameras_before} → {len(data['cameras'])}")

    # Remove from 'per_camera' dict if it exists
    if 'per_camera' in data and isinstance(data['per_camera'], dict):
        for camera_id in CAMERAS_TO_REMOVE:
            if camera_id in data['per_camera']:
                del data['per_camera'][camera_id]
                print(f"  Removed from per_camera: {camera_id}")

    # Remove from 'samples' list if it exists
    if 'samples' in data and isinstance(data['samples'], list):
        samples_before = len(data['samples'])
        data['samples'] = [
            sample for sample in data['samples']
            if sample.get('camera_id') not in CAMERAS_TO_REMOVE
        ]
        samples_after = len(data['samples'])
        if samples_before != samples_after:
            print(f"  Samples list: {samples_before} → {samples_after}")

    # Filter out samples from removed cameras in 'selection'
    if 'selection' in data:
        samples_before = len(data['selection'])
        data['selection'] = [
            sample for sample in data['selection']
            if sample.get('camera_id') not in CAMERAS_TO_REMOVE
        ]
        samples_after = len(data['selection'])
        print(f"  Selection: {samples_before} → {samples_after}")

    save_json(data, path)

def clean_yolo_results():
    """Remove cameras from yolo_results.json."""
    print("\n4. Cleaning yolo_results.json...")

    path = BASE_DIR / 'yolo_results.json'
    if not path.exists():
        print(f"  Skipped: {path.name} not found")
        return

    data = load_json(path)
    analyses_before = len(data.get('analyses', []))

    # Filter out analyses from removed cameras
    data['analyses'] = [
        analysis for analysis in data.get('analyses', [])
        if analysis.get('camera_id') not in CAMERAS_TO_REMOVE
    ]

    # Update statistics
    if 'statistics' in data:
        data['statistics']['total_analyzed'] = len(data['analyses'])

        # Recount by traffic level
        by_traffic_level = {}
        for analysis in data['analyses']:
            level = analysis.get('traffic_level', 'unknown')
            by_traffic_level[level] = by_traffic_level.get(level, 0) + 1
        data['statistics']['by_traffic_level'] = by_traffic_level

    analyses_after = len(data['analyses'])
    save_json(data, path)
    print(f"  Analyses: {analyses_before} → {analyses_after}")

def clean_balanced_selection():
    """Remove cameras from balanced_selection.json."""
    print("\n5. Cleaning balanced_selection.json...")

    path = BASE_DIR / 'balanced_selection.json'
    if not path.exists():
        print(f"  Skipped: {path.name} not found")
        return

    data = load_json(path)

    # Remove from 'camera_stats' dict if it exists
    if 'camera_stats' in data and isinstance(data['camera_stats'], dict):
        for camera_id in CAMERAS_TO_REMOVE:
            if camera_id in data['camera_stats']:
                del data['camera_stats'][camera_id]
                print(f"  Removed from camera_stats: {camera_id}")

    # Remove from 'selected_samples' list if it exists
    if 'selected_samples' in data and isinstance(data['selected_samples'], list):
        samples_before = len(data['selected_samples'])
        data['selected_samples'] = [
            sample for sample in data['selected_samples']
            if sample.get('camera_id') not in CAMERAS_TO_REMOVE
        ]
        samples_after = len(data['selected_samples'])
        if samples_before != samples_after:
            print(f"  Selected samples list: {samples_before} → {samples_after}")

    # Filter out selections from removed cameras
    if 'selection' in data:
        selections_before = len(data['selection'])
        data['selection'] = [
            item for item in data['selection']
            if item.get('camera_id') not in CAMERAS_TO_REMOVE
        ]
        selections_after = len(data['selection'])
        print(f"  Selection: {selections_before} → {selections_after}")

    # Update num_cameras count
    if 'num_cameras' in data and 'camera_stats' in data:
        data['num_cameras'] = len(data['camera_stats'])

    save_json(data, path)

def clean_roi_references():
    """Remove cameras from roi_references.json."""
    print("\n6. Cleaning roi_references.json...")

    path = BASE_DIR / 'roi_references.json'
    if not path.exists():
        print(f"  Skipped: {path.name} not found")
        return

    data = load_json(path)
    refs_before = len(data)

    # Remove cameras (flat structure with camera_id as keys)
    for camera_id in CAMERAS_TO_REMOVE:
        if camera_id in data:
            del data[camera_id]
            print(f"  Removed camera: {camera_id}")

    refs_after = len(data)
    save_json(data, path)
    print(f"  Cameras: {refs_before} → {refs_after}")

def delete_raw_images():
    """Delete raw image directories for removed cameras."""
    print("\n7. Deleting raw image directories...")

    raw_dir = BASE_DIR / 'raw'
    if not raw_dir.exists():
        print(f"  Skipped: raw directory not found")
        return

    for camera_id in CAMERAS_TO_REMOVE:
        camera_dir = raw_dir / camera_id
        if camera_dir.exists():
            # Count files before deleting
            file_count = len(list(camera_dir.glob('*')))

            # Delete directory
            shutil.rmtree(camera_dir)
            print(f"  Deleted: {camera_id}/ ({file_count} files)")
        else:
            print(f"  Not found: {camera_id}/")

def verify_cleanup():
    """Verify that cameras were removed from all files."""
    print("\n8. Verifying cleanup...")

    files_to_check = [
        'lane_polygons.json',
        'inventory.json',
        'sample_manifest.json',
        'yolo_results.json',
        'balanced_selection.json',
        'roi_references.json'
    ]

    all_clean = True

    for filename in files_to_check:
        path = BASE_DIR / filename
        if not path.exists():
            continue

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        found_cameras = []
        for camera_id in CAMERAS_TO_REMOVE:
            if camera_id in content:
                found_cameras.append(camera_id)

        if found_cameras:
            print(f"  ✗ {filename}: Still contains {found_cameras}")
            all_clean = False
        else:
            print(f"  ✓ {filename}: Clean")

    # Check raw directories
    raw_dir = BASE_DIR / 'raw'
    if raw_dir.exists():
        for camera_id in CAMERAS_TO_REMOVE:
            if (raw_dir / camera_id).exists():
                print(f"  ✗ raw/{camera_id}/: Still exists")
                all_clean = False
            else:
                print(f"  ✓ raw/{camera_id}/: Deleted")

    return all_clean

def main():
    """Main cleanup function."""
    print("=" * 60)
    print("Camera Cleanup Script")
    print("=" * 60)
    print(f"\nRemoving cameras: {', '.join(CAMERAS_TO_REMOVE)}")
    print(f"Base directory: {BASE_DIR}")

    # Confirm with user
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cleanup cancelled.")
        return

    # Run cleanup steps
    clean_lane_polygons()
    clean_inventory()
    clean_sample_manifest()
    clean_yolo_results()
    clean_balanced_selection()
    clean_roi_references()
    delete_raw_images()

    # Verify
    all_clean = verify_cleanup()

    # Summary
    print("\n" + "=" * 60)
    if all_clean:
        print("✓ Cleanup completed successfully!")
    else:
        print("⚠ Cleanup completed with warnings (see above)")
    print("=" * 60)
    print("\nBackup files created with .backup extension")
    print("To restore, rename .backup files back to .json")

if __name__ == '__main__':
    main()
