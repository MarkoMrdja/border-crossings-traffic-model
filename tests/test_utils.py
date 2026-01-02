"""
Unit tests for dataset_pipeline.utils module

Tests path parsing, JSON I/O, progress tracking, and validation utilities.
"""

import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

from dataset_pipeline.utils import (
    parse_camera_id,
    construct_blob_path,
    construct_local_path,
    parse_blob_name,
    load_json,
    save_json,
    create_progress_tracker,
    update_progress,
    ensure_directory,
    get_file_size_mb,
    count_files,
    validate_image_path
)


class TestPathParsing(unittest.TestCase):
    """Test path construction and parsing functions."""

    def test_parse_camera_id(self):
        """Test camera ID parsing."""
        border, direction = parse_camera_id("GRADINA_U")
        self.assertEqual(border, "GRADINA")
        self.assertEqual(direction, "U")

        border, direction = parse_camera_id("KELEBIJA_I")
        self.assertEqual(border, "KELEBIJA")
        self.assertEqual(direction, "I")

    def test_parse_camera_id_invalid(self):
        """Test camera ID parsing with invalid input."""
        with self.assertRaises(ValueError):
            parse_camera_id("INVALID")

        with self.assertRaises(ValueError):
            parse_camera_id("NOUNDERSCORE")

    def test_construct_blob_path(self):
        """Test Azure blob path construction."""
        path = construct_blob_path("GRADINA_U", "2024", "07", "15", "16-20-58")
        self.assertEqual(path, "GRADINA/U/2024/07/15/16-20-58.jpg")

        # Test with non-padded month
        path = construct_blob_path("GRADINA_U", "2024", "7", "15", "16-20-58")
        self.assertEqual(path, "GRADINA/U/2024/7/15/16-20-58.jpg")

    def test_construct_local_path(self):
        """Test local path construction."""
        path = construct_local_path("GRADINA_U", "2024", "07", "15", "16-20-58")
        self.assertEqual(path, "raw/GRADINA_U/2024-07-15_16-20-58.jpg")

        # Test with non-padded month (should be padded in output)
        path = construct_local_path("GRADINA_U", "2024", "7", "15", "16-20-58")
        self.assertEqual(path, "raw/GRADINA_U/2024-07-15_16-20-58.jpg")

        # Test with custom base subdir
        path = construct_local_path(
            "GRADINA_U", "2024", "07", "15", "16-20-58",
            base_subdir="crops"
        )
        self.assertEqual(path, "crops/GRADINA_U/2024-07-15_16-20-58.jpg")

    def test_parse_blob_name(self):
        """Test blob name parsing."""
        result = parse_blob_name("GRADINA/U/2024/07/15/16-20-58.jpg")

        self.assertIsNotNone(result)
        self.assertEqual(result["border"], "GRADINA")
        self.assertEqual(result["direction"], "U")
        self.assertEqual(result["camera_id"], "GRADINA_U")
        self.assertEqual(result["year"], "2024")
        self.assertEqual(result["month"], "07")
        self.assertEqual(result["day"], "15")
        self.assertEqual(result["time"], "16-20-58")

    def test_parse_blob_name_invalid(self):
        """Test blob name parsing with invalid input."""
        self.assertIsNone(parse_blob_name("invalid/path"))
        self.assertIsNone(parse_blob_name("TOO/SHORT.jpg"))
        self.assertIsNone(parse_blob_name("GRADINA/U/2024/07/15/not_a_jpg.txt"))


class TestJSONIO(unittest.TestCase):
    """Test JSON I/O functions."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def test_save_and_load_json(self):
        """Test saving and loading JSON data."""
        test_data = {
            "key": "value",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }

        file_path = Path(self.temp_dir) / "test.json"

        # Save
        result = save_json(test_data, file_path)
        self.assertTrue(result)
        self.assertTrue(file_path.exists())

        # Load
        loaded_data = load_json(file_path)
        self.assertEqual(loaded_data, test_data)

    def test_load_json_nonexistent(self):
        """Test loading JSON from nonexistent file."""
        file_path = Path(self.temp_dir) / "nonexistent.json"

        # Should return default value
        result = load_json(file_path, default={})
        self.assertEqual(result, {})

        result = load_json(file_path, default=None)
        self.assertIsNone(result)

    def test_load_json_invalid(self):
        """Test loading invalid JSON file."""
        file_path = Path(self.temp_dir) / "invalid.json"

        # Write invalid JSON
        with open(file_path, "w") as f:
            f.write("{ invalid json }")

        # Should return default value
        result = load_json(file_path, default={})
        self.assertEqual(result, {})

    def test_save_json_creates_directories(self):
        """Test that save_json creates parent directories."""
        nested_path = Path(self.temp_dir) / "nested" / "path" / "data.json"
        test_data = {"test": "data"}

        result = save_json(test_data, nested_path)
        self.assertTrue(result)
        self.assertTrue(nested_path.exists())

        loaded = load_json(nested_path)
        self.assertEqual(loaded, test_data)


class TestProgressTracking(unittest.TestCase):
    """Test progress tracking functions."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def test_create_progress_tracker(self):
        """Test creating a new progress tracker."""
        tracker = create_progress_tracker(1000, "Testing")

        self.assertEqual(tracker["total"], 1000)
        self.assertEqual(tracker["completed"], 0)
        self.assertEqual(tracker["description"], "Testing")
        self.assertEqual(tracker["last_index"], -1)
        self.assertIsInstance(tracker["failed"], list)
        self.assertIn("started_at", tracker)
        self.assertIn("updated_at", tracker)

    def test_update_progress(self):
        """Test updating progress tracker."""
        progress_file = Path(self.temp_dir) / "progress.json"

        # Create initial progress
        initial = create_progress_tracker(100, "Download")
        save_json(initial, progress_file)

        # Update progress
        updated = update_progress(
            progress_file,
            {"completed": 50, "last_index": 49},
            auto_save=True
        )

        self.assertEqual(updated["completed"], 50)
        self.assertEqual(updated["last_index"], 49)
        self.assertEqual(updated["total"], 100)

        # Verify it was saved
        loaded = load_json(progress_file)
        self.assertEqual(loaded["completed"], 50)

    def test_update_progress_creates_new(self):
        """Test that update_progress creates new file if doesn't exist."""
        progress_file = Path(self.temp_dir) / "new_progress.json"

        updated = update_progress(
            progress_file,
            {"total": 200, "completed": 10},
            auto_save=True
        )

        self.assertEqual(updated["total"], 200)
        self.assertEqual(updated["completed"], 10)
        self.assertTrue(progress_file.exists())


class TestFileSystemUtilities(unittest.TestCase):
    """Test file system utility functions."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def test_ensure_directory(self):
        """Test directory creation."""
        test_dir = Path(self.temp_dir) / "nested" / "directory"

        result = ensure_directory(test_dir)
        self.assertTrue(result.exists())
        self.assertTrue(result.is_dir())

        # Should not fail if directory already exists
        result2 = ensure_directory(test_dir)
        self.assertTrue(result2.exists())

    def test_get_file_size_mb(self):
        """Test file size calculation."""
        test_file = Path(self.temp_dir) / "test.bin"

        # Create file with known size (1 MB)
        with open(test_file, "wb") as f:
            f.write(b"0" * (1024 * 1024))

        size = get_file_size_mb(test_file)
        self.assertAlmostEqual(size, 1.0, places=1)

        # Test nonexistent file
        size = get_file_size_mb(Path(self.temp_dir) / "nonexistent.bin")
        self.assertEqual(size, 0.0)

    def test_count_files(self):
        """Test file counting."""
        test_dir = Path(self.temp_dir) / "count_test"
        ensure_directory(test_dir)

        # Create some test files
        for i in range(5):
            (test_dir / f"file{i}.txt").touch()
        for i in range(3):
            (test_dir / f"image{i}.jpg").touch()

        # Count all files
        count = count_files(test_dir)
        self.assertEqual(count, 8)

        # Count only JPG files
        count = count_files(test_dir, "*.jpg")
        self.assertEqual(count, 3)

        # Count in nonexistent directory
        count = count_files(Path(self.temp_dir) / "nonexistent")
        self.assertEqual(count, 0)


class TestValidationUtilities(unittest.TestCase):
    """Test validation utility functions."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def test_validate_image_path_valid(self):
        """Test image validation with valid file."""
        image_path = Path(self.temp_dir) / "test.jpg"

        # Create file with sufficient size
        with open(image_path, "wb") as f:
            f.write(b"0" * 2000)

        result = validate_image_path(image_path, min_size_bytes=1000)
        self.assertTrue(result)

    def test_validate_image_path_nonexistent(self):
        """Test image validation with nonexistent file."""
        image_path = Path(self.temp_dir) / "nonexistent.jpg"

        result = validate_image_path(image_path)
        self.assertFalse(result)

    def test_validate_image_path_wrong_extension(self):
        """Test image validation with wrong file extension."""
        file_path = Path(self.temp_dir) / "test.txt"

        with open(file_path, "wb") as f:
            f.write(b"0" * 2000)

        result = validate_image_path(file_path)
        self.assertFalse(result)

    def test_validate_image_path_too_small(self):
        """Test image validation with file too small."""
        image_path = Path(self.temp_dir) / "small.jpg"

        # Create small file
        with open(image_path, "wb") as f:
            f.write(b"0" * 100)

        result = validate_image_path(image_path, min_size_bytes=1000)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
