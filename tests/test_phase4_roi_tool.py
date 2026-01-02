"""
Unit tests for Phase 4: ROI Definition Tool
"""

import unittest
import json
import tempfile
from pathlib import Path
from dataset_pipeline.phase4_roi_tool import ROIDefinitionTool
from dataset_pipeline.config import PipelineConfig


class TestROIDefinitionTool(unittest.TestCase):
    """Test cases for ROI Definition Tool."""

    def test_roi_tool_initialization(self):
        """Test that ROIDefinitionTool initializes correctly."""
        config = PipelineConfig()
        tool = ROIDefinitionTool(config)

        self.assertEqual(tool.phase_name, "roi_definition")
        self.assertEqual(tool.description, "Defining ROI polygons for cameras")
        self.assertEqual(tool.current_polygon, [])
        self.assertIsNone(tool.current_camera_id)

    def test_roi_config_structure_validation(self):
        """Test validation of ROI config structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.base_dir = Path(tmpdir)
            tool = ROIDefinitionTool(config)

            # Create a valid test config
            test_config = {
                "created_at": "2024-01-01T12:00:00",
                "cameras": {
                    "TEST_CAMERA_U": {
                        "reference_image": "raw/TEST_CAMERA_U/2024-01-01_12-00-00.jpg",
                        "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                        "defined_at": "2024-01-01T12:30:00"
                    }
                },
                "skipped_cameras": []
            }

            # Save test config
            output_path = Path(tmpdir) / "roi_config.json"
            with open(output_path, 'w') as f:
                json.dump(test_config, f)

            # Override output path for testing
            tool.output_path = output_path

            # Validate
            self.assertTrue(tool.validate())

    def test_roi_config_missing_fields(self):
        """Test validation fails for missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.base_dir = Path(tmpdir)
            tool = ROIDefinitionTool(config)

            # Create invalid config (missing 'cameras' field)
            test_config = {
                "created_at": "2024-01-01T12:00:00",
                "skipped_cameras": []
            }

            output_path = Path(tmpdir) / "roi_config.json"
            with open(output_path, 'w') as f:
                json.dump(test_config, f)

            tool.output_path = output_path

            # Should fail validation
            self.assertFalse(tool.validate())

    def test_roi_polygon_validation(self):
        """Test validation of polygon structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.base_dir = Path(tmpdir)
            tool = ROIDefinitionTool(config)

            # Create config with invalid polygon (less than 3 vertices)
            test_config = {
                "created_at": "2024-01-01T12:00:00",
                "cameras": {
                    "TEST_CAMERA": {
                        "reference_image": "raw/TEST_CAMERA/test.jpg",
                        "polygon": [[100, 100], [200, 100]],  # Only 2 vertices
                        "defined_at": "2024-01-01T12:30:00"
                    }
                },
                "skipped_cameras": []
            }

            output_path = Path(tmpdir) / "roi_config.json"
            with open(output_path, 'w') as f:
                json.dump(test_config, f)

            tool.output_path = output_path

            # Should fail validation (polygon needs at least 3 vertices)
            self.assertFalse(tool.validate())

    def test_load_or_initialize_config(self):
        """Test loading existing config or creating new one."""
        config = PipelineConfig()
        tool = ROIDefinitionTool(config)

        # Test creating new config
        new_config = tool._load_or_initialize_config(resume=False)
        self.assertIn("created_at", new_config)
        self.assertIn("cameras", new_config)
        self.assertIn("skipped_cameras", new_config)
        self.assertEqual(new_config["cameras"], {})
        self.assertEqual(new_config["skipped_cameras"], [])

    def test_get_cameras_to_process(self):
        """Test filtering cameras to process based on resume mode."""
        config = PipelineConfig()
        tool = ROIDefinitionTool(config)

        # Mock ROI references
        roi_references = {
            "CAMERA_1": {"local_path": "path1.jpg"},
            "CAMERA_2": {"local_path": "path2.jpg"},
            "CAMERA_3": {"local_path": "path3.jpg"}
        }

        # Mock existing ROI config (CAMERA_1 already has ROI)
        roi_config = {
            "created_at": "2024-01-01T12:00:00",
            "cameras": {
                "CAMERA_1": {
                    "polygon": [[0, 0], [100, 0], [100, 100]]
                }
            },
            "skipped_cameras": []
        }

        # Test with resume=False (process all)
        cameras = tool._get_cameras_to_process(roi_references, roi_config, resume=False)
        self.assertEqual(set(cameras), {"CAMERA_1", "CAMERA_2", "CAMERA_3"})

        # Test with resume=True (skip CAMERA_1)
        cameras = tool._get_cameras_to_process(roi_references, roi_config, resume=True)
        self.assertEqual(set(cameras), {"CAMERA_2", "CAMERA_3"})


if __name__ == "__main__":
    unittest.main()
