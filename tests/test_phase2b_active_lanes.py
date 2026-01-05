"""
Unit tests for Phase 2b active lane detection functionality.

Tests the X-coordinate clustering logic that detects which lanes
are actually active at a border crossing.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_pipeline.phase2b_yolo_filtered import YOLOFilteredAnalysisPhase
from dataset_pipeline.config import PipelineConfig


class TestActiveLaneDetection(unittest.TestCase):
    """Test cases for active lane detection logic."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a phase instance for testing
        config = PipelineConfig()
        self.phase = YOLOFilteredAnalysisPhase(config)

    def test_two_lanes_skips_clustering(self):
        """Test that 2-lane crossings skip clustering and return total_lanes."""
        # Simulate polygon from x=0 to x=1000
        polygon = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)

        # 10 vehicles clustered on left side
        boxes = [
            {'xyxy': [100, 100, 150, 150]},  # Left lane
            {'xyxy': [120, 200, 170, 250]},  # Left lane
            {'xyxy': [110, 300, 160, 350]},  # Left lane
        ]

        # Should return 2 (total lanes) without clustering
        active_lanes = self.phase._detect_active_lanes(boxes, polygon, total_lanes=2)
        self.assertEqual(active_lanes, 2)

    def test_six_lanes_two_active(self):
        """Test 6-lane crossing with only 2 active lanes (user's example)."""
        # Simulate polygon from x=0 to x=1200 (6 lanes, 200px each)
        polygon = np.array([[0, 0], [1200, 0], [1200, 500], [0, 500]], dtype=np.float32)

        # 20 vehicles in first 2 lanes (0-400px range)
        boxes = []
        for i in range(20):
            x_pos = 50 + (i % 4) * 80  # Spread across first 2 lanes (0-400px)
            y_pos = 100 + (i // 4) * 50
            boxes.append({'xyxy': [x_pos, y_pos, x_pos + 50, y_pos + 50]})

        active_lanes = self.phase._detect_active_lanes(boxes, polygon, total_lanes=6)

        # Should detect 2 active lanes (not 6)
        self.assertEqual(active_lanes, 2,
                        f"Expected 2 active lanes, got {active_lanes}. "
                        f"This should normalize to 20/2=10 cars/lane (moderate), "
                        f"not 20/6=3.3 cars/lane (light)")

    def test_six_lanes_all_active(self):
        """Test 6-lane crossing with all lanes active."""
        # Simulate polygon from x=0 to x=1200 (6 lanes, 200px each)
        polygon = np.array([[0, 0], [1200, 0], [1200, 500], [0, 500]], dtype=np.float32)

        # 18 vehicles spread across all 6 lanes (3 per lane)
        boxes = []
        for lane in range(6):
            lane_center = lane * 200 + 100  # Center of each lane
            for i in range(3):
                x_pos = lane_center - 20 + i * 20
                y_pos = 100 + i * 50
                boxes.append({'xyxy': [x_pos, y_pos, x_pos + 50, y_pos + 50]})

        active_lanes = self.phase._detect_active_lanes(boxes, polygon, total_lanes=6)

        # Should detect 6 active lanes
        self.assertEqual(active_lanes, 6,
                        f"Expected 6 active lanes, got {active_lanes}")

    def test_empty_lanes_returns_total(self):
        """Test that empty crossings return total_lanes (avoids division by zero)."""
        polygon = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)

        active_lanes = self.phase._detect_active_lanes([], polygon, total_lanes=4)
        self.assertEqual(active_lanes, 4)

    def test_classification_with_active_lanes(self):
        """Test traffic classification using active lanes."""
        # Scenario 1: 20 cars in 2 active lanes = 10 cars/lane = moderate
        traffic_level = self.phase._classify_traffic_lane_aware(20, 2)
        self.assertEqual(traffic_level, "likely_heavy",  # 10 cars/lane is heavy (>= 5)
                        "20 cars in 2 lanes should be heavy traffic")

        # Scenario 2: 10 cars in 2 active lanes = 5 cars/lane = moderate/heavy boundary
        traffic_level = self.phase._classify_traffic_lane_aware(10, 2)
        self.assertEqual(traffic_level, "likely_heavy",  # 5 cars/lane is exactly at threshold
                        "10 cars in 2 lanes should be heavy traffic")

        # Scenario 3: 8 cars in 2 active lanes = 4 cars/lane = moderate
        traffic_level = self.phase._classify_traffic_lane_aware(8, 2)
        self.assertEqual(traffic_level, "likely_moderate",
                        "8 cars in 2 lanes should be moderate traffic")

        # Scenario 4: 20 cars in 6 active lanes = 3.3 cars/lane = moderate
        traffic_level = self.phase._classify_traffic_lane_aware(20, 6)
        self.assertEqual(traffic_level, "likely_moderate",
                        "20 cars in 6 lanes should be moderate traffic")

    def test_edge_cases(self):
        """Test edge cases for active lane detection."""
        polygon = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)

        # Single vehicle should detect 1 active lane
        boxes = [{'xyxy': [100, 100, 150, 150]}]
        active_lanes = self.phase._detect_active_lanes(boxes, polygon, total_lanes=6)
        self.assertEqual(active_lanes, 1)

        # Vehicles at exact boundaries
        boxes = [
            {'xyxy': [0, 100, 50, 150]},      # Far left
            {'xyxy': [950, 100, 1000, 150]},  # Far right
        ]
        active_lanes = self.phase._detect_active_lanes(boxes, polygon, total_lanes=6)
        # Should detect 2 lanes (first and last)
        self.assertGreaterEqual(active_lanes, 2)

    def test_degenerate_polygon(self):
        """Test that degenerate polygons (zero width) fall back to total_lanes."""
        # Polygon with zero width
        polygon = np.array([[100, 0], [100, 0], [100, 500], [100, 500]], dtype=np.float32)

        boxes = [{'xyxy': [100, 100, 150, 150]}]
        active_lanes = self.phase._detect_active_lanes(boxes, polygon, total_lanes=4)
        self.assertEqual(active_lanes, 4)  # Should fall back to total


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
