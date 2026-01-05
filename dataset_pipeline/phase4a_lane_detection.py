"""
Phase 4a: Lane Detection Algorithm

Auto-detects lane polygon boundaries using computer vision techniques:
- Canny edge detection (adaptive thresholds)
- Hough line transform (angle-filtered)
- YOLO box-guided region inference (convex hull)
- Road contour detection
- Polygon merging and simplification
- Confidence scoring

Returns polygon suggestions for interactive review.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def detect_lane_polygon(
    image_path: str,
    yolo_boxes: List[Dict[str, Any]],
    mode: str = 'single',
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Auto-detect lane polygon(s) from image.

    Args:
        image_path: Path to camera image
        yolo_boxes: List of YOLO bounding boxes with 'xyxy' coordinates
        mode: 'single' for one polygon, 'multi' for per-lane (TBD)
        config: Configuration dictionary (uses defaults if None)

    Returns:
        List of polygon suggestions with confidence scores
    """
    # Use default config if none provided
    if config is None:
        config = get_default_config()

    # Load and validate image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return []

    h, w = img.shape[:2]
    logger.debug(f"Processing image: {w}x{h}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Adaptive Canny edge detection
    edges = adaptive_canny_edge_detection(gray, config['edge_detection'])

    # Step 2: Hough line detection
    lines = hough_line_detection(edges, config['line_detection'])

    # Step 3: YOLO-guided traffic region
    traffic_region = infer_traffic_region_from_yolo(yolo_boxes, (h, w))

    # Step 4: Road contour detection
    road_contours = detect_road_contours(edges, (h, w))

    # Step 5: Merge signals into polygon
    if mode == 'single':
        polygon = merge_into_single_polygon(
            traffic_region,
            lines,
            road_contours,
            (h, w),
            config['polygon_simplification']
        )

        # Step 6: Calculate confidence
        confidence = calculate_confidence(polygon, yolo_boxes, edges, (h, w))

        return [{
            'id': 0,
            'name': 'all_lanes',
            'polygon': polygon.tolist() if isinstance(polygon, np.ndarray) else polygon,
            'confidence': confidence,
            'auto_detected': True,
            'user_modified': False
        }]

    elif mode == 'multi':
        # Multi-lane mode not implemented yet
        logger.warning("Multi-lane mode not yet implemented, falling back to single")
        return detect_lane_polygon(image_path, yolo_boxes, 'single', config)

    else:
        logger.error(f"Invalid mode: {mode}")
        return []


def adaptive_canny_edge_detection(
    gray: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Apply Canny edge detection with adaptive thresholds.

    Args:
        gray: Grayscale image
        config: Edge detection configuration

    Returns:
        Edge map (binary image)
    """
    if config.get('adaptive_thresholds', True):
        # Calculate adaptive thresholds based on median brightness
        median_brightness = np.median(gray)
        lower_percentile = config.get('lower_percentile', 0.5)
        upper_percentile = config.get('upper_percentile', 1.5)

        lower_threshold = int(max(0, median_brightness * lower_percentile))
        upper_threshold = int(min(255, median_brightness * upper_percentile))
    else:
        # Use fixed thresholds
        lower_threshold = config.get('lower_threshold', 50)
        upper_threshold = config.get('upper_threshold', 150)

    logger.debug(f"Canny thresholds: {lower_threshold}, {upper_threshold}")

    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    return edges


def hough_line_detection(
    edges: np.ndarray,
    config: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Detect lines using Hough transform and filter by angle.

    Args:
        edges: Edge map from Canny detection
        config: Line detection configuration

    Returns:
        Filtered lines array or None if no lines detected
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=config.get('rho', 1),
        theta=np.pi / config.get('theta_resolution', 180),
        threshold=config.get('threshold', 50),
        minLineLength=config.get('min_line_length', 100),
        maxLineGap=config.get('max_line_gap', 50)
    )

    if lines is None or len(lines) == 0:
        logger.debug("No lines detected by Hough transform")
        return None

    logger.debug(f"Detected {len(lines)} lines before filtering")

    # Filter lines by angle
    vertical_range = config.get('vertical_angle_range', (60, 120))
    horizontal_range = config.get('horizontal_angle_range', [(0, 30), (150, 180)])

    filtered_lines = filter_lines_by_angle(lines, vertical_range, horizontal_range)

    logger.debug(f"Filtered to {len(filtered_lines) if filtered_lines is not None else 0} lines")

    return filtered_lines


def filter_lines_by_angle(
    lines: np.ndarray,
    vertical_range: Tuple[int, int],
    horizontal_range: List[Tuple[int, int]]
) -> Optional[np.ndarray]:
    """
    Filter Hough lines by angle constraints.

    Args:
        lines: Lines from Hough transform
        vertical_range: (min, max) degrees for vertical lines
        horizontal_range: List of (min, max) tuples for horizontal lines

    Returns:
        Filtered lines array or None
    """
    filtered = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle in degrees (0-180)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        # Check if in vertical range
        if vertical_range[0] <= angle <= vertical_range[1]:
            filtered.append(line)
            continue

        # Check if in any horizontal range
        for h_min, h_max in horizontal_range:
            if h_min <= angle <= h_max:
                filtered.append(line)
                break

    if not filtered:
        return None

    return np.array(filtered)


def infer_traffic_region_from_yolo(
    yolo_boxes: List[Dict[str, Any]],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create traffic region polygon from YOLO vehicle detections.

    Strategy:
    - If vehicles detected: Create convex hull around vehicle centroids, expand by 20%
    - If no vehicles: Use default bottom 60% of image (roads typically at bottom)

    Args:
        yolo_boxes: YOLO bounding boxes
        image_shape: (height, width) of image

    Returns:
        Polygon as numpy array of shape (N, 2)
    """
    h, w = image_shape

    if yolo_boxes and len(yolo_boxes) > 0:
        # Calculate centroids of all YOLO boxes
        centroids = []
        for box in yolo_boxes:
            xyxy = box.get('xyxy', [])
            if len(xyxy) == 4:
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centroids.append([cx, cy])

        if centroids:
            centroids = np.array(centroids, dtype=np.float32)

            # Create convex hull
            if len(centroids) >= 3:
                hull = cv2.convexHull(centroids)
                hull = hull.reshape(-1, 2)

                # Expand hull by 20%
                traffic_region = expand_polygon(hull, (h, w), factor=1.2)
                logger.debug(f"Created traffic region from {len(centroids)} YOLO boxes")
                return traffic_region

    # Fallback: Use bottom 60% of image
    logger.debug("Using default traffic region (bottom 60%)")
    traffic_region = np.array([
        [0, int(h * 0.4)],      # Top-left
        [w, int(h * 0.4)],      # Top-right
        [w, h],                 # Bottom-right
        [0, h]                  # Bottom-left
    ], dtype=np.float32)

    return traffic_region


def expand_polygon(
    polygon: np.ndarray,
    image_shape: Tuple[int, int],
    factor: float = 1.2
) -> np.ndarray:
    """
    Expand polygon outward from its centroid.

    Args:
        polygon: Polygon vertices (N, 2)
        image_shape: (height, width) for bounds clamping
        factor: Expansion factor (1.2 = 20% larger)

    Returns:
        Expanded polygon
    """
    h, w = image_shape

    # Calculate centroid
    centroid = polygon.mean(axis=0)

    # Expand each vertex away from centroid
    expanded = centroid + (polygon - centroid) * factor

    # Clamp to image bounds
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h)

    return expanded.astype(np.float32)


def detect_road_contours(
    edges: np.ndarray,
    image_shape: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Detect large contours representing road surface.

    Args:
        edges: Edge map from Canny detection
        image_shape: (height, width) of image

    Returns:
        List of large contours (road candidates)
    """
    h, w = image_shape

    # Apply morphological closing to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area (keep large road-like regions)
    min_area = (h * w) * 0.1  # At least 10% of image
    road_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    logger.debug(f"Found {len(road_contours)} large road contours (from {len(contours)} total)")

    return road_contours


def merge_into_single_polygon(
    traffic_region: np.ndarray,
    lines: Optional[np.ndarray],
    road_contours: List[np.ndarray],
    image_shape: Tuple[int, int],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Merge multiple signals into a single polygon.

    Strategy:
    1. Start with traffic_region (YOLO-based hull or default)
    2. If road contours exist, intersect with largest contour
    3. Simplify polygon to reduce vertices

    Args:
        traffic_region: Polygon from YOLO boxes
        lines: Detected lines (currently not used, could refine boundaries)
        road_contours: Large contours from road surface
        image_shape: (height, width) of image
        config: Polygon simplification configuration

    Returns:
        Final simplified polygon
    """
    h, w = image_shape

    # Start with traffic region
    polygon = traffic_region

    # If we have road contours, try to refine with largest one
    if road_contours:
        largest_contour = max(road_contours, key=cv2.contourArea)

        # Approximate contour to polygon
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        contour_poly = cv2.approxPolyDP(largest_contour, epsilon, True)
        contour_poly = contour_poly.reshape(-1, 2).astype(np.float32)

        # For now, use the contour if it's reasonable size
        # (More sophisticated intersection could be added)
        contour_area = cv2.contourArea(largest_contour)
        image_area = h * w
        if contour_area / image_area > 0.15:  # At least 15% of image
            logger.debug("Using road contour as base polygon")
            polygon = contour_poly
        else:
            logger.debug("Road contour too small, using traffic region")

    # Simplify polygon
    polygon = simplify_polygon(polygon, config)

    return polygon


def simplify_polygon(
    polygon: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Simplify polygon by reducing number of vertices.

    Args:
        polygon: Input polygon (N, 2)
        config: Simplification configuration

    Returns:
        Simplified polygon
    """
    # Ensure polygon is in correct format for cv2.approxPolyDP
    polygon = polygon.reshape(-1, 1, 2).astype(np.float32)

    # Calculate epsilon based on perimeter
    perimeter = cv2.arcLength(polygon, True)
    epsilon_factor = config.get('epsilon_factor', 0.02)
    epsilon = epsilon_factor * perimeter

    # Approximate polygon
    simplified = cv2.approxPolyDP(polygon, epsilon, True)
    simplified = simplified.reshape(-1, 2)

    # Enforce vertex limits
    min_vertices = config.get('min_vertices', 3)
    max_vertices = config.get('max_vertices', 12)

    if len(simplified) < min_vertices:
        logger.warning(f"Simplified polygon has only {len(simplified)} vertices (min {min_vertices})")
        return polygon.reshape(-1, 2)  # Return original

    if len(simplified) > max_vertices:
        # Further simplification
        epsilon *= 1.5
        simplified = cv2.approxPolyDP(polygon, epsilon, True)
        simplified = simplified.reshape(-1, 2)

    logger.debug(f"Simplified polygon from {len(polygon)} to {len(simplified)} vertices")

    return simplified.astype(np.float32)


def calculate_confidence(
    polygon: np.ndarray,
    yolo_boxes: List[Dict[str, Any]],
    edges: np.ndarray,
    image_shape: Tuple[int, int]
) -> float:
    """
    Calculate confidence score for polygon suggestion.

    Metrics:
    - YOLO box coverage (40%): % of YOLO boxes inside polygon
    - Edge alignment (40%): % of polygon edges aligned with image edges
    - Simplicity (20%): Prefer 4-6 vertices

    Args:
        polygon: Suggested polygon
        yolo_boxes: YOLO bounding boxes
        edges: Edge map
        image_shape: (height, width) of image

    Returns:
        Confidence score (0.0 to 1.0)
    """
    score = 0.0

    # Metric 1: YOLO box coverage (40% weight)
    if yolo_boxes and len(yolo_boxes) > 0:
        boxes_inside = 0
        for box in yolo_boxes:
            xyxy = box.get('xyxy', [])
            if len(xyxy) == 4:
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Check if centroid is inside polygon
                if point_in_polygon((cx, cy), polygon):
                    boxes_inside += 1

        coverage = boxes_inside / len(yolo_boxes)
        score += coverage * 0.4
        logger.debug(f"YOLO coverage: {coverage:.2f} ({boxes_inside}/{len(yolo_boxes)})")
    else:
        # No boxes to validate against
        score += 0.2  # Neutral score

    # Metric 2: Edge alignment (40% weight)
    edge_alignment = calculate_edge_alignment(polygon, edges)
    score += edge_alignment * 0.4
    logger.debug(f"Edge alignment: {edge_alignment:.2f}")

    # Metric 3: Simplicity (20% weight)
    num_vertices = len(polygon)
    if 4 <= num_vertices <= 6:
        simplicity = 1.0
    else:
        simplicity = max(0, 1.0 - abs(num_vertices - 5) * 0.1)
    score += simplicity * 0.2
    logger.debug(f"Simplicity: {simplicity:.2f} ({num_vertices} vertices)")

    return min(1.0, score)


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Check if point is inside polygon using cv2.pointPolygonTest.

    Args:
        point: (x, y) coordinates
        polygon: Polygon vertices (N, 2)

    Returns:
        True if point is inside polygon
    """
    polygon = polygon.reshape(-1, 1, 2).astype(np.float32)
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0  # >= 0 means inside or on boundary


def calculate_edge_alignment(polygon: np.ndarray, edges: np.ndarray) -> float:
    """
    Calculate how well polygon edges align with detected edges.

    Strategy:
    - Draw polygon on blank image
    - Count pixels where polygon overlaps with edge map
    - Normalize by polygon perimeter

    Args:
        polygon: Polygon vertices
        edges: Edge map from Canny detection

    Returns:
        Alignment score (0.0 to 1.0)
    """
    h, w = edges.shape[:2]

    # Create mask with polygon edges
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon_int = polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [polygon_int], True, 255, thickness=5)

    # Count overlapping pixels
    overlap = np.logical_and(mask > 0, edges > 0).sum()

    # Normalize by polygon perimeter (approximate)
    perimeter_pixels = np.sum(mask > 0)
    if perimeter_pixels == 0:
        return 0.0

    alignment = min(1.0, overlap / perimeter_pixels)

    return alignment


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for lane detection.

    Returns:
        Configuration dictionary
    """
    return {
        'edge_detection': {
            'method': 'canny',
            'adaptive_thresholds': True,
            'lower_percentile': 0.5,
            'upper_percentile': 1.5,
            'lower_threshold': 50,
            'upper_threshold': 150
        },
        'line_detection': {
            'method': 'hough',
            'rho': 1,
            'theta_resolution': 180,
            'threshold': 50,
            'min_line_length': 100,
            'max_line_gap': 50,
            'vertical_angle_range': (60, 120),
            'horizontal_angle_range': [(0, 30), (150, 180)]
        },
        'polygon_simplification': {
            'epsilon_factor': 0.02,
            'min_vertices': 3,
            'max_vertices': 12
        },
        'confidence_thresholds': {
            'high': 0.75,
            'medium': 0.50,
            'low': 0.30
        }
    }
