"""
Point extraction utilities for pointing/spatial benchmarks.

Supports multiple formats:
- Molmo coords format: coords="x y" (0-1000 scale)
- Python tuple format: (0.45, 0.55) (0-1 or 0-100 scale)
- Pixel coordinate format: (450, 550) (pixel scale)
"""

import re
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union


def get_model_type(model_name: str) -> str:
    """Map model name to model type for prompt/parsing selection.

    Returns one of: 'molmo', 'qwen25', 'qwen3', 'llava', 'internvl', 'phi4'
    """
    if model_name is None:
        return 'molmo'
    name = model_name.lower()
    if 'molmo' in name:
        return 'molmo'
    elif 'qwen3' in name:
        return 'qwen3'
    elif 'qwen' in name:
        return 'qwen25'
    elif 'llava' in name:
        return 'llava'
    elif 'internvl' in name or 'intern_vl' in name:
        return 'internvl'
    elif 'phi4' in name or 'phi-4' in name:
        return 'phi4'
    return 'molmo'


def format_pointing_prompt(description: str, model_name: str = None) -> str:
    """Generate model-specific pointing prompt.

    Args:
        description: The object or task description (e.g., "the red cup").
        model_name: Model name to determine prompt format.

    Returns:
        Full prompt string appropriate for the model.
    """
    model_type = get_model_type(model_name)

    if model_type == 'molmo':
        return f"Point to {description}."
    elif model_type == 'qwen3':
        return (
            f"Locate {description} with a point and report its point coordinates in JSON format "
            'like this: {"point_2d": [x, y], "label": "object/region"}'
        )
    elif model_type == 'qwen25':
        return (
            f"{description}\n"
            "Output its coordinates in XML format <points x y>object</points>."
        )
    elif model_type == 'internvl':
        return f"Point to {description}."
    else:
        # llava, phi4, etc.
        return (
            f"{description}\n"
            "Give EXACT PIXEL COORDINATES in [x, y] format, "
            "where x is horizontal and y is vertical. "
            "ONLY return coordinates with no additional text."
        )


def extract_molmo_points(text: str) -> List[Tuple[float, float]]:
    """
    Extract points from Molmo v2 coords="frame_idx point_idx x y ..." format.

    Uses the official two-step parsing from UnifiedPointFormatter:
    1. frame_regex: separates frame_id from coordinates
    2. points_regex: extracts (point_idx, x, y) from coordinates

    Molmo2 format: <points coords="frame_idx point_1_idx x1 y1 point_2_idx x2 y2 ...">label</points>
    Example: coords="1 1 461 527 2 481 521" -> points [(46.1, 52.7), (48.1, 52.1)]

    Coordinates are in 0-1000 scale (3-4 digits with zero-padding).
    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text containing coords="..." patterns

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    points = []

    # Step 1: Extract the coords string (official coord_regex)
    coord_regex = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
    coord_match = coord_regex.search(text)

    if not coord_match:
        # Fallback: try simpler pattern
        coords_pattern = r'coords="([^"]+)"'
        coord_match = re.search(coords_pattern, text)
        if not coord_match:
            return points

    coords_str = coord_match.group(1)

    # Step 2: Use frame_regex to separate frame_id from coordinates (official method)
    # This explicitly handles frame_idx instead of relying on digit count
    # Supports multiple frames separated by \t, :, ,, or ;
    frame_regex = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")

    # Step 3: Extract points from coordinate part (official points_regex)
    points_regex = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

    # Use finditer to handle multiple frames (video case)
    frame_matches = list(frame_regex.finditer(coords_str))

    if frame_matches:
        # Official two-step method: frame_id is explicitly separated
        for frame_match in frame_matches:
            coord_part = frame_match.group(2)  # Get coordinates part after frame_id
            for match in points_regex.finditer(coord_part):
                try:
                    _, x_str, y_str = match.groups()
                    x = float(x_str) / 10.0  # Convert from 0-1000 to 0-100
                    y = float(y_str) / 10.0
                    points.append((x, y))
                except (ValueError, IndexError):
                    continue
    else:
        # Fallback: try direct matching on entire coords_str
        for match in points_regex.finditer(coords_str):
            try:
                _, x_str, y_str = match.groups()
                x = float(x_str) / 10.0
                y = float(y_str) / 10.0
                points.append((x, y))
            except (ValueError, IndexError):
                continue

    return points


def extract_qwen_xml_points(text: str, img_width: int = 0, img_height: int = 0) -> List[Tuple[float, float]]:
    """
    Extract points from Qwen2.5-VL/Qwen3-VL XML format.

    Qwen outputs: <points x1="450" y1="320">object</points>
    or multi-point: <points x1="309" y1="242" x2="450" y2="252" x3="605" y3="292">
    or: <points 450 320>object</points>

    Coordinates may be in 0-1000 scale (Qwen3-VL) or pixel space (Qwen2.5-VL).
    Converted to 0-100 scale using image dimensions if available.

    Returns points in 0-100 scale (percentage).
    """
    points = []

    def _scale(x, y):
        if img_width > 0 and img_height > 0:
            return (x / img_width) * 100.0, (y / img_height) * 100.0
        elif x > 100.0 or y > 100.0:
            return x / 10.0, y / 10.0
        return x, y

    # Pattern 1: Find <points ...> tags and extract all xN="V" yN="V" pairs
    tag_pattern = r'<points\s+([^>]+)>'
    for tag_match in re.finditer(tag_pattern, text):
        attrs = tag_match.group(1)
        # Extract all xN="V" yN="V" pairs from attributes
        xy_pattern = r'x\d*=["\']?(\d+\.?\d*)["\']?\s+y\d*=["\']?(\d+\.?\d*)["\']?'
        for match in re.finditer(xy_pattern, attrs):
            try:
                x, y = float(match.group(1)), float(match.group(2))
                x, y = _scale(x, y)
                points.append((x, y))
            except (ValueError, IndexError):
                continue

    if points:
        return points

    # Pattern 2: <points 450 320>object</points>
    pattern_direct = r'<points\s+(\d+\.?\d*)\s+(\d+\.?\d*)>.*?</points>'
    for match in re.finditer(pattern_direct, text):
        try:
            x, y = float(match.group(1)), float(match.group(2))
            x, y = _scale(x, y)
            points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def extract_json_point2d(text: str) -> List[Tuple[float, float]]:
    """
    Extract points from Qwen3-VL JSON point_2d format.

    Qwen3-VL outputs: {"point_2d": [x, y], "label": "object"}
    or a JSON array: [{"point_2d": [x, y], "label": "object"}, ...]

    Coordinates are in 0-1000 scale. Converted to 0-100 scale.

    Returns points in 0-100 scale (percentage).
    """
    import json

    points = []

    # Try to parse as JSON (possibly wrapped in markdown code fence)
    clean = text.strip()
    if clean.startswith('```json'):
        clean = clean[7:]
    if clean.startswith('```'):
        clean = clean[3:]
    if clean.endswith('```'):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'point_2d' in item:
                    coords = item['point_2d']
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        x, y = float(coords[0]), float(coords[1])
                        # Qwen3-VL uses 0-1000 scale
                        if x > 100 or y > 100:
                            x, y = x / 10.0, y / 10.0
                        points.append((x, y))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    if points:
        return points

    # Fallback: regex for point_2d pattern in text
    pattern = r'"point_2d"\s*:\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    for match in re.finditer(pattern, text):
        try:
            x, y = float(match.group(1)), float(match.group(2))
            if x > 100 or y > 100:
                x, y = x / 10.0, y / 10.0
            points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def extract_bracket_points(text: str, img_width: int = 0, img_height: int = 0) -> List[Tuple[float, float]]:
    """
    Extract points from bracket format: [x, y].

    Used by LLaVA-OneVision, InternVL, Phi4, etc.
    Coordinates are in pixel space. Converted to 0-100 scale using image dimensions.

    Returns points in 0-100 scale (percentage).
    """
    points = []
    pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    for match in re.finditer(pattern, text):
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            if img_width > 0 and img_height > 0:
                x = (x / img_width) * 100.0
                y = (y / img_height) * 100.0
            elif x > 100.0 or y > 100.0:
                x = x / 10.0
                y = y / 10.0
            elif x <= 1.0 and y <= 1.0:
                x = x * 100.0
                y = y * 100.0
            points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def extract_python_tuple_points(text: str) -> List[Tuple[float, float]]:
    """
    Extract points from Python tuple format: (x, y).

    Auto-scales based on value magnitude:
    - If x > 1.0: assumes 0-100 or 0-1000 scale
    - If x <= 1.0: assumes 0-1 scale, multiplies by 100

    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text containing (x, y) patterns

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    points = []
    # Pattern for Python tuple: (0.45, 0.55) or (45, 55) or (450, 550)
    pattern = r'\(\s*([0-9]+\.?[0-9]*)\s*,\s*([0-9]+\.?[0-9]*)\s*\)'
    matches = re.findall(pattern, text)

    for match in matches:
        try:
            x = float(match[0])
            y = float(match[1])

            # Auto-scale based on value magnitude
            if x > 100.0 or y > 100.0:
                # Likely 0-1000 scale, divide by 10
                x = x / 10.0
                y = y / 10.0
            elif x <= 1.0 and y <= 1.0:
                # Likely 0-1 scale, multiply by 100
                x = x * 100.0
                y = y * 100.0
            # else: already in 0-100 scale

            points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def extract_molmo_v1_points(text: str) -> List[Tuple[float, float]]:
    """
    Extract points from Molmo v1 format: p=xxx,yyy or 1=xxx,yyy.

    This is the format used by older Molmo models (v1).
    Pattern: (?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})

    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text containing p=xxx,yyy patterns

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    points = []
    # Pattern from official pointarena evaluator
    pattern = r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})'

    for match in re.finditer(pattern, text):
        try:
            x = int(match.group(1)) / 10.0  # Convert from 0-1000 to 0-100
            y = int(match.group(2)) / 10.0
            if 0 <= x <= 100 and 0 <= y <= 100:
                points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def extract_click_format_points(text: str) -> List[Tuple[float, float]]:
    """
    Extract points from Click(x,y) format (0-100 scale).

    Pattern: Click(x.x, y.y)

    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text containing Click(x,y) patterns

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    points = []
    pattern = r'Click\(([0-9]+\.?[0-9]*),?\s*([0-9]+\.?[0-9]*)\)'

    for match in re.finditer(pattern, text):
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            if 0 <= x <= 100 and 0 <= y <= 100:
                points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def extract_xy_attribute_points(text: str) -> List[Tuple[float, float]]:
    """
    Extract points from x="x" y="y" format (0-100 scale).

    Pattern: x1="45.5" y1="67.3"

    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text containing x="x" y="y" patterns

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    points = []
    pattern = r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"'

    for match in re.finditer(pattern, text):
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            if 0 <= x <= 100 and 0 <= y <= 100:
                points.append((x, y))
        except (ValueError, IndexError):
            continue

    return points


def extract_points_robust(text: str, img_width: int = 0, img_height: int = 0) -> List[Tuple[float, float]]:
    """
    Robust point extraction using multiple strategies.

    Tries in order:
    1. Molmo v2 coords format (coords="prefix x y")
    2. Molmo v1 format (p=xxx,yyy)
    3. Qwen XML format (<points x y>obj</points>)
    4. Bracket format ([x, y])
    5. Click format (Click(x,y))
    6. xy attribute format (x="x" y="y")
    7. Python tuple format ((x, y))

    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text that may contain point coordinates
        img_width: Image width in pixels (for pixel→percentage conversion)
        img_height: Image height in pixels (for pixel→percentage conversion)

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    # Strategy A: Try Molmo v2 coords format first (most common for Molmo2)
    points = extract_molmo_points(text)
    if points:
        return points

    # Strategy B: Try Molmo v1 format (p=xxx,yyy)
    points = extract_molmo_v1_points(text)
    if points:
        return points

    # Strategy C: Try Qwen3-VL JSON point_2d format
    points = extract_json_point2d(text)
    if points:
        return points

    # Strategy D: Try Qwen XML format
    points = extract_qwen_xml_points(text, img_width, img_height)
    if points:
        return points

    # Strategy E: Try bracket format [x, y]
    points = extract_bracket_points(text, img_width, img_height)
    if points:
        return points

    # Strategy F: Try Click format
    points = extract_click_format_points(text)
    if points:
        return points

    # Strategy G: Try xy attribute format
    points = extract_xy_attribute_points(text)
    if points:
        return points

    # Strategy H: Try Python tuple format
    points = extract_python_tuple_points(text)
    if points:
        return points

    return []


def check_point_in_mask(
    x: float,
    y: float,
    mask: Union[Image.Image, np.ndarray],
    width: int,
    height: int
) -> bool:
    """
    Check if a point (in 0-100 percentage scale) is inside a mask.

    Args:
        x: X coordinate in 0-100 scale (percentage of image width)
        y: Y coordinate in 0-100 scale (percentage of image height)
        mask: Binary mask as PIL Image or numpy array
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        True if point is inside the mask (non-zero value)
    """
    # Convert percentage to pixel coordinates
    px = int((x / 100.0) * width)
    py = int((y / 100.0) * height)

    # Clamp to valid range
    px = max(0, min(px, width - 1))
    py = max(0, min(py, height - 1))

    # Convert mask to numpy array if needed
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask)
    else:
        mask_array = mask

    # Handle multi-channel masks (take any channel)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    # Check if point is in mask (non-zero value)
    try:
        return mask_array[py, px] > 0
    except IndexError:
        return False


def text2pts_official(text: str, width: int, height: int) -> np.ndarray:
    """
    Convert text to pixel points using Where2Place official format.

    Supports:
    - Point format: (x, y) -> single pixel
    - Bbox format: (x1, y1, x2, y2) -> all pixels in bbox

    Args:
        text: Generated text containing point or bbox
        width: Image width
        height: Image height

    Returns:
        Numpy array of shape (N, 2) with pixel coordinates
    """
    # Pattern for tuple with 2 or 4 numbers
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)

    all_points = []

    for match in matches:
        nums = [float(x.strip()) for x in match.split(',')]

        if len(nums) == 2:
            # Point format: (x, y)
            x, y = nums
            if x <= 1.0 and y <= 1.0:
                # Normalized coordinates
                px = int(x * width)
                py = int(y * height)
            else:
                # Already pixel coordinates
                px = int(x)
                py = int(y)
            all_points.append([px, py])

        elif len(nums) == 4:
            # Bbox format: (x1, y1, x2, y2)
            x1, y1, x2, y2 = nums
            if all(v <= 1.0 for v in nums):
                # Normalized coordinates
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Generate all points in bbox
            for py in range(min(y1, y2), max(y1, y2) + 1):
                for px in range(min(x1, x2), max(x1, x2) + 1):
                    all_points.append([px, py])

    if not all_points:
        return np.array([]).reshape(0, 2)

    return np.array(all_points)


def calculate_accuracy_official(points: np.ndarray, mask_image: Image.Image) -> float:
    """
    Calculate accuracy for Where2Place: percentage of points inside mask.

    Args:
        points: Numpy array of shape (N, 2) with pixel coordinates
        mask_image: Binary mask as PIL Image

    Returns:
        Accuracy as float (0-1)
    """
    if len(points) == 0:
        return 0.0

    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    # Normalize mask to 0-1
    if mask_array.max() > 1:
        mask_array = mask_array / 255.0

    height, width = mask_array.shape

    # Filter valid points
    valid_mask = (
        (points[:, 0] >= 0) & (points[:, 0] < width) &
        (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    valid_points = points[valid_mask]

    if len(valid_points) == 0:
        return 0.0

    # Calculate accuracy
    inside_mask = mask_array[valid_points[:, 1], valid_points[:, 0]]
    accuracy = np.mean(inside_mask)

    return float(accuracy)
