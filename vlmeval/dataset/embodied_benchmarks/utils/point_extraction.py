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


def extract_molmo_points(text: str) -> List[Tuple[float, float]]:
    """
    Extract points from Molmo coords="x y" format.

    Molmo outputs coordinates in format: coords="prefix x_coord y_coord"
    where prefix is 1-2 digit number to skip, and x_coord/y_coord are 3-4 digits.

    Example: coords="1 1 946 029" -> point is (94.6, 2.9)
    The "1 1" is a prefix, actual point is "946 029" (divide by 10 to get percentage).

    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text containing coords="x y" patterns

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    points = []

    # First extract the coords string
    coords_pattern = r'coords="([^"]+)"'
    coords_match = re.search(coords_pattern, text)

    if not coords_match:
        return points

    coords_str = coords_match.group(1)

    # Pattern from notebook: skip prefix number, extract 3-4 digit x and y
    # Format: "prefix x_coord y_coord" where x_coord and y_coord are 3-4 digits
    point_pattern = r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})"

    for match in re.finditer(point_pattern, coords_str):
        try:
            # First group is prefix (ignored), second is x, third is y
            _, x_str, y_str = match.groups()
            x = float(x_str) / 10.0  # Convert from 0-1000 to 0-100
            y = float(y_str) / 10.0
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


def extract_points_robust(text: str) -> List[Tuple[float, float]]:
    """
    Robust point extraction using multiple strategies.

    Tries in order:
    1. Python tuple format (most common in model outputs)
    2. Molmo coords format

    Returns points in 0-100 scale (percentage).

    Args:
        text: Generated text that may contain point coordinates

    Returns:
        List of (x, y) tuples in 0-100 scale
    """
    # Strategy A: Try Python tuple format first
    points = extract_python_tuple_points(text)
    if points:
        return points

    # Strategy B: Try Molmo coords format
    points = extract_molmo_points(text)
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
