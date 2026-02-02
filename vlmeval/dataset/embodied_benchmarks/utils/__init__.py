"""Utility functions for embodied benchmarks."""

from .point_extraction import (
    extract_molmo_points,
    extract_python_tuple_points,
    extract_points_robust,
    check_point_in_mask,
    text2pts_official,
    calculate_accuracy_official,
)

from .evaluation_utils import (
    extract_answer_letter,
    normalize_text,
    extract_number,
    index_to_letter,
    letter_to_index,
    build_mcq_prompt,
)

__all__ = [
    'extract_molmo_points',
    'extract_python_tuple_points',
    'extract_points_robust',
    'check_point_in_mask',
    'text2pts_official',
    'calculate_accuracy_official',
    'extract_answer_letter',
    'normalize_text',
    'extract_number',
    'index_to_letter',
    'letter_to_index',
    'build_mcq_prompt',
]
