"""Utility functions for embodied benchmarks."""

from .point_extraction import (
    extract_molmo_points,
    extract_molmo_v1_points,
    extract_click_format_points,
    extract_xy_attribute_points,
    extract_python_tuple_points,
    extract_qwen_xml_points,
    extract_json_point2d,
    extract_bracket_points,
    extract_points_robust,
    check_point_in_mask,
    text2pts_official,
    calculate_accuracy_official,
    get_model_type,
    format_pointing_prompt,
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
    'extract_molmo_v1_points',
    'extract_click_format_points',
    'extract_xy_attribute_points',
    'extract_python_tuple_points',
    'extract_qwen_xml_points',
    'extract_json_point2d',
    'extract_bracket_points',
    'extract_points_robust',
    'check_point_in_mask',
    'text2pts_official',
    'calculate_accuracy_official',
    'get_model_type',
    'format_pointing_prompt',
    'extract_answer_letter',
    'normalize_text',
    'extract_number',
    'index_to_letter',
    'letter_to_index',
    'build_mcq_prompt',
]
