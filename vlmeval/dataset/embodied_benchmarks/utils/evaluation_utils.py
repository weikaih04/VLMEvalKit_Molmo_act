"""
Evaluation utilities for embodied benchmarks.

Provides common functions for:
- Answer letter extraction (MCQ)
- Text normalization
- Number extraction
"""

import re
import string
from typing import Optional


def extract_answer_letter(text: str, max_letter: str = 'H') -> Optional[str]:
    """
    Extract answer letter from model output.

    Supports multiple formats:
    - Direct letter: "A", "B", "C"
    - Parenthesized: "(A)", "(B)", "(C)"
    - With prefix: "Answer: A", "The answer is B"
    - With period: "A.", "B."

    Args:
        text: Model generated text
        max_letter: Maximum valid letter (default 'H' for up to 8 options)

    Returns:
        Extracted letter with parentheses like "(A)", or None if not found
    """
    if not text:
        return None

    # Convert to uppercase first (matching notebook behavior)
    text = text.strip().upper()
    valid_letters = string.ascii_uppercase[:ord(max_letter) - ord('A') + 1]
    letter_pattern = f'[{valid_letters}]'

    # Priority patterns (ordered by specificity)
    patterns = [
        # Exact parenthesized letter
        rf'\(({letter_pattern})\)',
        # "ANSWER: A" or "ANSWER IS A" format
        rf'ANSWER:?\s*IS?\s*\(?({letter_pattern})\)?',
        # Letter followed by period at start
        rf'^({letter_pattern})\.',
        # Letter followed by ) at start (e.g., "A)")
        rf'^({letter_pattern})\)',
        # Letter followed by colon
        rf'^({letter_pattern}):',
        # Single letter at start of text
        rf'^({letter_pattern})\s',
        # Single letter at end of text
        rf'\s({letter_pattern})$',
        # Any standalone letter (word boundary)
        rf'\b({letter_pattern})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            letter = match.group(1).upper()
            return f'({letter})'

    # Last resort: check if text is just a single letter
    if len(text) == 1 and text in valid_letters:
        return f'({text})'

    return None


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Operations:
    - Convert to lowercase
    - Strip whitespace
    - Remove punctuation

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ''

    text = str(text).lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse multiple spaces
    text = ' '.join(text.split())

    return text


def extract_number(text: str) -> Optional[str]:
    """
    Extract a number from text.

    Supports:
    - Digit format: "5", "42"
    - Word format: "five", "forty-two"

    Args:
        text: Input text

    Returns:
        Extracted number as string, or None if not found
    """
    if not text:
        return None

    # Word to number mapping
    word_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    }

    text_lower = text.lower().strip()

    # Try to find digit number first
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return match.group(1)

    # Try word numbers
    for word, num in word_map.items():
        if word in text_lower:
            return num

    return None


def index_to_letter(index: int) -> str:
    """
    Convert 0-based index to letter with parentheses.

    Args:
        index: 0-based index (0 -> A, 1 -> B, etc.)

    Returns:
        Letter with parentheses like "(A)"
    """
    return f'({chr(ord("A") + index)})'


def letter_to_index(letter: str) -> int:
    """
    Convert letter to 0-based index.

    Args:
        letter: Letter like "A", "(A)", or "a"

    Returns:
        0-based index
    """
    # Extract just the letter
    letter = letter.strip().upper()
    if letter.startswith('(') and letter.endswith(')'):
        letter = letter[1:-1]
    return ord(letter) - ord('A')


def build_mcq_prompt(question: str, options: list, suffix: str = "Answer with the option letter.") -> str:
    """
    Build a standard MCQ prompt.

    Format:
    {question}
    (A) option1
    (B) option2
    ...
    Answer with the option letter.

    Args:
        question: The question text
        options: List of option texts
        suffix: Instruction suffix

    Returns:
        Formatted prompt string
    """
    formatted_options = []
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        formatted_options.append(f"({letter}) {opt}")

    prompt = f"{question}\n" + "\n".join(formatted_options) + f"\n{suffix}"
    return prompt
