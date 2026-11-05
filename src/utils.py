"""Digit-result types, confidence validation, and spoken-number formatting."""

from dataclasses import dataclass

import numpy as np
from num2words import num2words


def digits_to_words(number_str: str) -> str:
	"""Convert a digit string to words without losing leading zeroes."""
	# num2words accepts integers, which discard leading zeroes such as the first two in "007".
	leading_zero_count = len(number_str) - len(number_str.lstrip("0"))
	if number_str and leading_zero_count == len(number_str):
		return " ".join(num2words(0) for _ in number_str)

	remaining = number_str[leading_zero_count:]
	try:
		number_words = num2words(int(remaining))
	except (ValueError, OverflowError):
		return " ".join(
			num2words(int(character)) if character.isdigit() else character
			for character in number_str
		)

	return " ".join([*("zero" for _ in range(leading_zero_count)), number_words])


@dataclass
class PredictionResult:
	"""One predicted digit, its confidence, and the model input crop."""

	digit: str
	confidence: float
	image_tensor: np.ndarray


def validate_predictions(
	results: list[PredictionResult], confidence_threshold: float = 0.95
) -> dict:
	"""Accept a number only when every digit meets the threshold (0.95 by default)."""
	# A single uncertain digit can change the whole number, so the 0.95 default gates output.
	is_valid = all(result.confidence >= confidence_threshold for result in results)
	if is_valid:
		return {
			"is_valid": True,
			"full_number": "".join(result.digit for result in results),
			"error_message": None,
		}

	unreadable_index = next(
		index
		for index, result in enumerate(results, start=1)
		if not result.confidence >= confidence_threshold
	)
	return {
		"is_valid": False,
		"full_number": None,
		"error_message": f"Couldn't read digit {unreadable_index}, please redraw",
	}
