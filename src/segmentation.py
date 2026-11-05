"""Convert a drawn canvas into left-to-right digit candidate blobs."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DigitBlob:
	"""A cropped connected component and its canvas-space bounding box."""

	image: np.ndarray
	bbox: tuple[int, int, int, int]
	is_too_wide: bool


def _ensure_white_foreground(grayscale: np.ndarray) -> np.ndarray:
	"""Normalize polarity so ink is white and the background is black."""
	border = np.concatenate(
		(grayscale[0, :], grayscale[-1, :], grayscale[:, 0], grayscale[:, -1])
	)
	# Canvas background reaches the image border; invert only when that border is light.
	if np.median(border) > 127:
		return cv2.bitwise_not(grayscale)
	return grayscale


def extract_digit_blobs(
	canvas_data: np.ndarray,
	min_area: int = 50,
	wide_aspect_ratio: float = 1.6,
) -> list[DigitBlob]:
	"""Threshold a canvas, reject noise, and return candidate digits by x-position."""
	if canvas_data.ndim == 2:
		grayscale = _ensure_white_foreground(canvas_data)
	elif canvas_data.ndim == 3 and canvas_data.shape[2] == 4:
		alpha = canvas_data[:, :, 3]
		if np.any(alpha != alpha.flat[0]):
			grayscale = alpha
		else:
			grayscale = cv2.cvtColor(canvas_data[:, :, :3], cv2.COLOR_RGB2GRAY)
			grayscale = _ensure_white_foreground(grayscale)
	elif canvas_data.ndim == 3 and canvas_data.shape[2] == 3:
		grayscale = cv2.cvtColor(canvas_data, cv2.COLOR_RGB2GRAY)
		grayscale = _ensure_white_foreground(grayscale)
	else:
		raise ValueError("canvas_data must be a grayscale, RGB, or RGBA image")

	if grayscale.dtype != np.uint8:
		grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

	# Otsu adapts to each canvas's stroke intensity and antialiasing without a fixed cutoff.
	_, binary = cv2.threshold(
		grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
	)
	count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
	blobs = [
		(int(x), int(y), int(width), int(height), int(area))
		for x, y, width, height, area in stats[1:count]
		if area >= min_area
	]

	if not blobs:
		return []

	largest_height = max(height for _, _, _, height, _ in blobs)
	largest_area = max(area for _, _, _, _, area in blobs)
	# Require 24px of height; among multiple blobs, reject candidates below 35% of max height or 10% of max area.
	blobs = [
		blob
		for blob in blobs
		if blob[3] >= 24
		and (len(blobs) == 1 or (blob[3] >= 0.35 * largest_height and blob[4] >= 0.1 * largest_area))
	]
	if not blobs:
		return []

	median_width = float(np.median([width for _, _, width, _, _ in blobs]))
	result = []
	for x, y, width, height, _ in sorted(blobs, key=lambda blob: blob[0]):
		# A high width/height ratio or width over 1.8x the peer median can indicate touching digits.
		too_wide = (
			width / height > wide_aspect_ratio or width > 1.8 * median_width
		)
		result.append(
			DigitBlob(
				image=binary[y : y + height, x : x + width].copy(),
				bbox=(x, y, width, height),
				is_too_wide=too_wide,
			)
		)

	return result
