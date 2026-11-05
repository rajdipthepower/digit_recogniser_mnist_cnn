"""Prepare segmented digit crops in the input format expected by the MNIST model."""

import cv2
import numpy as np
from scipy.ndimage import center_of_mass


def shift_by_center_of_mass(img_28x28: np.ndarray) -> np.ndarray:
	"""Translate a 28x28 image so its ink-weighted center is at pixel (14, 14)."""
	image = np.asarray(img_28x28)
	if image.shape != (28, 28):
		raise ValueError("img_28x28 must have shape (28, 28)")

	if not np.any(image):
		return image.copy()

	# MNIST alignment follows ink center of mass, not the crop's bounding-box center.
	center_y, center_x = center_of_mass(image)
	shift_x = 14.0 - center_x
	shift_y = 14.0 - center_y
	transform = np.array(
		[[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32
	)
	return cv2.warpAffine(
		image,
		transform,
		(28, 28),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=0,
	)


def prepare_mnist_crop(blob_img: np.ndarray) -> np.ndarray:
	"""Scale a binary crop into a normalized, batched 28x28 MNIST input."""
	image = np.asarray(blob_img)
	if image.ndim != 2 or image.size == 0:
		raise ValueError("blob_img must be a non-empty 2D image")

	image = image.astype(np.float32)
	if image.max() > 1.0:
		image /= 255.0
	image = np.clip(image, 0.0, 1.0)

	height, width = image.shape
	side = max(height, width)
	# Square padding preserves the digit's aspect ratio during the 20x20 resize.
	top = (side - height) // 2
	bottom = side - height - top
	left = (side - width) // 2
	right = side - width - left
	square = cv2.copyMakeBorder(
		image,
		top,
		bottom,
		left,
		right,
		cv2.BORDER_CONSTANT,
		value=0,
	)

	# MNIST places a 20x20 glyph inside a 28x28 image with a four-pixel border.
	resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
	image_28x28 = np.zeros((28, 28), dtype=np.float32)
	image_28x28[4:24, 4:24] = resized
	shifted = shift_by_center_of_mass(image_28x28)
	normalized = np.clip(shifted, 0.0, 1.0).astype(np.float32)
	return normalized[np.newaxis, np.newaxis, :, :]
