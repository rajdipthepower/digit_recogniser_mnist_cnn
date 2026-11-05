"""MNIST convolutional classifier and cached checkpoint loader."""

import streamlit as st
import torch
from torch import nn


class MNISTClassifier(nn.Module):
	"""Classify a 28x28 grayscale image as one of ten digits."""

	def __init__(self) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Flatten(),
			nn.Linear(1600, 128),
			nn.Dropout(0.5),
			nn.Linear(128, 10),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Return the ten class logits for each input image."""
		return self.layers(inputs)


@st.cache_resource
def load_model(weights_path: str) -> torch.nn.Module:
	"""Load a checkpoint on CPU and return the model in inference mode."""
	model = MNISTClassifier()
	state_dict = torch.load(weights_path, map_location="cpu")
	# Raw Sequential saves omit the wrapper's `layers.` prefix.
	if any(key.startswith("layers.") for key in state_dict):
		model.load_state_dict(state_dict)
	else:
		model.layers.load_state_dict(state_dict)
	model.eval()
	return model
