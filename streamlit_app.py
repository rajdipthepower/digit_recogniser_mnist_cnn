"""Streamlit interface for drawing and recognizing handwritten digit sequences."""

import base64
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas

from src.model import load_model
from src.preprocessing import prepare_mnist_crop
from src.segmentation import extract_digit_blobs
from src.utils import PredictionResult, digits_to_words, validate_predictions


def set_background(image_path: str) -> None:
	"""Encode a local image as a CSS data URL for the browser background."""
	img_file = Path(image_path)
	if not img_file.is_file():
		st.warning(f"Background image '{image_path}' not found.")
		return

	# CSS runs in the browser, so embed the local file instead of referencing a server path.
	with img_file.open("rb") as file:
		encoded = base64.b64encode(file.read()).decode()

	css = f"""
	<style>
	.stApp {{
		background: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)),
					url("data:image/jpeg;base64,{encoded}");
		background-size: cover;
		background-position: center;
		background-repeat: no-repeat;
		background-attachment: fixed;
	}}

	.stApp, .stApp * {{
		color: #FFFFFF !important;
	}}

	.stApp button[data-testid="stBaseButton-secondary"],
	.stApp button[data-testid="stBaseButton-secondary"] * {{
		color: #000000 !important;
	}}

	iframe[title="streamlit_drawable_canvas.drawable_canvas"] {{
		border: 2px solid rgba(255, 255, 255, 0.3) !important;
		border-radius: 8px !important;
	}}

	section[data-testid="stSidebar"] {{
		background-color: rgba(20, 20, 20, 0.85) !important;
	}}
	</style>
	"""
	st.markdown(css, unsafe_allow_html=True)


st.set_page_config(page_title="Handwritten Multi-Digit Recognizer", layout="centered")
set_background("Nature.jpg")
st.title("Handwritten Multi-Digit Recognizer")
st.info(
	"**Drawing tips:** Leave a visible gap between digits, keep them a similar size, "
	"and connect all strokes within each digit."
)

model = load_model("models/mnist_cnn_weights.pt")

tool = st.radio("Drawing tool", ("Draw", "Eraser"), horizontal=True)
stroke_color = "#FFFFFF" if tool == "Draw" else "#000000"

if "canvas_version" not in st.session_state:
	st.session_state.canvas_version = 0

# Changing this key remounts the canvas component with a clean drawing after Clear Canvas.
canvas_result = st_canvas(
	fill_color="rgba(255, 255, 255, 1)",
	stroke_width=12,
	stroke_color=stroke_color,
	background_color="#000000",
	width=600,
	height=200,
	drawing_mode="freedraw",
	return_image_data=True,
	key=f"digit_canvas_{st.session_state.canvas_version}",
)

predict_col, clear_col = st.columns(2)
predict_clicked = predict_col.button("Predict", type="primary", use_container_width=True)
clear_clicked = clear_col.button("Clear Canvas", use_container_width=True)

if "prediction_output" not in st.session_state:
	st.session_state.prediction_output = None

if clear_clicked:
	st.session_state.canvas_version += 1
	st.session_state.prediction_output = None
	st.rerun()

canvas_data = canvas_result.image_data
prediction_output = st.session_state.prediction_output
if (
	prediction_output is not None
		and prediction_output.get("canvas_data") is not None
	and canvas_data is not None
	and not np.array_equal(prediction_output["canvas_data"], canvas_data)
):
	st.session_state.prediction_output = None
	prediction_output = None

if predict_clicked:
	if canvas_data is None:
		st.session_state.prediction_output = {
			"status": "empty",
			"canvas_data": None,
		}
	else:
		blobs = extract_digit_blobs(canvas_data)
		if not blobs:
			st.session_state.prediction_output = {
				"status": "no_blobs",
				"detected_count": 0,
				"canvas_data": canvas_data.copy(),
			}
		elif any(blob.is_too_wide for blob in blobs):
			st.session_state.prediction_output = {
				"status": "too_wide",
				"detected_count": len(blobs),
				"canvas_data": canvas_data.copy(),
			}
		else:
			predictions = []
			for blob in blobs:
				image_tensor = prepare_mnist_crop(blob.image)
				input_tensor = torch.from_numpy(image_tensor)
				with torch.no_grad():
					probabilities = torch.softmax(model(input_tensor), dim=1)
					confidence_tensor, digit_tensor = probabilities.max(dim=1)

				confidence = float(confidence_tensor.item())
				digit = str(int(digit_tensor.item())) if confidence >= 0.95 else "?"
				predictions.append(
					PredictionResult(
						digit=digit,
						confidence=confidence,
						image_tensor=image_tensor,
					)
				)

			st.session_state.prediction_output = {
				"status": "results",
				"detected_count": len(blobs),
				"predictions": predictions,
				"validation": validate_predictions(
					predictions, confidence_threshold=0.95
				),
				"canvas_data": canvas_data.copy(),
			}

prediction_output = st.session_state.prediction_output
if prediction_output is not None:
	if prediction_output["status"] == "empty":
		st.warning("Please draw digits before predicting.")
	else:
		st.metric("Detected digits", prediction_output["detected_count"])
		if prediction_output["status"] == "no_blobs":
			st.warning("Please draw digits before predicting.")
		elif prediction_output["status"] == "too_wide":
			st.warning("Leave a gap between digits / Avoid drawing tiny digits or of different sizes")
		else:
			predictions = prediction_output["predictions"]
			for column, prediction in zip(st.columns(len(predictions)), predictions):
				with column:
					st.image(prediction.image_tensor[0, 0], width=80)
					st.markdown(f"**Prediction: {prediction.digit}**")
					st.caption(f"Confidence: {prediction.confidence:.1%}")

			validation = prediction_output["validation"]
			if not validation["is_valid"]:
				st.error(validation["error_message"])
			else:
				# Display the combined number only after every digit passes the confidence gate.
				full_number = validation["full_number"]
				st.header(full_number)
				st.write(digits_to_words(full_number))

st.markdown(
	'<div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #FFFFFF;">'
	'Made by Rajdip <span style="color: #FF3333;">&#10084;&#65039;&#10084;&#65039;</span>'
	"</div>",
	unsafe_allow_html=True,
)
