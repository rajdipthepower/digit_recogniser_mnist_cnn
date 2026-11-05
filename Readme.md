Multi-Digit Handwritten Number Recognizer

A web application that segments, preprocesses, and classifies multi-digit
handwritten sequences from an interactive canvas using a PyTorch CNN.

Live Demo 

App Screenshot 

How It Works

1.  Canvas Ingestion: Extracts drawing canvas data, handles RGB/RGBA conversion,
    and normalizes foreground ink to white on a black background.
2.  Segmentation: Applies Otsu's thresholding and 8-connectivity connected
    component analysis (cv2.connectedComponentsWithStats) to detect individual
    digit blobs.
3.  Filtering & Sorting: Discards noise blobs (area < 50 px, height < 24 px) and
    sorts remaining digits left-to-right by their horizontal bounding box
    coordinate.
4.  MNIST Preprocessing: Pads each blob to a square, resizes it to 20x20
    (cv2.INTER_AREA), embeds it in a 28x28 black canvas, and translates it using
    scipy.ndimage.center_of_mass so the intensity center aligns with
    (14.0, 14.0).
5.  Inference & Safety Rules:
      - Blobs with an aspect ratio width / height > 1.6 trigger a warning:
        "Leave a gap between digits".
      - Each crop is evaluated by the CNN. If any digit has a softmax confidence
        below 95%, it is flagged as ?, and the full number output is suppressed.
6.  Output: If all digits pass the threshold, digits are joined into the final
    number and verbalized using num2words.

Model and Results

The classification model is a convolutional neural network trained on the
standard MNIST dataset (~99.2% test accuracy) as part of an MIT edX course.

  - Architecture: Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2) -> Conv2d(32, 64, 3)
    -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(1600, 128) -> Dropout(0.5) ->
    Linear(128, 10)

Evaluation & Failure Analysis

| Confusion Matrix                               | Misclassified Digits                                   | Failure Cases                            |
| :--------------------------------------------: | :----------------------------------------------------: | :--------------------------------------: |
| ![Confusion Matrix](docs/confusion_matrix.png) | ![Misclassified Digits](docs/misclassified.png) | ![Failure Cases](docs/failure_cases.png) |

Repository Structure

├── app.py              # Streamlit entry point and UI layout
├── src/
│   ├── model.py        # PyTorch model definition and cached loading
│   ├── segmentation.py # Blob extraction, noise filtering, and ordering
│   ├── preprocessing.py# Center-of-mass alignment and MNIST formatting
│   └── utils.py        # Confidence validation and number-to-words logic
├── models/             # Contains mnist_cnn_weights.pt
├── docs/               # Evaluation charts, confusion matrices, and failure cases
└── requirements.txt    # Pinned dependencies (CPU-compatible)

Run Locally

1.  Clone the repository:

    git clone https://github.com/your-username/mnist-multidigit-recognizer.git
    cd mnist-multidigit-recognizer

2.  Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3.  Install dependencies:

    pip install -r requirements.txt

4.  Launch the app:

    streamlit run app.py

Known Limitations

  - Touching/Overlapping Digits: Digits written without spacing merge into a
    single connected component and cannot be split.
  - Disconnected Strokes: Digits drawn in multiple disconnected parts (e.g., a
    detached top bar on a "5") may be segmented as separate blobs or filtered
    out as noise.
  - Scale Variance: Excessively small digits or extreme size differences across
    strokes can be discarded by the minimum area and height filters.

Tech Stack

  - Framework: Streamlit
  - Deep Learning: PyTorch (CPU-only)
  - Computer Vision & Image Processing: OpenCV (opencv-python-headless), SciPy,
    NumPy
  - Interactive Component: streamlit-drawable-canvas
  - Text Conversion: num2words
  - Hosting: Streamlit Community Cloud

Credits

  - MNIST Dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
  - CNN model training and architecture adapted from MIT edX Course (6.86x /
    Machine Learning with Python).
