# main.py
# FastAPI Application with Breast Cancer ML Model

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Predicts whether a tumor is Malignant or Benign using a trained ML model",
    version="1.0.0"
)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Label mapping
LABELS = {
    0: "Malignant",
    1: "Benign"
}

# ---------------------------------------------------------------
# Input Schema — only 5 most important features (kept simple)
# ---------------------------------------------------------------
class TumorInput(BaseModel):
    mean_radius: float           # e.g. 17.99
    mean_texture: float          # e.g. 10.38
    mean_perimeter: float        # e.g. 122.8
    mean_area: float             # e.g. 1001.0
    mean_smoothness: float       # e.g. 0.1184
    mean_compactness: float      # e.g. 0.2776
    mean_concavity: float        # e.g. 0.3001
    mean_concave_points: float   # e.g. 0.1471
    mean_symmetry: float         # e.g. 0.2419
    mean_fractal_dimension: float # e.g. 0.07871
    radius_se: float             # e.g. 1.095
    texture_se: float            # e.g. 0.9053
    perimeter_se: float          # e.g. 8.589
    area_se: float               # e.g. 153.4
    smoothness_se: float         # e.g. 0.006399
    compactness_se: float        # e.g. 0.04904
    concavity_se: float          # e.g. 0.05373
    concave_points_se: float     # e.g. 0.01587
    symmetry_se: float           # e.g. 0.03003
    fractal_dimension_se: float  # e.g. 0.006193
    worst_radius: float          # e.g. 25.38
    worst_texture: float         # e.g. 17.33
    worst_perimeter: float       # e.g. 184.6
    worst_area: float            # e.g. 2019.0
    worst_smoothness: float      # e.g. 0.1622
    worst_compactness: float     # e.g. 0.6656
    worst_concavity: float       # e.g. 0.7119
    worst_concave_points: float  # e.g. 0.2654
    worst_symmetry: float        # e.g. 0.4601
    worst_fractal_dimension: float # e.g. 0.1189

# Output Schema
class PredictionOutput(BaseModel):
    predicted_class: int
    diagnosis: str


# Root endpoint
@app.get("/")
def root():
    return {"message": "Breast Cancer Prediction API is running!"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(data: TumorInput):
    # Prepare input array (30 features in correct order)
    input_array = np.array([[
        data.mean_radius, data.mean_texture, data.mean_perimeter,
        data.mean_area, data.mean_smoothness, data.mean_compactness,
        data.mean_concavity, data.mean_concave_points, data.mean_symmetry,
        data.mean_fractal_dimension, data.radius_se, data.texture_se,
        data.perimeter_se, data.area_se, data.smoothness_se,
        data.compactness_se, data.concavity_se, data.concave_points_se,
        data.symmetry_se, data.fractal_dimension_se, data.worst_radius,
        data.worst_texture, data.worst_perimeter, data.worst_area,
        data.worst_smoothness, data.worst_compactness, data.worst_concavity,
        data.worst_concave_points, data.worst_symmetry,
        data.worst_fractal_dimension
    ]])

    # Predict
    prediction = model.predict(input_array)[0]
    diagnosis = LABELS[int(prediction)]

    return PredictionOutput(
        predicted_class=int(prediction),
        diagnosis=diagnosis
    )
