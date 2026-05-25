# Breast Cancer Prediction API
### FastAPI + Machine Learning + Jenkins

---

## Project Structure

```
breast-cancer-api/
├── train_model.py      ← Step 1: Train model & create model.pkl
├── main.py             ← Step 2: FastAPI application
├── requirements.txt    ← Python dependencies
├── Jenkinsfile         ← Step 3: Jenkins CI/CD pipeline
├── model.pkl           ← Auto-generated after running train_model.py
└── README.md
```

---

## Dataset Info

- **Name:** Breast Cancer Wisconsin Dataset
- **Source:** Built into scikit-learn (`load_breast_cancer()`)
- **Samples:** 569
- **Features:** 30 tumor measurements
- **Target:** 0 = Malignant (cancerous), 1 = Benign (non-cancerous)

---

## Step-by-Step Setup

### Step 1 – Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Train the Model
```bash
python train_model.py
```
Expected output:
```
Model Accuracy: 96.49%
Model saved as model.pkl
```

### Step 3 – Run FastAPI App
```bash
uvicorn main:app --reload
```

### Step 4 – Test on Swagger UI
Open browser → **http://127.0.0.1:8000/docs**

Click `/predict` → **Try it out** → paste this sample input:

```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001.0,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871,
  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.006193,
  "worst_radius": 25.38,
  "worst_texture": 17.33,
  "worst_perimeter": 184.6,
  "worst_area": 2019.0,
  "worst_smoothness": 0.1622,
  "worst_compactness": 0.6656,
  "worst_concavity": 0.7119,
  "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601,
  "worst_fractal_dimension": 0.1189
}
```

Expected Response:
```json
{
  "predicted_class": 0,
  "diagnosis": "Malignant"
}
```

---

## Jenkins Setup

1. Open Jenkins → **New Item** → **Pipeline** → name it `breast-cancer-pipeline`
2. Scroll to **Pipeline** section → choose **Pipeline script**
3. Paste the full content of `Jenkinsfile`
4. Click **Save** → **Build Now**
5. Jenkins will run all 5 stages automatically

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict Malignant or Benign |
| GET | `/docs` | Swagger UI |
