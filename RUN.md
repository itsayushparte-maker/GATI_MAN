# FitPredict AI - Setup and Execution Guide

## Overview
FitPredict AI is a full-stack Machine Learning web application that predicts athletic performance using a Random Forest Classifier trained on Strava running data.

---

## Step 1: Install Python Dependencies

Open your terminal and navigate to the project directory, then run:

```bash
pip install -r requirements.txt
```

This will install all required libraries:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pandas (data manipulation)
- NumPy (numerical computing)
- Scikit-Learn (machine learning)
- Joblib (model serialization)

---

## Step 2: Train the Machine Learning Model

Run the training script to process the dataset and train the Random Forest model:

```bash
python train_model.py
```

**What this does:**
- Loads `extended_strava_data.csv`
- Engineers the target variable `Goal_Achieved` (success = pace <= 6.50 min/km)
- Splits data into training (80%) and testing (20%) sets
- Trains a Random Forest Classifier with 100 trees
- Evaluates model accuracy and displays metrics
- Saves the trained model to `fitpredict_model.pkl`

**Expected Output:**
- Model accuracy percentage
- Classification report (precision, recall, F1-score)
- Feature importance rankings
- Confirmation that model was saved

---

## Step 3: Start the FastAPI Backend Server

Launch the backend API server:

```bash
python main.py
```

**What this does:**
- Loads the trained model from `fitpredict_model.pkl`
- Starts a FastAPI server on `http://localhost:8000`
- Enables CORS for frontend communication
- Provides API documentation at `http://localhost:8000/api/docs`
- Enables hot reload for development

**Expected Output:**
- "FitPredict AI model loaded successfully!"
- Server startup confirmation
- API documentation URLs

---

## Step 4: Open the Frontend

Open `index.html` in your web browser:

```bash
# On macOS
open index.html

# On Linux
xdg-open index.html

# On Windows
start index.html
```

Alternatively, you can simply double-click the `index.html` file in your file explorer.

---

## Step 5: Make Predictions

1. Enter your run parameters in the frontend form:
   - **Target Distance (km)**: e.g., 5.0, 10.0, 15.0
   - **Route Elevation (m)**: e.g., 50, 120, 200
   - **Rest Days**: Number of days since last run (0-30)
   - **Run Time (Hour)**: Hour of day (0-23)

2. Click **"Predict Success Probability"**

3. View your results:
   - **Green (70%+)**: High likelihood of achieving your goal
   - **Orange (40-70%)**: Moderate chance of success
   - **Red (<40%)**: Low probability, consider adjusting your parameters

---

## Troubleshooting

### Error: "Model file not found"
**Solution**: Run `python train_model.py` first to generate `fitpredict_model.pkl`

### Error: "Connection refused" when making predictions
**Solution**: Ensure the FastAPI server is running (`python main.py`)

### Port 8000 already in use
**Solution**: Either:
- Kill the existing process: `lsof -ti:8000 | xargs kill` (macOS/Linux)
- Edit `main.py` and change `port=8000` to another port (e.g., `port=8001`)

### Module import errors
**Solution**: Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

---

## API Endpoints

### Health Check
```
GET http://localhost:8000/
```

### Prediction Endpoint
```
POST http://localhost:8000/api/predict
Content-Type: application/json

{
  "distance_km": 5.0,
  "elevation_gain": 50.0,
  "days_since_last_run": 2,
  "hour_of_day": 7
}
```

---

## Project Structure

```
fitpredict/
├── extended_strava_data.csv    # Training dataset
├── train_model.py              # ML model training script
├── fitpredict_model.pkl        # Trained model (generated after training)
├── main.py                     # FastAPI backend server
├── index.html                  # Frontend UI
├── requirements.txt            # Python dependencies
└── RUN.md                      # This file
```

---

## Technology Stack

**Backend & ML:**
- Python 3.8+
- FastAPI (web framework)
- Scikit-Learn (Random Forest Classifier)
- Pandas (data processing)
- Joblib (model serialization)

**Frontend:**
- HTML5, CSS3, Vanilla JavaScript
- Tailwind CSS (via CDN)
- Fetch API (HTTP requests)

---

## Made by FitPredict AI Team

This project demonstrates practical application of machine learning in athletic performance prediction.
