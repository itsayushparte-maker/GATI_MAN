"""
GATI-MAN.AI - Backend API (FastAPI)
======================================
Post-Run Performance Dashboard - Advanced Athletic Analytics

Endpoints:
- POST /api/predict: Analyze post-run performance and compare against Expected Pace
- GET /: Serve the main application dashboard

Made by GATI-MAN.AI Team
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import math
from datetime import datetime
from scipy.stats import norm

# ============================================================================
# APPLICATION SETUP
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="GATI-MAN.AI API",
    description="Post-Run Performance Dashboard with AI-powered analytics",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Mount static files directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL LOADING
# ============================================================================

model = None

def load_model():
    """
    Load the trained Random Forest Regressor model.
    """
    global model
    
    model_path = "fitpredict_regressor.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found! "
            "Please ensure the regressor model is available."
        )
    
    try:
        model = joblib.load(model_path)
        print("=" * 60)
        print("✓ GATI-MAN.AI Regressor model loaded successfully!")
        print(f"✓ Model file: {model_path}")
        print("=" * 60)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class PredictionInput(BaseModel):
    """Input schema for post-run analytics request."""
    distance_km: float = Field(..., gt=0, le=100, description="Actual distance covered in kilometers")
    time_taken_minutes: float = Field(..., gt=0, le=1440, description="Actual time taken in minutes")
    elevation_gain: float = Field(..., ge=0, le=5000, description="Route terrain difficulty (elevation gain)")
    days_since_last_run: float = Field(..., ge=0, le=30, description="Fatigue level (days since last run)")
    hour_of_day: float = Field(..., ge=0, le=23, description="Time of day (hour)")

class PredictionOutput(BaseModel):
    """Output schema for post-run performance analytics."""
    actual_time: str
    actual_pace: str
    expected_pace: str
    performance_analysis: str
    calories_burned: float
    global_ranking: str
    projected_10k: str
    defense_eligibility: str
    timestamp: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_time(minutes: float) -> str:
    """Convert decimal minutes to MM:SS format."""
    total_seconds = int(round(minutes * 60))
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{mins:02d}:{secs:02d}"

def calculate_global_ranking(actual_pace: float) -> str:
    """
    Calculate user's percentile based on global pace distribution.
    Global average: 6.5 min/km, std_dev: 1.2 min/km.
    Lower pace = better percentile.
    """
    mean = 6.5
    std_dev = 1.2
    
    # Standard normal cumulative distribution function
    # Percentile = 0.5 * (1 + erf((x - mean) / (std_dev * sqrt(2))))
    z = (actual_pace - mean) / (std_dev * math.sqrt(2))
    percentile = 0.5 * (1 + math.erf(z)) * 100
    
    # Since lower pace is better, we use the calculated percentile directly for ranking
    # e.g., if you are faster than 90% of people, you are in the top 10%
    return f"Top {max(0.1, round(percentile, 1))}%"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load ML model on startup."""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application HTML page."""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "app": "GATI-MAN.AI",
        "version": "3.0.0",
        "status": "running"
    }

@app.post("/api/predict", response_model=PredictionOutput)
async def predict(prediction_input: PredictionInput):
    """
    Analyze post-run performance and compare against Expected Pace.
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Calculate Actual Metrics
        actual_pace_decimal = prediction_input.time_taken_minutes / prediction_input.distance_km
        
        # 2. Prepare features for Expected Pace prediction
        features = np.array([[
            prediction_input.distance_km,
            prediction_input.elevation_gain,
            prediction_input.days_since_last_run,
            prediction_input.hour_of_day
        ]])
        
        # 3. Predict Expected Pace using the trained model
        expected_pace_decimal = model.predict(features)[0]
        
        # 4. Calculate expected time
        expected_time_decimal = expected_pace_decimal * prediction_input.distance_km
        
        # 5. Performance Analysis
        time_diff = expected_time_decimal - prediction_input.time_taken_minutes
        if time_diff > 0:
            analysis = f"Outstanding! You were {abs(time_diff):.1f} mins FASTER than your Expected Pace."
        else:
            analysis = f"You were {abs(time_diff):.1f} mins slower than your Expected Pace."
            
        # 6. Calories Burned (Distance * 65)
        calories_burned = prediction_input.distance_km * 65
        
        # 7. Global Ranking (Based on actual pace)
        global_ranking = calculate_global_ranking(actual_pace_decimal)
        
        # 8. Projected 10K (Actual pace + 5% fatigue penalty)
        projected_10k_pace = actual_pace_decimal * 1.05
        projected_10k_time = projected_10k_pace * 10.0
        
        # 9. Defense & Forces Eligibility (Strict 25-min 5K cutoff)
        projected_5k_time = actual_pace_decimal * 5.0
        if projected_5k_time <= 25.0:
            defense_status = f"Passes Defense & Forces PET (Projected 5K: {format_time(projected_5k_time)})"
        else:
            defense_status = f"Fails Defense & Forces PET (Projected 5K: {format_time(projected_5k_time)})"
        
        # Return comprehensive analytics
        return PredictionOutput(
            actual_time=format_time(prediction_input.time_taken_minutes),
            actual_pace=format_time(actual_pace_decimal),
            expected_pace=format_time(expected_pace_decimal),
            performance_analysis=analysis,
            calories_burned=round(calories_burned, 1),
            global_ranking=global_ranking,
            projected_10k=format_time(projected_10k_time),
            defense_eligibility=defense_status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("GATI-MAN.AI - Post-Run Performance Dashboard")
    print("=" * 60)
    print("Starting FastAPI server...")
    print("API Documentation: http://localhost:8000/api/docs")
    print("Frontend: http://localhost:8000")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
