"""  
FitPredict AI - Machine Learning Model Training Script
=======================================================
This script trains a Random Forest Regressor to predict the exact pace (min/km)
for athletic performance based on historical Strava data.

Features used:
- Distance (km)
- Elevation Gain
- Days Since Last Run
- Hour of Day

Target Variable:
- Pace (min/km) - Continuous regression target

Made by FitPredict AI Team
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """
    Load the CSV dataset and prepare features for regression.
    
    Parameters:
    -----------
    filepath : str
        Path to the extended_strava_data.csv file
    
    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with features and target variable
    """
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv(filepath)
    print(f"\n✓ Loaded {len(df)} records from {filepath}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display target variable statistics
    print(f"\nTarget variable (Pace) statistics:")
    print(f"  - Mean: {df['Pace (min/km)'].mean():.2f} min/km")
    print(f"  - Median: {df['Pace (min/km)'].median():.2f} min/km")
    print(f"  - Std Dev: {df['Pace (min/km)'].std():.2f} min/km")
    print(f"  - Min: {df['Pace (min/km)'].min():.2f} min/km")
    print(f"  - Max: {df['Pace (min/km)'].max():.2f} min/km")
    
    return df

def prepare_features(df):
    """
    Select and prepare input features for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with features and target variable
    
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector (Pace)
    """
    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)
    
    # Define input features
    feature_columns = [
        'Distance (km)',
        'Elevation Gain',
        'Days_Since_Last_Run',
        'Hour_of_Day'
    ]
    
    # Separate features and target
    X = df[feature_columns]
    y = df['Pace (min/km)']
    
    print(f"\n✓ Selected {len(feature_columns)} input features:")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\n✓ Target variable: Pace (min/km)")
    
    return X, y

def train_model(X, y):
    """
    Train a Random Forest Regressor model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector (Pace)
    
    Returns:
    --------
    model : RandomForestRegressor
        Trained model object
    """
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST REGRESSOR")
    print("=" * 60)
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n✓ Data split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=200,           # Number of trees in the forest
        max_depth=15,               # Maximum depth of each tree
        min_samples_split=5,        # Minimum samples required to split a node
        min_samples_leaf=2,         # Minimum samples required at a leaf node
        random_state=42,            # For reproducibility
        n_jobs=-1                   # Use all available CPU cores
    )
    
    print(f"\n✓ Model configuration:")
    print(f"  - Algorithm: Random Forest Regressor")
    print(f"  - Number of trees: 200")
    print(f"  - Maximum depth: 15")
    print(f"  - Random state: 42")
    
    # Train the model
    print(f"\n→ Training model...")
    model.fit(X_train, y_train)
    print(f"✓ Model training complete!")
    
    # Evaluate model performance
    print(f"\n→ Evaluating model performance...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'=' * 60}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n✓ Mean Absolute Error (MAE): {mae:.3f} min/km")
    print(f"✓ Root Mean Squared Error (RMSE): {rmse:.3f} min/km")
    print(f"✓ R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
    
    print(f"\nPrediction vs Actual Comparison:")
    for i in range(min(5, len(y_test))):
        print(f"  Sample {i+1}: Actual={y_test.iloc[i]:.2f}, Predicted={y_pred[i]:.2f}")
    
    # Display feature importance
    print(f"\n{'=' * 60}")
    print("FEATURE IMPORTANCE")
    print(f"{'=' * 60}")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:25s} : {row['importance']:.4f}")
    
    return model

def save_model(model, filepath):
    """
    Save the trained model to a file using joblib.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model object
    filepath : str
        Path to save the model file
    """
    print(f"\n{'=' * 60}")
    print("SAVING MODEL")
    print(f"{'=' * 60}")
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"\n✓ Model successfully saved to: {filepath}")
    print(f"✓ Model file size: {round(os.path.getsize(filepath) / 1024, 2)} KB")

def main():
    """
    Main function to orchestrate the entire training pipeline.
    """
    print("\n" + "=" * 60)
    print("FitPredict AI - Model Training Pipeline")
    print("=" * 60)
    print("Athletic Performance Prediction System")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data('extended_strava_data.csv')
    
    # Step 2: Prepare features
    X, y = prepare_features(df)
    
    # Step 3: Train the model
    model = train_model(X, y)
    
    # Step 4: Save the trained model
    save_model(model, 'fitpredict_regressor.pkl')
    
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\n✓ Model file: fitpredict_regressor.pkl")
    print(f"✓ Ready for deployment with FastAPI backend")
    print(f"\n{'=' * 60}\n")

if __name__ == "__main__":
    main()
