"""
Script to improve model accuracy by experimenting with different algorithms
and hyperparameters. This can be run to retrain the model with better settings.
"""
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from backend.config import DATA_FILE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load and prepare data for training."""
    logger.info("Loading data...")
    df = pd.read_csv(DATA_FILE_PATH)
    
    # Sort by date
    df = df.sort_values("date")
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Add target variable
    def add_target(team):
        team["target"] = team["won"].shift(-1)
        return team
    df = df.groupby("team", group_keys=False).apply(add_target)
    
    # Handle missing values
    df.loc[pd.isnull(df["target"]), "target"] = 2
    df["target"] = df["target"].astype(int, errors="ignore")
    
    # Add features
    df['home_court_advantage'] = df['home'].astype(int)
    df['rest_days'] = df.groupby('team')['date'].diff().dt.days.fillna(0)
    df['win_pct_last_10'] = df.groupby('team')['won'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    
    if 'season' in df.columns:
        df['win_pct_season'] = df.groupby(['team', 'season'])['won'].transform(lambda x: x.expanding().mean())
    else:
        df['win_pct_season'] = df.groupby('team')['won'].transform(lambda x: x.expanding().mean())
    
    # Momentum score
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    df['momentum_score'] = df.groupby('team')['won'].rolling(5, min_periods=1).apply(
        lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)])
    ).reset_index(0, drop=True)
    
    # Opponent strength
    df['opp_strength'] = df.groupby('team_opp')['won'].transform('mean')
    
    # Load saved predictors
    models_dir = os.path.join(os.getcwd(), "backend", "models")
    predictors_path = os.path.join(models_dir, "predictors.pkl")
    
    if os.path.exists(predictors_path):
        predictors = joblib.load(predictors_path)
        logger.info(f"Loaded {len(predictors)} predictors from saved file")
    else:
        # Use default predictors if file doesn't exist
        base_predictors = [
            'fg%', '3p', 'trb', 'efg%', 'ast%', 'usg%', 'ortg',
            'fga_max', '3pa_max', 'ft_max', 'orb_max', 'gmsc_max',
            'ftr_max', 'stl%_max', 'blk%_max', 'fg%_opp', 'ast_opp',
            'pts_opp', 'ts%_opp', 'efg%_opp', 'blk%_opp', 'usg%_opp',
            'drtg_opp', 'fg%_max_opp', 'stl_max_opp', 'tov_max_opp',
            'gmsc_max_opp', 'drb%_max_opp', 'ast%_max_opp', 'total_opp'
        ]
        predictors = base_predictors + [
            'home_court_advantage', 'rest_days', 'win_pct_last_10',
            'win_pct_season', 'momentum_score', 'opp_strength'
        ]
        logger.info(f"Using default {len(predictors)} predictors")
    
    # Filter to only valid predictions (target != 2)
    df = df[df['target'] != 2].copy()
    
    # Ensure all predictors exist
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        logger.warning(f"Missing predictors: {missing}")
        predictors = [p for p in predictors if p in df.columns]
    
    # Remove rows with missing values in predictors
    df = df.dropna(subset=predictors)
    
    X = df[predictors]
    y = df['target']
    
    logger.info(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, predictors, df

def evaluate_models(X, y, predictors):
    """Evaluate different models and return the best one."""
    models = {
        'Ridge Classifier (alpha=1)': RidgeClassifier(alpha=1),
        'Ridge Classifier (alpha=0.5)': RidgeClassifier(alpha=0.5),
        'Ridge Classifier (alpha=2)': RidgeClassifier(alpha=2),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        try:
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            results[name] = {
                'model': model,
                'mean_accuracy': mean_score,
                'std_accuracy': std_score,
                'scores': scores
            }
            logger.info(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
    
    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
    best_result = results[best_name]
    
    logger.info(f"\nBest model: {best_name} with accuracy {best_result['mean_accuracy']:.4f}")
    
    return best_result['model'], scaler, best_name

def train_and_save_best_model():
    """Train the best model and save it."""
    X, y, predictors, df = load_and_prepare_data()
    
    # Evaluate models
    best_model, scaler, model_name = evaluate_models(X, y, predictors)
    
    # Train on full data
    logger.info("Training best model on full dataset...")
    X_scaled = scaler.transform(X)
    best_model.fit(X_scaled, y)
    
    # Final evaluation
    y_pred = best_model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    logger.info(f"\nFinal Model: {model_name}")
    logger.info(f"Training Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y, y_pred))
    
    # Save model
    models_dir = os.path.join(os.getcwd(), "backend", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "nba_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    predictors_path = os.path.join(models_dir, "predictors.pkl")
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(predictors, predictors_path)
    
    logger.info(f"\nModel saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Predictors saved to {predictors_path}")
    
    return best_model, scaler, predictors

if __name__ == "__main__":
    train_and_save_best_model()

