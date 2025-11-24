"""
Prediction Storage Utility
Stores predictions and retrieves them for history tracking
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions.db")

def init_database():
    """Initialize the predictions database"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                game_date TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                confidence REAL NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                actual_winner TEXT,
                actual_score_home INTEGER,
                actual_score_away INTEGER,
                is_correct INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_game_date 
            ON predictions(game_date)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_timestamp 
            ON predictions(prediction_timestamp)
        """)
        
        conn.commit()
        conn.close()
        logger.info("Predictions database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def save_prediction(
    home_team: str,
    away_team: str,
    game_date: str,
    predicted_winner: str,
    confidence: float
) -> int:
    """
    Save a prediction to the database
    
    Returns:
        int: Prediction ID
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        prediction_timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO predictions 
            (home_team, away_team, game_date, predicted_winner, confidence, prediction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (home_team, away_team, game_date, predicted_winner, confidence, prediction_timestamp))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Saved prediction {prediction_id}: {home_team} vs {away_team} on {game_date}")
        return prediction_id
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        raise

def update_prediction_result(
    prediction_id: int,
    actual_winner: str,
    actual_score_home: Optional[int] = None,
    actual_score_away: Optional[int] = None
):
    """Update a prediction with actual game results"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get the original prediction
        cursor.execute("SELECT predicted_winner FROM predictions WHERE id = ?", (prediction_id,))
        result = cursor.fetchone()
        if not result:
            logger.warning(f"Prediction {prediction_id} not found")
            conn.close()
            return
        
        predicted_winner = result[0]
        is_correct = 1 if predicted_winner == actual_winner else 0
        
        cursor.execute("""
            UPDATE predictions 
            SET actual_winner = ?,
                actual_score_home = ?,
                actual_score_away = ?,
                is_correct = ?
            WHERE id = ?
        """, (actual_winner, actual_score_home, actual_score_away, is_correct, prediction_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated prediction {prediction_id}: {'Correct' if is_correct else 'Incorrect'}")
    except Exception as e:
        logger.error(f"Error updating prediction result: {str(e)}")
        raise

def get_prediction_history(
    limit: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_future: bool = True
) -> List[Dict]:
    """
    Get prediction history
    
    Args:
        limit: Maximum number of predictions to return
        start_date: Filter by start date (YYYY-MM-DD)
        end_date: Filter by end date (YYYY-MM-DD)
        include_future: Include predictions for future games
    
    Returns:
        List of prediction dictionaries
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND game_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND game_date <= ?"
            params.append(end_date)
        
        if not include_future:
            query += " AND actual_winner IS NOT NULL"
        
        query += " ORDER BY game_date DESC, prediction_timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        predictions = []
        for row in rows:
            predictions.append({
                'id': row['id'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'game_date': row['game_date'],
                'predicted_winner': row['predicted_winner'],
                'confidence': row['confidence'],
                'prediction_timestamp': row['prediction_timestamp'],
                'actual_winner': row['actual_winner'],
                'actual_score_home': row['actual_score_home'],
                'actual_score_away': row['actual_score_away'],
                'is_correct': bool(row['is_correct']) if row['is_correct'] is not None else None
            })
        
        conn.close()
        return predictions
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise

def get_prediction_stats() -> Dict:
    """Get overall prediction statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        # Completed predictions (have actual results)
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_winner IS NOT NULL")
        completed = cursor.fetchone()[0]
        
        # Correct predictions
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_correct = 1")
        correct = cursor.fetchone()[0]
        
        # Accuracy
        accuracy = (correct / completed * 100) if completed > 0 else 0
        
        # Recent accuracy (last 50 completed predictions)
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT * FROM predictions 
                WHERE actual_winner IS NOT NULL 
                ORDER BY game_date DESC, prediction_timestamp DESC 
                LIMIT 50
            ) WHERE is_correct = 1
        """)
        recent_correct = cursor.fetchone()[0]
        recent_accuracy = (recent_correct / min(50, completed) * 100) if completed > 0 else 0
        
        conn.close()
        
        return {
            'total_predictions': total,
            'completed_predictions': completed,
            'correct_predictions': correct,
            'accuracy': round(accuracy, 2),
            'recent_accuracy': round(recent_accuracy, 2)
        }
    except Exception as e:
        logger.error(f"Error getting prediction stats: {str(e)}")
        return {
            'total_predictions': 0,
            'completed_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0,
            'recent_accuracy': 0
        }

# Initialize database on import
init_database()

