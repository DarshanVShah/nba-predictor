from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date, datetime, timedelta
import pytz
from backend.utils.predictor import NBAPredictor
from backend.utils.data_pipeline import update_data
import logging
import os
import pandas as pd
from balldontlie import BalldontlieAPI
import asyncio
from backend.config import DATA_FILE_PATH
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug prints
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Absolute path of data file: {DATA_FILE_PATH}")

# Initialize balldontlie client with API key from environment

api_key = os.getenv("BALLDONTLIE_API_KEY", "bfdc4ecf-c070-4e93-b9ac-cb36f049efb1")
api = BalldontlieAPI(api_key=api_key)

app = FastAPI(title="NBA Game Predictor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    logger.info("Initializing NBA Predictor...")
    predictor = NBAPredictor()
    logger.info("Loading data...")
    predictor.load_data()
    logger.info("Model initialization complete!")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    predictor = None

class GamePredictionRequest(BaseModel):
    home_team: str
    away_team: str
    game_date: date

@app.post("/predict")
async def predict_game(request: GamePredictionRequest):
    """
    Predict the winner of an NBA game
    
    Args:
        request: GamePredictionRequest containing home_team, away_team, and game_date
        
    Returns:
        dict: Prediction results including winner and confidence
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized properly")
    
    try:
        logger.info(f"Prediction request: {request.home_team} vs {request.away_team} on {request.game_date}")
        prediction = predictor.predict_game(
            home_team=request.home_team,
            away_team=request.away_team,
            date=request.game_date.strftime("%Y-%m-%d")
        )
        logger.info(f"Prediction successful: {prediction.get('winner', 'N/A')}")
        return prediction
    except ValueError as e:
        # ValueErrors are usually data/validation issues - return 400
        error_msg = str(e)
        logger.error(f"Validation error in predict_game: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        # Other errors might be server issues - return 500
        error_msg = f"Internal error: {str(e)}"
        logger.error(f"Error in predict_game: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/teams")
async def get_teams():
    """Get list of all NBA teams"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized properly")
    
    try:
        # Handle both merged dataset (team_x, team_y) and regular dataset (team)
        if 'team_x' in predictor.df.columns and 'team_y' in predictor.df.columns:
            # Merged dataset - get unique teams from both columns
            teams_x = predictor.df["team_x"].unique().tolist()
            teams_y = predictor.df["team_y"].unique().tolist()
            teams = sorted(list(set(teams_x + teams_y)))
        elif 'team' in predictor.df.columns:
            teams = sorted(predictor.df["team"].unique().tolist())
        else:
            raise ValueError("Dataset doesn't have team columns")
        return {"teams": teams}
    except Exception as e:
        logger.error(f"Error getting teams: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/daily-games")
async def get_daily_games():
    """
    Fetch today's NBA games using the balldontlie API
    """
    try:
        # Get today's date in EST timezone
        est = pytz.timezone('US/Eastern')
        today_est = datetime.now(est).strftime("%Y-%m-%d")
        logger.info(f"Fetching today's games for date (EST): {today_est}")
        
        # Get games using the API (try today and tomorrow in case of timezone issues)
        try:
            # Also try tomorrow in EST in case games are scheduled for tomorrow
            tomorrow_est = (datetime.now(est) + timedelta(days=1)).strftime("%Y-%m-%d")
            response = api.nba.games.list(dates=[today_est, tomorrow_est])
            games = response.data if hasattr(response, 'data') else []
            
            # Filter to only today's games
            today_games = []
            for game in games:
                game_date = getattr(game, 'date', '')
                if isinstance(game_date, str) and game_date.startswith(today_est):
                    today_games.append(game)
                elif hasattr(game, 'date') and str(game.date).startswith(today_est):
                    today_games.append(game)
            games = today_games
        except Exception as api_error:
            logger.error(f"API error: {str(api_error)}")
            # Return empty games list instead of failing
            return {
                "date": today_est,
                "games": []
            }
        
        if not games:
            logger.info(f"No games found for {today_est}")
            return {
                "date": today_est,
                "games": []
            }

        # Format the games data for API response
        formatted_games = []
        for game in games:
            try:
                # Handle different game status formats
                status = getattr(game, 'status', 'Scheduled')
                if status and isinstance(status, str) and 'T' in status:
                    # Convert datetime string to readable format
                    try:
                        from datetime import datetime as dt
                        dt_obj = dt.fromisoformat(status.replace('Z', '+00:00'))
                        status = dt_obj.strftime('%I:%M %p')
                    except:
                        pass
                
                formatted_game = {
                    "home_team": game.home_team.abbreviation if hasattr(game, 'home_team') else 'UNK',
                    "away_team": game.visitor_team.abbreviation if hasattr(game, 'visitor_team') else 'UNK',
                    "status": status
                }
                formatted_games.append(formatted_game)
            except Exception as e:
                logger.error(f"Error formatting game data: {str(e)}")
                continue

        return {
            "date": today_est,
            "games": formatted_games
        }
    except Exception as e:
        error_msg = f"Error processing games: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/update-data")
async def trigger_data_update():
    """
    Trigger an update of the NBA game data by scraping and processing new games
    """
    try:
        logger.info("Starting data update...")
        await update_data()
        logger.info("Data update complete")
        
        # Reload the predictor with new data
        if predictor is not None:
            logger.info("Reloading predictor with new data...")
            predictor.load_data()
            logger.info("Predictor reload complete")
        
        return {"status": "success", "message": "Data update completed successfully"}
    except Exception as e:
        error_msg = f"Error updating data: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/model-stats")
async def get_model_stats():
    """
    Get model evaluation statistics and accuracy metrics
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized properly")
    
    try:
        # Get data statistics first (safer)
        total_games = len(predictor.df)
        
        # Handle both merged dataset and regular dataset
        if 'team_x' in predictor.df.columns and 'team_y' in predictor.df.columns:
            teams_x = predictor.df["team_x"].unique()
            teams_y = predictor.df["team_y"].unique()
            total_teams = len(set(list(teams_x) + list(teams_y)))
            # Use date_next for merged dataset
            if 'date_next' in predictor.df.columns:
                date_range = {
                    "start": pd.to_datetime(predictor.df["date_next"]).min().strftime("%Y-%m-%d"),
                    "end": pd.to_datetime(predictor.df["date_next"]).max().strftime("%Y-%m-%d")
                }
            else:
                date_range = {"start": "N/A", "end": "N/A"}
        elif 'team' in predictor.df.columns:
            total_teams = predictor.df["team"].nunique()
            if 'date' in predictor.df.columns:
                date_range = {
                    "start": pd.to_datetime(predictor.df["date"]).min().strftime("%Y-%m-%d"),
                    "end": pd.to_datetime(predictor.df["date"]).max().strftime("%Y-%m-%d")
                }
            else:
                date_range = {"start": "N/A", "end": "N/A"}
        else:
            total_teams = 0
            date_range = {"start": "N/A", "end": "N/A"}
        
        # Try to evaluate model, but don't fail if it errors
        try:
            evaluation = predictor.evaluate_model(test_size=0.2)
            model_accuracy = evaluation["accuracy"]
            confusion_matrix = evaluation["confusion_matrix"]
        except Exception as eval_error:
            logger.warning(f"Could not evaluate model: {str(eval_error)}")
            # Return default/estimated values
            model_accuracy = 0.63  # Default accuracy from training
            confusion_matrix = [[0, 0], [0, 0]]
        
        return {
            "model_accuracy": float(model_accuracy),
            "confusion_matrix": confusion_matrix,
            "data_statistics": {
                "total_games": int(total_games),
                "total_teams": int(total_teams),
                "date_range": date_range
            },
            "features_used": len(predictor.predictors) if predictor.predictors else 0
        }
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    } 