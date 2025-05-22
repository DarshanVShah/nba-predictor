from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date, datetime
from backend.utils.predictor import NBAPredictor
from backend.utils.data_pipeline import update_data
import logging
import os
import pandas as pd
from balldontlie import BalldontlieAPI
import asyncio
from backend.config import DATA_FILE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug prints
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Absolute path of data file: {DATA_FILE_PATH}")

# Initialize balldontlie client with API key
api = BalldontlieAPI(api_key="bfdc4ecf-c070-4e93-b9ac-cb36f049efb1")

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
        prediction = predictor.predict_game(
            home_team=request.home_team,
            away_team=request.away_team,
            date=request.game_date.strftime("%Y-%m-%d")
        )
        return prediction
    except Exception as e:
        logger.error(f"Error in predict_game: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/teams")
async def get_teams():
    """Get list of all NBA teams"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not initialized properly")
    
    try:
        teams = sorted(predictor.df["team"].unique().tolist())
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
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Fetching today's games for date: {today}")
        
        # Get games using the API
        response = api.nba.games.list(dates=[today])
        games = response.data
        
        if not games:
            logger.info(f"No games found for {today}")
            return {
                "date": today,
                "games": []
            }

        # Format the games data for API response
        formatted_games = []
        for game in games:
            try:
                formatted_game = {
                    "home_team": game.home_team.abbreviation,
                    "away_team": game.visitor_team.abbreviation,
                    "status": game.status
                }
                formatted_games.append(formatted_game)
            except Exception as e:
                logger.error(f"Error formatting game data: {str(e)}")
                continue

        return {
            "date": today,
            "games": formatted_games
        }
    except Exception as e:
        error_msg = f"Error processing games: {str(e)}"
        logger.error(error_msg)
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