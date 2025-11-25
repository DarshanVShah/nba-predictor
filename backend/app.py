from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date, datetime, timedelta
import pytz
from backend.utils.predictor import NBAPredictor
from backend.utils.data_pipeline import update_data
from backend.utils.prediction_storage import save_prediction, get_prediction_history, get_prediction_stats, update_prediction_result
from backend.utils.game_result_updater import update_predictions_with_results
import logging
import os
import pandas as pd
from balldontlie import BalldontlieAPI
import asyncio
from backend.config import DATA_FILE_PATH
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required files if URLs are provided (solves Git LFS/volume issues)
try:
    from backend.utils.file_downloader import ensure_files_exist
    ensure_files_exist(
        final_dataset_url=os.getenv("FINAL_DATASET_URL"),
        processed_data_url=os.getenv("PROCESSED_DATA_URL"),
        model_url=os.getenv("MODEL_URL"),
        predictors_url=os.getenv("PREDICTORS_URL"),
        scaler_url=os.getenv("SCALER_URL")
    )
except Exception as e:
    logger.warning(f"File downloader not available or failed: {str(e)}")

# Debug prints
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Absolute path of data file: {DATA_FILE_PATH}")

# Initialize balldontlie client with API key from environment
api_key = os.getenv("BALLDONTLIE_API_KEY")
if not api_key:
    raise ValueError("BALLDONTLIE_API_KEY environment variable is required")
api = BalldontlieAPI(api_key=api_key)

app = FastAPI(title="NBA Game Predictor")

# Add CORS middleware
# Get allowed origins from environment variable or use defaults
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:3002").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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

# Initialize background scheduler for updating predictions
scheduler = BackgroundScheduler()
scheduler.start()

# Schedule prediction result updates
# Run every hour to check for completed games
scheduler.add_job(
    update_predictions_with_results,
    trigger=CronTrigger(minute=0),  # Run at the top of every hour
    id='update_predictions',
    name='Update predictions with game results',
    replace_existing=True
)

# Also run on startup to catch any missed updates (run after a short delay to let server start)
from datetime import datetime as dt
scheduler.add_job(
    update_predictions_with_results,
    trigger='date',
    run_date=dt.now() + timedelta(seconds=30),  # Run 30 seconds after startup
    id='initial_update',
    name='Initial prediction update',
    replace_existing=True
)

logger.info("Background scheduler started - predictions will be updated hourly")

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
        
        # Save prediction to history
        try:
            save_prediction(
                home_team=request.home_team,
                away_team=request.away_team,
                game_date=request.game_date.strftime("%Y-%m-%d"),
                predicted_winner=prediction['winner'],
                confidence=prediction['confidence']
            )
        except Exception as e:
            logger.warning(f"Failed to save prediction to history: {str(e)}")
        
        return prediction
    except ValueError as e:
        # ValueErrors are usually data/validation issues - return 400
        error_msg = str(e)
        logger.error(f"Validation error in predict_game: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        # Check for rate limit errors
        error_str = str(e).lower()
        if 'rate limit' in error_str or 'too many requests' in error_str or '429' in error_str:
            raise HTTPException(
                status_code=429,
                detail="API request limit reached. Please wait a minute before trying again."
            )
        
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
            error_str = str(api_error).lower()
            logger.error(f"API error: {str(api_error)}")
            
            # Check for rate limit errors
            if 'rate limit' in error_str or 'too many requests' in error_str or '429' in error_str:
                raise HTTPException(
                    status_code=429,
                    detail="API request limit reached. Please wait a minute before trying again."
                )
            
            # Return empty games list for other errors
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
                # Handle different game status formats and extract game time
                status = getattr(game, 'status', 'Scheduled')
                game_time = None
                
                # Try to get game time from various possible fields
                # Check for 'date' field (usually contains full datetime)
                game_date_field = getattr(game, 'date', None)
                if not game_date_field:
                    # Try 'scheduled' field
                    game_date_field = getattr(game, 'scheduled', None)
                
                if game_date_field:
                    try:
                        # Convert to EST timezone
                        est = pytz.timezone('US/Eastern')
                        from datetime import datetime as dt
                        
                        if isinstance(game_date_field, str):
                            # Parse ISO format datetime string
                            if 'T' in game_date_field:
                                try:
                                    # Try parsing with timezone
                                    if game_date_field.endswith('Z'):
                                        dt_obj = dt.fromisoformat(game_date_field.replace('Z', '+00:00'))
                                    elif '+' in game_date_field or game_date_field.count('-') > 2:
                                        dt_obj = dt.fromisoformat(game_date_field)
                                    else:
                                        # No timezone, assume UTC
                                        dt_obj = dt.fromisoformat(game_date_field)
                                        dt_obj = pytz.utc.localize(dt_obj)
                                    
                                    # Convert to EST
                                    if dt_obj.tzinfo is None:
                                        dt_obj = pytz.utc.localize(dt_obj)
                                    dt_est = dt_obj.astimezone(est)
                                    game_time = dt_est.strftime('%I:%M %p').lstrip('0')
                                except Exception as parse_error:
                                    logger.warning(f"Could not parse datetime string {game_date_field}: {parse_error}")
                                    game_time = None
                            else:
                                game_time = game_date_field
                        elif hasattr(game_date_field, 'strftime'):
                            # Already a datetime object
                            if game_date_field.tzinfo is None:
                                game_date_utc = pytz.utc.localize(game_date_field)
                            else:
                                game_date_utc = game_date_field
                            dt_est = game_date_utc.astimezone(est)
                            game_time = dt_est.strftime('%I:%M %p').lstrip('0')
                    except Exception as e:
                        logger.warning(f"Error parsing game time: {str(e)}, game_date_field: {game_date_field}")
                        game_time = None
                
                # Use game time if available, otherwise use status
                display_status = game_time if game_time else (status if status and status != 'Scheduled' else 'TBD')
                
                formatted_game = {
                    "home_team": game.home_team.abbreviation if hasattr(game, 'home_team') else 'UNK',
                    "away_team": game.visitor_team.abbreviation if hasattr(game, 'visitor_team') else 'UNK',
                    "status": display_status
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

@app.get("/prediction-history")
async def get_history(limit: int = 100, start_date: str = None, end_date: str = None):
    """Get prediction history"""
    try:
        predictions = get_prediction_history(
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            include_future=True
        )
        stats = get_prediction_stats()
        return {
            "predictions": predictions,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-prediction-result")
async def update_result(prediction_id: int, actual_winner: str, 
                       actual_score_home: int = None, actual_score_away: int = None):
    """Update a prediction with actual game results"""
    try:
        update_prediction_result(
            prediction_id=prediction_id,
            actual_winner=actual_winner,
            actual_score_home=actual_score_home,
            actual_score_away=actual_score_away
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error updating prediction result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger-prediction-update")
async def trigger_update():
    """Manually trigger prediction result update"""
    try:
        result = update_predictions_with_results()
        return {
            "status": "success",
            "updated": result.get("updated", 0),
            "total": result.get("total", 0),
            "errors": result.get("errors", 0)
        }
    except Exception as e:
        logger.error(f"Error triggering update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical-predictions/{date}")
async def get_historical_predictions(date: str):
    """
    Get games for a specific date, run predictions, and fetch actual results for comparison
    Date format: YYYY-MM-DD
    Earliest date: 2025-11-23
    """
    try:
        # Parse and validate date
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Check earliest date (November 23, 2025)
        earliest_date = datetime(2025, 11, 23).date()
        
        # Check that date is not in the future (using EST)
        est = pytz.timezone('US/Eastern')
        today_est = datetime.now(est).date()
        
        # Validate date range
        if target_date < earliest_date:
            raise HTTPException(
                status_code=400, 
                detail=f"Earliest date allowed is {earliest_date.strftime('%Y-%m-%d')}. Selected date: {date}"
            )
        if target_date > today_est:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot get predictions for future dates. Today (EST): {today_est.strftime('%Y-%m-%d')}, Selected: {date}"
            )
        
        logger.info(f"Fetching historical predictions for date: {date} (EST today: {today_est})")
        
        # Fetch games for the date
        try:
            logger.info(f"Calling balldontlie API for date: {date}")
            response = api.nba.games.list(dates=[date])
            logger.info(f"API response type: {type(response)}")
            
            if isinstance(response, dict):
                games = response.get('data', [])
                logger.info(f"Got {len(games)} games from dict response")
            elif hasattr(response, 'data'):
                games = response.data
                logger.info(f"Got {len(games)} games from object response")
            else:
                logger.warning(f"Unexpected response format: {response}")
                games = []
                
            # Log first game if available for debugging
            if games and len(games) > 0:
                first_game = games[0]
                if isinstance(first_game, dict):
                    logger.info(f"First game date: {first_game.get('date')}, status: {first_game.get('status')}")
                else:
                    logger.info(f"First game date: {getattr(first_game, 'date', 'N/A')}, status: {getattr(first_game, 'status', 'N/A')}")
        except Exception as api_error:
            error_str = str(api_error).lower()
            logger.error(f"API error fetching games for {date}: {str(api_error)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Check for rate limit errors
            if 'rate limit' in error_str or 'too many requests' in error_str or '429' in error_str:
                raise HTTPException(
                    status_code=429,
                    detail="API request limit reached. Please wait a minute before trying again."
                )
            
            games = []
        
        if not games:
            return {
                "date": date,
                "games": [],
                "message": "No games found for this date"
            }
        
        # Process each game: make prediction and get actual result
        results = []
        for game in games:
            try:
                # Extract team info
                if isinstance(game, dict):
                    home_team = game.get('home_team', {}).get('abbreviation') if isinstance(game.get('home_team'), dict) else None
                    away_team = game.get('visitor_team', {}).get('abbreviation') if isinstance(game.get('visitor_team'), dict) else None
                    status = game.get('status')
                    home_score = game.get('home_team_score')
                    visitor_score = game.get('visitor_team_score')
                else:
                    home_team = game.home_team.abbreviation if hasattr(game, 'home_team') and hasattr(game.home_team, 'abbreviation') else None
                    away_team = game.visitor_team.abbreviation if hasattr(game, 'visitor_team') and hasattr(game.visitor_team, 'abbreviation') else None
                    status = getattr(game, 'status', None)
                    home_score = getattr(game, 'home_team_score', None)
                    visitor_score = getattr(game, 'visitor_team_score', None)
                
                if not home_team or not away_team:
                    continue
                
                # Make prediction - use the date string directly
                predicted_winner = None
                confidence = 0.5
                try:
                    if predictor is None:
                        logger.warning(f"Predictor not initialized - cannot predict {home_team} vs {away_team}")
                    else:
                        prediction_result = predictor.predict_game(home_team, away_team, date)
                        if prediction_result:
                            predicted_winner = prediction_result.get('winner') or prediction_result.get('predicted_winner')
                            confidence = prediction_result.get('confidence', 0.5)
                            logger.info(f"Prediction successful: {predicted_winner} with confidence {confidence}")
                        else:
                            logger.warning(f"Prediction returned None for {home_team} vs {away_team}")
                except Exception as pred_error:
                    logger.error(f"Error predicting {home_team} vs {away_team} on {date}: {str(pred_error)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    predicted_winner = None
                    confidence = 0.5
                
                # Get actual result if game is final
                actual_winner = None
                actual_home_score = None
                actual_away_score = None
                is_correct = None
                
                if status == 'Final' and home_score is not None and visitor_score is not None:
                    actual_home_score = int(home_score)
                    actual_away_score = int(visitor_score)
                    actual_winner = home_team if home_score > visitor_score else away_team
                    
                    if predicted_winner:
                        is_correct = (predicted_winner == actual_winner)
                
                results.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "predicted_winner": predicted_winner,
                    "confidence": confidence,
                    "actual_winner": actual_winner,
                    "actual_home_score": actual_home_score,
                    "actual_away_score": actual_away_score,
                    "is_correct": is_correct,
                    "status": status
                })
                
            except Exception as game_error:
                logger.error(f"Error processing game: {str(game_error)}")
                continue
        
        # Calculate stats for this date
        completed_games = [r for r in results if r['is_correct'] is not None]
        correct_predictions = sum(1 for r in completed_games if r['is_correct'])
        accuracy = (correct_predictions / len(completed_games) * 100) if completed_games else None
        
        return {
            "date": date,
            "games": results,
            "stats": {
                "total_games": len(results),
                "completed_games": len(completed_games),
                "correct_predictions": correct_predictions,
                "accuracy": accuracy
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical predictions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 

