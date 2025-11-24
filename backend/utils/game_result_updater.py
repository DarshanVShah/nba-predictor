"""
Game Result Updater
Automatically updates predictions with actual game results
"""

import logging
from datetime import datetime, timedelta
import pytz
from balldontlie import BalldontlieAPI
import os
from dotenv import load_dotenv
from backend.utils.prediction_storage import get_prediction_history, update_prediction_result

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize API
api_key = os.getenv("BALLDONTLIE_API_KEY", "bfdc4ecf-c070-4e93-b9ac-cb36f049efb1")
api = BalldontlieAPI(api_key=api_key)

def update_predictions_with_results():
    """
    Fetch completed games from the last 7 days and update predictions
    """
    try:
        logger.info("Starting prediction result update...")
        
        # Get predictions from last 7 days that don't have results yet
        est = pytz.timezone('US/Eastern')
        today = datetime.now(est).date()
        start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        
        # Get predictions without results
        predictions = get_prediction_history(
            limit=500,
            start_date=start_date,
            end_date=end_date,
            include_future=False  # Only get predictions for past games
        )
        
        # Filter to only predictions without actual results
        incomplete_predictions = [p for p in predictions if p['actual_winner'] is None]
        
        if not incomplete_predictions:
            logger.info("No incomplete predictions to update")
            return {"updated": 0, "total": 0}
        
        logger.info(f"Found {len(incomplete_predictions)} incomplete predictions to update")
        
        # Fetch games from API for the date range
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        
        updated_count = 0
        error_count = 0
        
        # Fetch games for each date
        all_games = []
        for date_str in dates:
            try:
                # Get games from API
                response = api.nba.games.list(dates=[date_str])
                # Handle both dict response and object response
                if isinstance(response, dict):
                    games = response.get('data', [])
                elif hasattr(response, 'data'):
                    games = response.data
                else:
                    games = []
                all_games.extend(games)
            except Exception as e:
                logger.error(f"Error fetching games for {date_str}: {str(e)}")
                continue
        
        games = all_games
        
        # Create a lookup dictionary: (home_team, away_team, date) -> game result
            game_results = {}
            for game in games:
                try:
                    # Check if game is final - handle both object attributes and dict keys
                    if isinstance(game, dict):
                        status = game.get('status')
                    else:
                        status = getattr(game, 'status', None)
                    
                    if status != 'Final':
                        continue
                    
                    # Get teams - handle both dict and object formats
                    if isinstance(game, dict):
                        home_team_obj = game.get('home_team', {})
                        visitor_team_obj = game.get('visitor_team', {})
                        home_team = home_team_obj.get('abbreviation') if isinstance(home_team_obj, dict) else None
                        away_team = visitor_team_obj.get('abbreviation') if isinstance(visitor_team_obj, dict) else None
                        game_date = game.get('date')
                        home_score = game.get('home_team_score')
                        visitor_score = game.get('visitor_team_score')
                    else:
                        # Object format
                        if hasattr(game, 'home_team'):
                            home_team_obj = game.home_team
                            if hasattr(home_team_obj, 'abbreviation'):
                                home_team = home_team_obj.abbreviation
                            elif isinstance(home_team_obj, dict):
                                home_team = home_team_obj.get('abbreviation')
                            else:
                                continue
                        else:
                            continue
                        
                        if hasattr(game, 'visitor_team'):
                            visitor_team_obj = game.visitor_team
                            if hasattr(visitor_team_obj, 'abbreviation'):
                                away_team = visitor_team_obj.abbreviation
                            elif isinstance(visitor_team_obj, dict):
                                away_team = visitor_team_obj.get('abbreviation')
                            else:
                                continue
                        else:
                            continue
                        
                        game_date = getattr(game, 'date', None)
                        home_score = getattr(game, 'home_team_score', None)
                        visitor_score = getattr(game, 'visitor_team_score', None)
                    
                    if not home_team or not away_team or not game_date:
                        continue
                    
                    # Parse date
                    if isinstance(game_date, str):
                        if 'T' in game_date:
                            try:
                                game_date_obj = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                            except:
                                game_date_obj = datetime.strptime(game_date.split('T')[0], "%Y-%m-%d")
                        else:
                            game_date_obj = datetime.strptime(game_date, "%Y-%m-%d")
                    else:
                        game_date_obj = game_date
                    
                    date_str = game_date_obj.strftime("%Y-%m-%d")
                    
                    # Determine winner
                    if home_score is not None and visitor_score is not None:
                        actual_winner = home_team if home_score > visitor_score else away_team
                        
                        # Store both team orders for lookup
                        game_results[(home_team, away_team, date_str)] = {
                            'winner': actual_winner,
                            'home_score': int(home_score),
                            'away_score': int(visitor_score)
                        }
                        game_results[(away_team, home_team, date_str)] = {
                            'winner': actual_winner,
                            'home_score': int(home_score),
                            'away_score': int(visitor_score)
                        }
                        logger.debug(f"Added game result: {home_team} {home_score} - {away_team} {visitor_score} on {date_str}")
                    else:
                        logger.warning(f"Game {home_team} vs {away_team} on {date_str} is Final but scores not available")
                except Exception as e:
                    logger.warning(f"Error processing game result: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
            
            logger.info(f"Found {len(game_results)} completed games from API")
            
            # Update predictions
            for pred in incomplete_predictions:
                try:
                    key = (pred['home_team'], pred['away_team'], pred['game_date'])
                    reverse_key = (pred['away_team'], pred['home_team'], pred['game_date'])
                    
                    result = game_results.get(key) or game_results.get(reverse_key)
                    
                    if result:
                        # Determine which score is home/away
                        if key in game_results:
                            # Original order
                            home_score = result['home_score']
                            away_score = result['away_score']
                        else:
                            # Reversed order
                            home_score = result['away_score']
                            away_score = result['home_score']
                        
                        update_prediction_result(
                            prediction_id=pred['id'],
                            actual_winner=result['winner'],
                            actual_score_home=home_score,
                            actual_score_away=away_score
                        )
                        updated_count += 1
                        logger.info(f"Updated prediction {pred['id']}: {pred['home_team']} vs {pred['away_team']} - Winner: {result['winner']}")
                    else:
                        logger.debug(f"No result found for prediction {pred['id']}: {pred['home_team']} vs {pred['away_team']} on {pred['game_date']}")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error updating prediction {pred['id']}: {str(e)}")
            
            logger.info(f"Update complete: {updated_count} updated, {error_count} errors")
            return {
                "updated": updated_count,
                "total": len(incomplete_predictions),
                "errors": error_count
            }
            
        except Exception as api_error:
            logger.error(f"Error fetching games from API: {str(api_error)}")
            return {
                "updated": 0,
                "total": len(incomplete_predictions),
                "errors": len(incomplete_predictions),
                "error": str(api_error)
            }
            
    except Exception as e:
        logger.error(f"Error in update_predictions_with_results: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "updated": 0,
            "total": 0,
            "errors": 1,
            "error": str(e)
        }

