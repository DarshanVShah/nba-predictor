import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from datetime import datetime
import pytz
from backend.config import DATA_FILE_PATH

# Set up logging
logger = logging.getLogger(__name__)

# Define the absolute path to the model file
MODEL_FILE_PATH = os.path.join(os.getcwd(), 'backend', 'models', 'nba_model.pkl')

class NBAPredictor:
    def __init__(self):
        logger.info("Initializing NBAPredictor...")
        self.model = None
        self.scaler = None
        self.predictors = None
        self.df = None  # This will be the merged dataset (full)
        self.raw_df = None  # Raw dataset for calculating rolling averages
        logger.info("NBAPredictor initialized successfully")
        
    def load_data(self):
        """Load the merged dataset from CSV file"""
        try:
            logger.info(f"Loading merged dataset from CSV...")
            logger.info(f"Data path: {DATA_FILE_PATH}")
            
            # Load the merged dataset (this is the 'full' dataset from the notebook)
            self.df = pd.read_csv(DATA_FILE_PATH)
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
            
            # Check if this is the merged dataset (has team_x, team_y columns)
            if 'team_x' not in self.df.columns or 'team_y' not in self.df.columns:
                logger.warning("⚠️ Dataset doesn't appear to be merged. Expected team_x and team_y columns.")
                logger.warning("⚠️ You need to run the Predict.ipynb notebook to create the merged dataset.")
                logger.warning("⚠️ Predictions may not work correctly until the merged dataset is created.")
            else:
                logger.info("✅ Merged dataset detected (has team_x and team_y columns)")
            
            # Always try to load raw data for future predictions (needed for rolling averages)
            raw_data_path = os.path.join(os.path.dirname(DATA_FILE_PATH), "processed_nba_data.csv")
            if os.path.exists(raw_data_path):
                logger.info("Loading raw dataset for rolling average calculations...")
                try:
                    self.raw_df = pd.read_csv(raw_data_path)
                    if 'date' in self.raw_df.columns:
                        self.raw_df['date'] = pd.to_datetime(self.raw_df['date'])
                        self.raw_df = self.raw_df.sort_values('date')
                    if 'team' in self.raw_df.columns:
                        logger.info(f"✅ Raw dataset loaded: {len(self.raw_df)} rows, {self.raw_df['team'].nunique()} teams")
                    else:
                        logger.warning("⚠️ Raw dataset loaded but missing 'team' column")
                except Exception as e:
                    logger.error(f"Error loading raw dataset: {str(e)}")
                    self.raw_df = None
            else:
                logger.warning(f"⚠️ Raw dataset not found at {raw_data_path}. Future predictions may not work correctly.")
                logger.warning("⚠️ Make sure to run cell 49 of Predict.ipynb to save processed_nba_data.csv")
            
            # Convert date columns to datetime if they exist
            if 'date_next' in self.df.columns:
                self.df['date_next'] = pd.to_datetime(self.df['date_next'])
            
            # Load the saved model, scaler, and predictors
            self.load_saved_components()
            
            # Verify that saved predictors exist in the dataframe
            if self.predictors:
                original_count = len(self.predictors)
                missing_predictors = [p for p in self.predictors if p not in self.df.columns]
                if missing_predictors:
                    logger.warning(f"Some saved predictors are missing from data: {missing_predictors[:10]}...")
                    # Filter out missing predictors
                    self.predictors = [p for p in self.predictors if p in self.df.columns]
                    logger.warning(f"Using {len(self.predictors)} available predictors out of {original_count}")
                    if len(self.predictors) < original_count * 0.8:  # Less than 80% of predictors
                        logger.error(f"Too many predictors missing! Only {len(self.predictors)}/{original_count} available.")
                        raise ValueError(f"Too many predictors missing from dataset. Expected {original_count}, found {len(self.predictors)}")
            
            logger.info("Data loading complete")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_saved_components(self):
        """Load the saved model, scaler, and predictors"""
        try:
            models_dir = os.path.join(os.getcwd(), "backend", "models")
            # Load model
            model_path = os.path.join(models_dir, "nba_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            logger.info("Loading saved model...")
            self.model = joblib.load(model_path)
            # Load scaler
            scaler_path = os.path.join(models_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            logger.info("Loading saved scaler...")
            self.scaler = joblib.load(scaler_path)
            # Load predictors
            predictors_path = os.path.join(models_dir, "predictors.pkl")
            if not os.path.exists(predictors_path):
                raise FileNotFoundError(f"Predictors file not found at {predictors_path}")
            logger.info("Loading saved predictors...")
            self.predictors = joblib.load(predictors_path)
            logger.info(f"Loaded {len(self.predictors)} predictors")
            logger.info("All components loaded successfully")
        except Exception as e:
            logger.error(f"Error loading saved components: {str(e)}")
            raise
        
    def predict_game(self, home_team, away_team, date):
        """
        Predict the winner of a game between two teams.
        Works for both historical and future games.
        
        Args:
            home_team (str): Home team abbreviation (e.g., 'LAL')
            away_team (str): Away team abbreviation (e.g., 'BOS')
            date (str): Game date in YYYY-MM-DD format
            
        Returns:
            dict: Prediction results including winner and confidence
        """
        try:
            # Validate inputs
            if not home_team or not away_team:
                raise ValueError("home_team and away_team must be provided")
            if home_team == away_team:
                raise ValueError("home_team and away_team cannot be the same")
            
            # Check if predictors are loaded
            if not self.predictors:
                raise ValueError("Predictors not loaded. Model may not be initialized correctly.")
            
            logger.info(f"Predicting game: {home_team} vs {away_team} on {date}")
            
            # Use EST timezone for date comparison
            est = pytz.timezone('US/Eastern')
            game_date = pd.to_datetime(date)
            today_est = pd.to_datetime(datetime.now(est).date())
            
            # Check if this is a future game (using EST)
            # Treat today's games as future if not in dataset (games scheduled for today)
            is_future_game = game_date >= today_est
            logger.info(f"Game date: {game_date}, Today (EST): {today_est}, Is future: {is_future_game}")
            
            # Try to find matching row in merged dataset
            # The merged dataset has team_x (home) and team_y (away) with date_next
            matching_row = None
            
            # Check if we have the merged dataset structure
            has_merged_structure = ('team_x' in self.df.columns and 
                                   'team_y' in self.df.columns and 
                                   'date_next' in self.df.columns)
            
            if has_merged_structure:
                # Convert date_next to datetime if it's not already
                if self.df['date_next'].dtype != 'datetime64[ns]':
                    self.df['date_next'] = pd.to_datetime(self.df['date_next'])
                
                # Look for exact match (convert to date for comparison to ignore time)
                matches = self.df[
                    (self.df['team_x'] == home_team) & 
                    (self.df['team_y'] == away_team) &
                    (self.df['date_next'].dt.date == game_date.date())
                ]
                
                if len(matches) > 0:
                    matching_row = matches.iloc[0]
                    logger.info("Found matching row in merged dataset")
                else:
                    # Try reverse (maybe team_x is away and team_y is home)
                    matches = self.df[
                        (self.df['team_x'] == away_team) & 
                        (self.df['team_y'] == home_team) &
                        (self.df['date_next'].dt.date == game_date.date())
                    ]
                    if len(matches) > 0:
                        matching_row = matches.iloc[0]
                        logger.info("Found matching row (reversed teams)")
            
            if matching_row is not None:
                # Use the matching row from merged dataset
                logger.info("Using historical data from merged dataset")
                try:
                    # Extract features from matching row
                    feature_dict = {}
                    for pred in self.predictors:
                        if pred in matching_row.index:
                            feature_dict[pred] = matching_row[pred]
                        else:
                            logger.warning(f"Feature {pred} not found in matching row, using 0.0")
                            feature_dict[pred] = 0.0
                    game_features = pd.DataFrame([feature_dict])
                except Exception as e:
                    logger.error(f"Error extracting features from matching row: {str(e)}")
                    logger.error(f"Available columns: {list(matching_row.index)[:20]}")
                    logger.error(f"Required predictors: {self.predictors[:10]}")
                    raise ValueError(f"Error extracting features: {str(e)}")
            elif is_future_game:
                # For future games, calculate features from raw data
                if self.raw_df is None:
                    error_msg = (
                        "Cannot make prediction for future game: Raw dataset not available. "
                        "Please ensure processed_nba_data.csv exists in backend/data/processed/. "
                        "Run cell 49 of Predict.ipynb to generate this file."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                logger.info("Future game - calculating features from raw data...")
                game_features = self._calculate_future_game_features(home_team, away_team, game_date)
            elif not has_merged_structure:
                # Dataset is not merged - cannot make predictions
                error_msg = (
                    "Dataset is not in merged format. "
                    "Please run the Predict.ipynb notebook to create the merged dataset. "
                    "The notebook should save 'full' DataFrame to final_dataset.csv"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif not is_future_game:
                # Historical game but not in merged dataset - try to find closest match
                logger.info("Historical game not in merged dataset - finding closest match...")
                if 'date_next' in self.df.columns:
                    # Convert date_next to datetime if needed
                    if self.df['date_next'].dtype != 'datetime64[ns]':
                        self.df['date_next'] = pd.to_datetime(self.df['date_next'])
                    
                    # Find closest date
                    date_diff = (self.df['date_next'] - game_date).abs()
                    closest_idx = date_diff.idxmin()
                    if date_diff.iloc[closest_idx] < pd.Timedelta(days=7):  # Within 7 days
                        matching_row = self.df.iloc[closest_idx]
                        # Check if teams match
                        if ((matching_row['team_x'] == home_team and matching_row['team_y'] == away_team) or
                            (matching_row['team_x'] == away_team and matching_row['team_y'] == home_team)):
                            game_features = pd.DataFrame([matching_row[self.predictors]])
                            logger.info("Using closest matching row")
                        else:
                            # Teams don't match - treat as future game if raw_df available
                            logger.info("Teams don't match, treating as future game...")
                            if self.raw_df is not None:
                                game_features = self._calculate_future_game_features(home_team, away_team, game_date)
                            else:
                                raise ValueError(f"No matching game found for {home_team} vs {away_team} on {date}")
                    else:
                        # Date too far - treat as future game if raw_df available
                        logger.info("Date too far, treating as future game...")
                        if self.raw_df is not None:
                            game_features = self._calculate_future_game_features(home_team, away_team, game_date)
                        else:
                            raise ValueError(f"No matching game found for {home_team} vs {away_team} on {date}")
                else:
                    # No date_next column - treat as future game if raw_df available
                    if self.raw_df is not None:
                        game_features = self._calculate_future_game_features(home_team, away_team, game_date)
                    else:
                        raise ValueError(f"No matching game found and cannot calculate features")
            else:
                # This should not happen, but provide a helpful error
                error_msg = (
                    f"Cannot make prediction: No matching data found for {home_team} vs {away_team} on {date}. "
                    f"Game is {'future' if is_future_game else 'historical'}. "
                    f"Raw dataset {'available' if self.raw_df is not None else 'not available'}."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Handle missing features
            missing_features = []
            for feature in self.predictors:
                if feature not in game_features.columns:
                    logger.warning(f"Feature {feature} missing, adding with value 0.0")
                    game_features[feature] = 0.0
                    missing_features.append(feature)
                else:
                    game_features[feature] = game_features[feature].fillna(0.0)
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}")
            
            # Reorder columns to match training data EXACTLY as the scaler expects
            try:
                # Ensure all predictors are present
                missing = set(self.predictors) - set(game_features.columns)
                if missing:
                    logger.error(f"Missing {len(missing)} required features: {list(missing)[:10]}")
                    raise ValueError(f"Missing required features: {list(missing)[:10]}")
                
                # Reorder to match exact order of predictors (critical for scaler)
                game_features = game_features[self.predictors]
                
                # Verify order matches
                if list(game_features.columns) != self.predictors:
                    logger.error(f"Feature order mismatch!")
                    logger.error(f"Expected order: {self.predictors[:5]}...")
                    logger.error(f"Actual order: {list(game_features.columns)[:5]}...")
                    # Force correct order
                    game_features = game_features[self.predictors]
                    
            except KeyError as e:
                logger.error(f"Error reordering features: {str(e)}")
                logger.error(f"Game features columns ({len(game_features.columns)}): {list(game_features.columns)[:20]}")
                logger.error(f"Required predictors ({len(self.predictors)}): {self.predictors[:20]}")
                missing = set(self.predictors) - set(game_features.columns)
                raise ValueError(f"Missing required features: {list(missing)[:10]}")
            
            # Convert to numeric and handle any remaining issues
            for col in game_features.columns:
                game_features[col] = pd.to_numeric(game_features[col], errors='coerce').fillna(0.0)
            
            # Scale features
            # CRITICAL: The scaler was fit on original dataset (132 features like "3p", "3p%"),
            # but model uses merged dataset (30 predictors like "fga_10_x", "usg%_10_x").
            # We need to use a scaler that was fit on the merged dataset.
            # For now, try to bypass feature name checking by using numpy array.
            try:
                # Convert to numpy array - this might bypass feature name validation in some sklearn versions
                game_features_array = game_features.values.reshape(1, -1)
                
                # Check if scaler expects different number of features
                if hasattr(self.scaler, 'feature_names_in_') and self.scaler.feature_names_in_ is not None:
                    expected_features = len(self.scaler.feature_names_in_)
                    actual_features = game_features_array.shape[1]
                    
                    if expected_features != actual_features:
                        raise ValueError(
                            f"Scaler expects {expected_features} features but got {actual_features}. "
                            f"The scaler was fit on the original dataset, but the model uses the merged dataset. "
                            f"Please re-run Predict.ipynb and ensure the scaler is fit on 'full[selected_columns]' "
                            f"AFTER merging (not on 'df[selected_columns]' before merging)."
                        )
                
                # Try scaling with numpy array
                game_features_scaled = self.scaler.transform(game_features_array)
                
                logger.info(f"Features scaled successfully: {game_features_scaled.shape}")
                
            except ValueError as e:
                # Re-raise ValueError with our message
                raise
            except Exception as e:
                error_msg = str(e)
                if "feature names" in error_msg.lower() or "unseen at fit time" in error_msg.lower():
                    raise ValueError(
                        "Scaler was fit on original dataset features (like '3p', '3p%'), "
                        "but model uses merged dataset features (like 'fga_10_x', 'usg%_10_x'). "
                        "Please re-run Predict.ipynb and fit the scaler on the merged dataset 'full' "
                        "AFTER feature selection, not on the original 'df' dataset."
                    )
                raise
                    
            except Exception as e:
                logger.error(f"Error scaling features: {str(e)}")
                logger.error(f"Feature shape: {game_features.shape}")
                logger.error(f"Feature columns ({len(game_features.columns)}): {list(game_features.columns)}")
                logger.error(f"Predictors ({len(self.predictors)}): {self.predictors}")
                
                # Check what the scaler expects
                if hasattr(self.scaler, 'feature_names_in_'):
                    scaler_feats = list(self.scaler.feature_names_in_) if self.scaler.feature_names_in_ is not None else []
                    logger.error(f"Scaler expects {len(scaler_feats)} features: {scaler_feats[:20]}")
                
                raise ValueError(f"Feature scaling error: {str(e)}")
            
            # Make prediction
            prediction = self.model.predict(game_features_scaled)[0]
            
            # Get proper probability/confidence - use best available method
            confidence = 0.5  # Default
            
            # Try predict_proba first (most accurate for most models)
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(game_features_scaled)[0]
                    # proba[0] is probability of class 0 (away wins), proba[1] is class 1 (home wins)
                    if prediction == 1:  # Home wins
                        confidence = proba[1]
                    else:  # Away wins
                        confidence = proba[0]
                    logger.info(f"Using predict_proba: confidence = {confidence:.3f}")
                except Exception as e:
                    logger.warning(f"predict_proba failed: {e}, falling back to decision_function")
            
            # Fallback to decision_function if predict_proba not available
            if confidence == 0.5 and hasattr(self.model, 'decision_function'):
                try:
                    decision_score = self.model.decision_function(game_features_scaled)[0]
                    # Convert decision function to probability using sigmoid
                    confidence = 1 / (1 + np.exp(-decision_score))
                    # Adjust confidence to be more conservative (less extreme)
                    # This helps with calibration - decision scores can be extreme
                    if confidence > 0.5:
                        confidence = 0.5 + (confidence - 0.5) * 0.85
                    else:
                        confidence = 0.5 - (0.5 - confidence) * 0.85
                    logger.info(f"Using decision_function: confidence = {confidence:.3f}")
                except Exception as e:
                    logger.warning(f"decision_function failed: {e}")
            
            # Ensure confidence is in reasonable range
            confidence = max(0.51, min(0.95, confidence))
            
            winner = home_team if prediction == 1 else away_team
            
            logger.info(f"Prediction complete: {winner} wins with confidence {confidence:.2%}")
            
            return {
                "winner": winner,
                "confidence": float(confidence),
                "home_team": home_team,
                "away_team": away_team,
                "date": date
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _calculate_future_game_features(self, home_team, away_team, game_date):
        """
        Calculate features for a future game by computing rolling averages
        similar to how the notebook does it.
        """
        try:
            logger.info(f"Calculating features for future game: {home_team} vs {away_team} on {game_date}")
            
            # Team name mapping (handle variations)
            team_mapping = {
                'BKN': 'BRK',  # Brooklyn Nets
                'BRK': 'BRK',
                'PHO': 'PHO',  # Phoenix Suns (dataset uses PHO)
                'PHX': 'PHO',  # Phoenix Suns (API uses PHX)
                'CHA': 'CHO',  # Charlotte Hornets (API uses CHA)
                'CHO': 'CHO',  # Charlotte Hornets (dataset uses CHO)
            }
            
            # Normalize team names
            home_team_normalized = team_mapping.get(home_team, home_team)
            away_team_normalized = team_mapping.get(away_team, away_team)
            
            # Get home team's most recent games (up to game_date)
            # Try both original and normalized names
            home_games = self.raw_df[
                ((self.raw_df['team'] == home_team) | (self.raw_df['team'] == home_team_normalized)) & 
                (self.raw_df['date'] < game_date)
            ].tail(10)
            
            # Get away team's most recent games (up to game_date)
            # Try both original and normalized names
            away_games = self.raw_df[
                ((self.raw_df['team'] == away_team) | (self.raw_df['team'] == away_team_normalized)) & 
                (self.raw_df['date'] < game_date)
            ].tail(10)
            
            if home_games.empty:
                # Check what teams are available
                available_teams = sorted(self.raw_df['team'].unique().tolist())
                raise ValueError(
                    f"Insufficient historical data for {home_team} (need at least 1 game before {game_date}). "
                    f"Available teams: {', '.join(available_teams[:10])}..."
                )
            if away_games.empty:
                # Check what teams are available
                available_teams = sorted(self.raw_df['team'].unique().tolist())
                raise ValueError(
                    f"Insufficient historical data for {away_team} (need at least 1 game before {game_date}). "
                    f"Available teams: {', '.join(available_teams[:10])}..."
                )
            
            # Calculate rolling averages (10-game window) for home team
            # Use weighted average - more recent games weighted higher
            try:
                if len(home_games) > 1:
                    home_weights = np.linspace(0.5, 1.0, len(home_games))
                    home_numeric = home_games.select_dtypes(include=[np.number])
                    if not home_numeric.empty:
                        # Reshape weights to multiply correctly
                        weights_reshaped = home_weights.reshape(-1, 1)
                        weighted_sum = (home_numeric * weights_reshaped).sum()
                        home_rolling = weighted_sum / home_weights.sum()
                    else:
                        home_rolling = home_games.mean(numeric_only=True)
                else:
                    home_rolling = home_games.mean(numeric_only=True)
            except Exception as e:
                logger.warning(f"Error calculating weighted average for home team: {e}, using simple mean")
                home_rolling = home_games.mean(numeric_only=True)
            
            # Calculate rolling averages (10-game window) for away team
            # Use weighted average - more recent games weighted higher
            try:
                if len(away_games) > 1:
                    away_weights = np.linspace(0.5, 1.0, len(away_games))
                    away_numeric = away_games.select_dtypes(include=[np.number])
                    if not away_numeric.empty:
                        # Reshape weights to multiply correctly
                        weights_reshaped = away_weights.reshape(-1, 1)
                        weighted_sum = (away_numeric * weights_reshaped).sum()
                        away_rolling = weighted_sum / away_weights.sum()
                    else:
                        away_rolling = away_games.mean(numeric_only=True)
                else:
                    away_rolling = away_games.mean(numeric_only=True)
            except Exception as e:
                logger.warning(f"Error calculating weighted average for away team: {e}, using simple mean")
                away_rolling = away_games.mean(numeric_only=True)
            
            # Construct features matching the merged dataset structure
            # Predictors can be:
            # - Simple: fga, usg%, fg_max (from home team, no suffix)
            # - With _opp: fg_opp, usg%_opp (from away team, opponent stats)
            # - With _10_x: fga_10_x, usg%_10_x (home team rolling average)
            # - With _10_y: pts_10_y, usg%_10_y (away team rolling average)
            # - With _opp_10_x: fg_opp_10_x (home team's opponent stats rolling average)
            # - With _opp_10_y: fg_opp_10_y (away team's opponent stats rolling average)
            # - home_next: home court advantage
            
            # Calculate additional context features for better accuracy
            home_recent_form = home_games['won'].mean() if 'won' in home_games.columns else 0.5
            away_recent_form = away_games['won'].mean() if 'won' in away_games.columns else 0.5
            
            # Calculate momentum (trend in last 5 games vs previous 5)
            if len(home_games) >= 5:
                home_recent_5 = home_games.tail(5)['won'].mean() if 'won' in home_games.columns else 0.5
                home_prev_5 = home_games.head(5)['won'].mean() if len(home_games) >= 10 and 'won' in home_games.columns else home_recent_5
                home_momentum = home_recent_5 - home_prev_5
            else:
                home_momentum = 0.0
            
            if len(away_games) >= 5:
                away_recent_5 = away_games.tail(5)['won'].mean() if 'won' in away_games.columns else 0.5
                away_prev_5 = away_games.head(5)['won'].mean() if len(away_games) >= 10 and 'won' in away_games.columns else away_recent_5
                away_momentum = away_recent_5 - away_prev_5
            else:
                away_momentum = 0.0
            
            game_features = {}
            
            for feature in self.predictors:
                if feature == 'home_next':
                    # Home court advantage
                    game_features[feature] = 1.0
                elif feature.endswith('_opp_10_y'):
                    # Away team's opponent stats rolling average (e.g., fg_opp_10_y)
                    # This is the away team's opponent (which is the home team)
                    base_feature = feature[:-9]  # Remove _opp_10_y
                    if base_feature in home_rolling.index:
                        game_features[feature] = home_rolling[base_feature]
                    else:
                        logger.warning(f"Feature {base_feature} not found for {feature}")
                        game_features[feature] = 0.0
                elif feature.endswith('_opp_10_x'):
                    # Home team's opponent stats rolling average (e.g., fg_opp_10_x)
                    # This is the home team's opponent (which is the away team)
                    base_feature = feature[:-9]  # Remove _opp_10_x
                    if base_feature in away_rolling.index:
                        game_features[feature] = away_rolling[base_feature]
                    else:
                        logger.warning(f"Feature {base_feature} not found for {feature}")
                        game_features[feature] = 0.0
                elif feature.endswith('_10_y'):
                    # Away team rolling average (e.g., pts_10_y, usg%_10_y)
                    base_feature = feature[:-5]  # Remove _10_y
                    if base_feature in away_rolling.index:
                        game_features[feature] = away_rolling[base_feature]
                    else:
                        logger.warning(f"Feature {base_feature} not found in away team for {feature}")
                        game_features[feature] = 0.0
                elif feature.endswith('_10_x'):
                    # Home team rolling average (e.g., fga_10_x, usg%_10_x)
                    base_feature = feature[:-5]  # Remove _10_x
                    if base_feature in home_rolling.index:
                        game_features[feature] = home_rolling[base_feature]
                    else:
                        logger.warning(f"Feature {base_feature} not found in home team for {feature}")
                        game_features[feature] = 0.0
                elif feature.endswith('_opp'):
                    # Opponent stats (e.g., fg_opp, usg%_opp)
                    # For home team, opponent is away team
                    base_feature = feature[:-4]  # Remove _opp
                    if base_feature in away_rolling.index:
                        game_features[feature] = away_rolling[base_feature]
                    else:
                        logger.warning(f"Feature {base_feature} not found for opponent stats {feature}")
                        game_features[feature] = 0.0
                else:
                    # Simple feature without suffix (e.g., fga, usg%, fg_max)
                    # These come from the home team
                    if feature in home_rolling.index:
                        game_features[feature] = home_rolling[feature]
                    else:
                        # Try smart defaults based on feature name
                        if 'win_pct' in feature.lower() or 'win%' in feature.lower():
                            game_features[feature] = home_recent_form
                        elif 'momentum' in feature.lower():
                            game_features[feature] = home_momentum
                        else:
                            logger.warning(f"Feature {feature} not found in home team data, using 0.0")
                            game_features[feature] = 0.0
            
            # Ensure all predictors are present and in correct order
            for pred in self.predictors:
                if pred not in game_features:
                    # Fill missing predictors with smart defaults
                    if 'win_pct' in pred.lower() or 'win%' in pred.lower():
                        if '_x' in pred or 'home' in pred.lower():
                            game_features[pred] = home_recent_form
                        elif '_y' in pred or 'away' in pred.lower():
                            game_features[pred] = away_recent_form
                        else:
                            game_features[pred] = 0.5
                    elif 'momentum' in pred.lower():
                        if '_x' in pred or 'home' in pred.lower():
                            game_features[pred] = home_momentum
                        elif '_y' in pred or 'away' in pred.lower():
                            game_features[pred] = away_momentum
                        else:
                            game_features[pred] = 0.0
                    else:
                        game_features[pred] = 0.0
            
            logger.info(f"Calculated {len(game_features)} features for future game")
            game_features_df = pd.DataFrame([game_features])
            
            # Ensure correct column order
            game_features_df = game_features_df[self.predictors] if all(p in game_features_df.columns for p in self.predictors) else game_features_df
            
            return game_features_df
            
        except Exception as e:
            logger.error(f"Error calculating future game features: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def evaluate_model(self, test_size=0.2):
        """
        Evaluate the model's accuracy on historical data
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Evaluation metrics including accuracy and confusion matrix
        """
        try:
            logger.info("Evaluating model...")
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            from sklearn.model_selection import train_test_split
            
            if 'target' not in self.df.columns:
                raise ValueError("Target column not found in dataset")
            
            # Filter out rows where target == 2 (no next game)
            eval_df = self.df[self.df['target'] != 2].copy()
            
            if len(eval_df) == 0:
                raise ValueError("No valid evaluation data (all targets are 2)")
            
            # Split data into training and testing sets
            X = eval_df[self.predictors]
            y = eval_df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
            
            # Train model on training data
            self.model.fit(X_train, y_train)
            
            # Make predictions on test data
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            logger.info(f"Model evaluation complete. Accuracy: {accuracy:.2f}")
            
            return {
                "accuracy": float(accuracy),
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
