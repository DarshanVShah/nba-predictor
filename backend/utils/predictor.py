import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
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
        self.base_predictors = [
            'fg%', '3p', 'trb', 'efg%', 'ast%', 'usg%', 'ortg',
            'fga_max', '3pa_max', 'ft_max', 'orb_max', 'gmsc_max',
            'ftr_max', 'stl%_max', 'blk%_max', 'fg%_opp', 'ast_opp',
            'pts_opp', 'ts%_opp', 'efg%_opp', 'blk%_opp', 'usg%_opp',
            'drtg_opp', 'fg%_max_opp', 'stl_max_opp', 'tov_max_opp',
            'gmsc_max_opp', 'drb%_max_opp', 'ast%_max_opp', 'total_opp'
        ]
        self.predictors = self.base_predictors + [
            'home_court_advantage',
            'rest_days',
            'win_pct_last_10',
            'win_pct_season',
            'momentum_score',
            'opp_strength'
        ]
        logger.info("NBAPredictor initialized successfully")
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            logger.info(f"Loading data from CSV...")
            logger.info(f"Data path: {DATA_FILE_PATH}")
            self.df = pd.read_csv(DATA_FILE_PATH)
            logger.info("Data loaded successfully")
            
            # Verify required columns exist
            required_columns = ['team', 'date', 'won'] + self.base_predictors
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.df = self.df.sort_values("date")
            self.df = self.df.reset_index(drop=True)
            
            # Convert date to datetime
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Add target variable
            def add_target(team):
                team["target"] = team["won"].shift(-1)
                return team
            self.df = self.df.groupby("team", group_keys=False).apply(add_target)
            
            # Handle missing values using loc to avoid chained assignment warning
            self.df.loc[pd.isnull(self.df["target"]), "target"] = 2
            self.df["target"] = self.df["target"].astype(int, errors="ignore")
            
            # Add home court advantage
            self.df['home_court_advantage'] = self.df['home'].astype(int)
            
            # Add rest days
            self.df['rest_days'] = self.df.groupby('team')['date'].diff().dt.days.fillna(0)
            
            # Add win percentage features
            self.df['win_pct_last_10'] = self.df.groupby('team')['won'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
            self.df['win_pct_season'] = self.df.groupby(['team', 'season'])['won'].transform(lambda x: x.expanding().mean())
            
            # Add momentum score (weighted average of last 5 games)
            weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # More weight to recent games
            self.df['momentum_score'] = self.df.groupby('team')['won'].rolling(5, min_periods=1).apply(
                lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)])
            ).reset_index(0, drop=True)
            
            # Add opponent strength (opponent's win percentage)
            self.df['opp_strength'] = self.df.groupby('team_opp')['won'].transform('mean')
            
            # Load the saved model, scaler, and predictors
            self.load_saved_components()
            
            logger.info("Data processing complete")
            
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
            logger.info("All components loaded successfully")
        except Exception as e:
            logger.error(f"Error loading saved components: {str(e)}")
            raise

    def load_or_train_model(self):
        """Load existing model or train a new one if not found"""
        model_path = os.path.join("data", "models", "model.pkl")
        try:
            if os.path.exists(model_path):
                logger.info("Loading existing model...")
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.info("No existing model found. Training new model...")
                self.train()
                # Create models directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # Save the trained model
                joblib.dump(self.model, model_path)
                logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error in load_or_train_model: {str(e)}")
            raise
        
    def train(self):
        """Train the model on historical data"""
        try:
            logger.info("Training model...")
            self.model = RidgeClassifier(alpha=1)
            self.model.fit(self.df[self.predictors], self.df["target"])
            logger.info("Model training complete")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
        
    def predict_game(self, home_team, away_team, date):
        """
        Predict the winner of a game between two teams
        
        Args:
            home_team (str): Home team abbreviation (e.g., 'LAL')
            away_team (str): Away team abbreviation (e.g., 'BOS')
            date (str): Game date in YYYY-MM-DD format
            
        Returns:
            dict: Prediction results including winner and confidence
        """
        try:
            logger.info(f"Predicting game: {home_team} vs {away_team} on {date}")
            
            # Get the most recent game data for both teams
            home_data = self.df[self.df['team'] == home_team].copy()
            away_data = self.df[self.df['team'] == away_team].copy()
            
            if home_data.empty:
                raise ValueError(f"No data found for home team: {home_team}")
            if away_data.empty:
                raise ValueError(f"No data found for away team: {away_team}")
            
            # Get the most recent game for each team
            home_data = home_data.iloc[-1]
            away_data = away_data.iloc[-1]
            
            # Create feature vector for prediction
            game_features = pd.DataFrame()
            for feature in self.predictors:
                game_features[feature] = [home_data[feature] - away_data[feature]]
            
            # Scale features
            game_features_scaled = self.scaler.transform(game_features)
            
            # Make prediction
            prediction = self.model.predict(game_features_scaled)[0]
            confidence = np.max(self.model.decision_function(game_features_scaled))
            
            winner = home_team if prediction == 1 else away_team
            
            logger.info(f"Prediction complete: {winner} wins with confidence {confidence:.2f}")
            
            return {
                "winner": winner,
                "confidence": float(confidence),
                "home_team": home_team,
                "away_team": away_team,
                "date": date,
                "home_team_features": {
                    "win_pct_last_10": float(home_data['win_pct_last_10']),
                    "win_pct_season": float(home_data['win_pct_season']),
                    "momentum_score": float(home_data['momentum_score']),
                    "rest_days": float(home_data['rest_days'])
                },
                "away_team_features": {
                    "win_pct_last_10": float(away_data['win_pct_last_10']),
                    "win_pct_season": float(away_data['win_pct_season']),
                    "momentum_score": float(away_data['momentum_score']),
                    "rest_days": float(away_data['rest_days'])
                }
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
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
            
            # Split data into training and testing sets
            X = self.df[self.predictors]
            y = self.df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
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

    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {MODEL_FILE_PATH}")
            self.model = joblib.load(MODEL_FILE_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 