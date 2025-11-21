"""
Utility module for fetching current team statistics for future game predictions.
This module fetches real-time team stats from the balldontlie API and calculates
rolling averages needed for predictions.
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from balldontlie import BalldontlieAPI
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Initialize API client (will be used if needed for future enhancements)
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("BALLDONTLIE_API_KEY", "bfdc4ecf-c070-4e93-b9ac-cb36f049efb1")
api = BalldontlieAPI(api_key=api_key) if api_key else None

# Team abbreviation mapping (balldontlie uses different abbreviations sometimes)
TEAM_ABBR_MAP = {
    'ATL': 1, 'BOS': 2, 'BKN': 3, 'BRK': 3, 'CHA': 4, 'CHI': 5, 'CLE': 6,
    'DAL': 7, 'DEN': 8, 'DET': 9, 'GSW': 10, 'HOU': 11, 'IND': 12,
    'LAC': 13, 'LAL': 14, 'MEM': 15, 'MIA': 16, 'MIL': 17, 'MIN': 18,
    'NOP': 19, 'NYK': 20, 'OKC': 21, 'ORL': 22, 'PHI': 23, 'PHX': 24,
    'POR': 25, 'SAC': 26, 'SAS': 27, 'TOR': 28, 'UTA': 29, 'WAS': 30
}

def get_team_id(team_abbr: str) -> Optional[int]:
    """Convert team abbreviation to balldontlie team ID."""
    return TEAM_ABBR_MAP.get(team_abbr.upper())


class TeamStatsFetcher:
    """Fetches and calculates current team statistics for predictions."""
    
    def __init__(self, historical_df: pd.DataFrame):
        """
        Initialize with historical data for calculating rolling averages.
        
        Args:
            historical_df: DataFrame with historical game data
        """
        self.historical_df = historical_df.copy()
        if 'date' in self.historical_df.columns:
            self.historical_df['date'] = pd.to_datetime(self.historical_df['date'])
            self.historical_df = self.historical_df.sort_values('date')
    
    def get_recent_games(self, team_abbr: str, n_games: int = 10) -> pd.DataFrame:
        """Get the most recent n games for a team from historical data."""
        team_games = self.historical_df[self.historical_df['team'] == team_abbr].copy()
        if team_games.empty:
            return pd.DataFrame()
        return team_games.tail(n_games)
    
    def calculate_rolling_stats(self, team_abbr: str, date: datetime) -> Dict:
        """
        Calculate rolling statistics for a team up to a given date.
        
        Args:
            team_abbr: Team abbreviation
            date: Date to calculate stats up to
            
        Returns:
            Dictionary of calculated statistics
        """
        # Get games up to the specified date
        team_games = self.historical_df[
            (self.historical_df['team'] == team_abbr) & 
            (self.historical_df['date'] < pd.to_datetime(date))
        ].copy()
        
        if team_games.empty:
            # Return default values if no historical data
            return {
                'win_pct_last_10': 0.5,
                'win_pct_season': 0.5,
                'momentum_score': 0.5,
                'rest_days': 2.0,  # Default rest days
                'fg%': 0.45,
                '3p': 10.0,
                'trb': 44.0,
                'efg%': 0.52,
                'ast%': 0.60,
                'usg%': 0.20,
                'ortg': 110.0,
                'fga_max': 90.0,
                '3pa_max': 35.0,
                'ft_max': 20.0,
                'orb_max': 10.0,
                'gmsc_max': 15.0,
                'ftr_max': 0.25,
                'stl%_max': 0.08,
                'blk%_max': 0.05,
            }
        
        # Calculate win percentages
        last_10 = team_games.tail(10)
        win_pct_last_10 = last_10['won'].mean() if len(last_10) > 0 else 0.5
        
        # Get current season
        if 'season' in team_games.columns:
            current_season = team_games['season'].iloc[-1]
            season_games = team_games[team_games['season'] == current_season]
            win_pct_season = season_games['won'].mean() if len(season_games) > 0 else 0.5
        else:
            win_pct_season = team_games['won'].mean()
        
        # Calculate momentum (weighted average of last 5 games)
        last_5 = team_games.tail(5)
        if len(last_5) > 0:
            weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1][:len(last_5)])
            momentum_score = np.sum(last_5['won'].values * weights) / np.sum(weights)
        else:
            momentum_score = 0.5
        
        # Calculate rest days (days since last game)
        if len(team_games) > 0:
            last_game_date = pd.to_datetime(team_games['date'].iloc[-1])
            rest_days = (pd.to_datetime(date) - last_game_date).days
            rest_days = max(0, min(rest_days, 7))  # Cap at 7 days
        else:
            rest_days = 2.0
        
        # Get average stats from recent games
        recent_games = team_games.tail(10)
        stats = {}
        
        # Base predictors
        base_predictors = [
            'fg%', '3p', 'trb', 'efg%', 'ast%', 'usg%', 'ortg',
            'fga_max', '3pa_max', 'ft_max', 'orb_max', 'gmsc_max',
            'ftr_max', 'stl%_max', 'blk%_max'
        ]
        
        for pred in base_predictors:
            if pred in recent_games.columns:
                stats[pred] = recent_games[pred].mean()
            else:
                # Default values if missing
                defaults = {
                    'fg%': 0.45, '3p': 10.0, 'trb': 44.0, 'efg%': 0.52,
                    'ast%': 0.60, 'usg%': 0.20, 'ortg': 110.0,
                    'fga_max': 90.0, '3pa_max': 35.0, 'ft_max': 20.0,
                    'orb_max': 10.0, 'gmsc_max': 15.0, 'ftr_max': 0.25,
                    'stl%_max': 0.08, 'blk%_max': 0.05
                }
                stats[pred] = defaults.get(pred, 0.0)
        
        # Add calculated features
        stats['win_pct_last_10'] = win_pct_last_10
        stats['win_pct_season'] = win_pct_season
        stats['momentum_score'] = momentum_score
        stats['rest_days'] = rest_days
        
        return stats
    
    def get_opponent_stats(self, team_abbr: str, date: datetime) -> Dict:
        """
        Get opponent-related statistics (opponent's defensive stats).
        
        Args:
            team_abbr: Opponent team abbreviation
            date: Date to calculate stats up to
            
        Returns:
            Dictionary of opponent statistics
        """
        team_games = self.historical_df[
            (self.historical_df['team'] == team_abbr) & 
            (self.historical_df['date'] < pd.to_datetime(date))
        ].copy()
        
        if team_games.empty:
            return {
                'fg%_opp': 0.45,
                'ast_opp': 25.0,
                'pts_opp': 110.0,
                'ts%_opp': 0.56,
                'efg%_opp': 0.52,
                'blk%_opp': 0.05,
                'usg%_opp': 0.20,
                'drtg_opp': 110.0,
                'fg%_max_opp': 0.45,
                'stl_max_opp': 8.0,
                'tov_max_opp': 14.0,
                'gmsc_max_opp': 15.0,
                'drb%_max_opp': 0.70,
                'ast%_max_opp': 0.60,
                'total_opp': 110.0,
            }
        
        recent_games = team_games.tail(10)
        
        opp_stats = {}
        opp_predictors = [
            'fg%_opp', 'ast_opp', 'pts_opp', 'ts%_opp', 'efg%_opp',
            'blk%_opp', 'usg%_opp', 'drtg_opp', 'fg%_max_opp',
            'stl_max_opp', 'tov_max_opp', 'gmsc_max_opp',
            'drb%_max_opp', 'ast%_max_opp', 'total_opp'
        ]
        
        for pred in opp_predictors:
            if pred in recent_games.columns:
                opp_stats[pred] = recent_games[pred].mean()
            else:
                # Try to derive from team stats
                if pred == 'fg%_opp' and 'fg%' in recent_games.columns:
                    opp_stats[pred] = recent_games['fg%'].mean()
                elif pred == 'efg%_opp' and 'efg%' in recent_games.columns:
                    opp_stats[pred] = recent_games['efg%'].mean()
                elif pred == 'usg%_opp' and 'usg%' in recent_games.columns:
                    opp_stats[pred] = recent_games['usg%'].mean()
                elif pred == 'drtg_opp' and 'drtg' in recent_games.columns:
                    opp_stats[pred] = recent_games['drtg'].mean()
                elif pred == 'ast%_max_opp' and 'ast%' in recent_games.columns:
                    opp_stats[pred] = recent_games['ast%'].mean()
                elif pred == 'stl_max_opp' and 'stl_max' in recent_games.columns:
                    opp_stats[pred] = recent_games['stl_max'].mean()
                elif pred == 'tov_max_opp' and 'tov_max' in recent_games.columns:
                    opp_stats[pred] = recent_games['tov_max'].mean()
                else:
                    defaults = {
                        'fg%_opp': 0.45, 'ast_opp': 25.0, 'pts_opp': 110.0,
                        'ts%_opp': 0.56, 'efg%_opp': 0.52, 'blk%_opp': 0.05,
                        'usg%_opp': 0.20, 'drtg_opp': 110.0, 'fg%_max_opp': 0.45,
                        'stl_max_opp': 8.0, 'tov_max_opp': 14.0,
                        'gmsc_max_opp': 15.0, 'drb%_max_opp': 0.70,
                        'ast%_max_opp': 0.60, 'total_opp': 110.0
                    }
                    opp_stats[pred] = defaults.get(pred, 0.0)
        
        return opp_stats
    
    def get_team_features_for_prediction(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str
    ) -> Dict:
        """
        Get all features needed for prediction for a future game.
        Returns features in the format expected by the model (home - away differential).
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Dictionary with all features (home team stats minus away team stats)
        """
        date = pd.to_datetime(game_date)
        
        # Get home team stats
        home_stats = self.calculate_rolling_stats(home_team, date)
        # Get away team's defensive stats (as opponent stats for home team)
        away_defensive_stats = self.get_opponent_stats(away_team, date)
        
        # Get away team stats
        away_stats = self.calculate_rolling_stats(away_team, date)
        # Get home team's defensive stats (as opponent stats for away team)
        home_defensive_stats = self.get_opponent_stats(home_team, date)
        
        # Combine features - this represents home team's perspective
        features = {}
        
        # Home team base offensive features
        base_features = [
            'fg%', '3p', 'trb', 'efg%', 'ast%', 'usg%', 'ortg',
            'fga_max', '3pa_max', 'ft_max', 'orb_max', 'gmsc_max',
            'ftr_max', 'stl%_max', 'blk%_max'
        ]
        
        for key in base_features:
            features[key] = home_stats.get(key, 0.0)
        
        # Opponent (away team) defensive features
        opp_features = [
            'fg%_opp', 'ast_opp', 'pts_opp', 'ts%_opp', 'efg%_opp',
            'blk%_opp', 'usg%_opp', 'drtg_opp', 'fg%_max_opp',
            'stl_max_opp', 'tov_max_opp', 'gmsc_max_opp',
            'drb%_max_opp', 'ast%_max_opp', 'total_opp'
        ]
        
        for key in opp_features:
            features[key] = away_defensive_stats.get(key, 0.0)
        
        # Add home court advantage
        features['home_court_advantage'] = 1.0
        
        # Add rest days (home team)
        features['rest_days'] = home_stats.get('rest_days', 2.0)
        
        # Add win percentages and momentum (home team)
        features['win_pct_last_10'] = home_stats.get('win_pct_last_10', 0.5)
        features['win_pct_season'] = home_stats.get('win_pct_season', 0.5)
        features['momentum_score'] = home_stats.get('momentum_score', 0.5)
        
        # Calculate opponent strength (away team's win percentage)
        features['opp_strength'] = away_stats.get('win_pct_season', 0.5)
        
        return features

