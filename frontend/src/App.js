/**
 * NBA Game Predictor - Main Application Component
 * 
 * This application uses machine learning to predict NBA game outcomes.
 * It fetches daily games and makes predictions using a trained model.
 */

import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import GameCard from './components/GameCard';
import './App.css';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  // State Management
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [modelStats, setModelStats] = useState(null);
  // Get today's date in EST timezone
  const getTodayEST = () => {
    const now = new Date();
    const estDate = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
    return estDate.toISOString().split('T')[0];
  };
  
  const [selectedDate, setSelectedDate] = useState(getTodayEST());
  const [predictingGames, setPredictingGames] = useState(new Set());

  /**
   * Fetch model statistics from the backend
   */
  const fetchModelStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model-stats`);
      if (response.ok) {
        const data = await response.json();
        setModelStats(data);
      }
    } catch (err) {
      console.error('Error fetching model stats:', err);
    }
  };

  /**
   * Fetch daily games and make predictions for each game
   * @param {string} date - Optional date string (YYYY-MM-DD)
   */
  const fetchDailyGames = async (date = null) => {
    const targetDate = date || selectedDate;
    setLoading(true);
    setError(null);
    
    try {
      // Fetch list of games for today
      const response = await fetch(`${API_BASE_URL}/daily-games`);
      if (!response.ok) {
        throw new Error(`Failed to fetch games: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      const gamesList = data.games || [];
      setGames(gamesList);
      
      // Fetch predictions for each game in parallel
      const predictionPromises = gamesList.map(async (game) => {
        const gameKey = `${game.home_team}-${game.away_team}`;
        setPredictingGames(prev => new Set(prev).add(gameKey));
        
        try {
          const predResponse = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              home_team: game.home_team,
              away_team: game.away_team,
              game_date: targetDate
            })
          });
          
          if (!predResponse.ok) {
            // Try to extract error message from response
            let errorMessage = 'Prediction failed';
            try {
              const errorData = await predResponse.json();
              errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
              errorMessage = `HTTP ${predResponse.status}: ${predResponse.statusText}`;
            }
            console.error(`Error predicting game ${gameKey}:`, errorMessage);
            return null;
          }
          
          return predResponse.json();
        } catch (err) {
          console.error(`Error predicting game ${gameKey}:`, err.message || err);
          return null;
        } finally {
          setPredictingGames(prev => {
            const newSet = new Set(prev);
            newSet.delete(gameKey);
            return newSet;
          });
        }
      });
      
      // Wait for all predictions to complete
      const predResults = await Promise.all(predictionPromises);
      
      // Map predictions to games
      const predMap = {};
      gamesList.forEach((game, idx) => {
        if (predResults[idx]) {
          predMap[`${game.home_team}-${game.away_team}`] = predResults[idx];
        }
      });
      
      setPredictions(predMap);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred');
      console.error('Error fetching daily games:', err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Format date for display (in EST timezone)
   * @param {string} dateString - Date string (YYYY-MM-DD)
   * @returns {string} - Formatted date string
   */
  const formatDate = (dateString) => {
    // Parse date and format in EST timezone
    const date = new Date(dateString + 'T00:00:00');
    return date.toLocaleDateString('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      timeZone: 'America/New_York'
    });
  };

  // Initialize: Fetch games and model stats on component mount
  useEffect(() => {
    fetchDailyGames();
    fetchModelStats();
    
    // Auto-refresh every 5 minutes
    const interval = setInterval(() => {
      fetchDailyGames();
      fetchModelStats();
    }, 300000); // 5 minutes
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="App">
      <Header modelStats={modelStats} />

      <main className="app-main">
        {/* Error Display */}
        {error && (
          <div className="error-banner">
            <div className="error-content">
              <span className="error-message">{error}</span>
              <button 
                onClick={() => fetchDailyGames()} 
                className="error-retry-button"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Controls Bar */}
        <div className="controls-section">
          <button 
            className="refresh-button"
            onClick={() => fetchDailyGames()} 
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="button-spinner"></span>
                <span>Refreshing...</span>
              </>
            ) : (
              <>
                <span>Refresh Games</span>
              </>
            )}
          </button>
        </div>

        {/* Loading State */}
        {loading && games.length === 0 ? (
          <div className="loading-state">
            <div className="loading-spinner-large"></div>
            <p className="loading-text">Loading today's games...</p>
            <p className="loading-subtext">Fetching predictions from AI model</p>
          </div>
        ) : games.length === 0 ? (
          /* No Games State */
          <div className="empty-state">
            <div className="empty-icon">📅</div>
            <h2 className="empty-title">No Games Scheduled</h2>
            <p className="empty-description">
              There are no NBA games scheduled for today.
            </p>
          </div>
        ) : (
          /* Games List */
          <>
            <div className="games-section-header">
              <h2 className="section-title">Today's Games</h2>
              <p className="section-date">{formatDate(selectedDate)}</p>
            </div>
            
            <div className="games-grid">
              {games.map((game, index) => {
                const gameKey = `${game.home_team}-${game.away_team}`;
                const isPredicting = predictingGames.has(gameKey);
                const prediction = predictions[gameKey];
                
                return (
                  <GameCard
                    key={index}
                    game={game}
                    prediction={prediction}
                    isPredicting={isPredicting}
                  />
                );
              })}
            </div>
          </>
        )}
      </main>

      <Footer modelStats={modelStats} />
    </div>
  );
}

export default App;
