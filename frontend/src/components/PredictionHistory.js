/**
 * PredictionHistory Component
 * Displays past predictions and their accuracy
 */

import React, { useState, useEffect } from 'react';
import { getTeamLogo, getTeamName } from '../utils/teamLogos';
import './PredictionHistory.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const PredictionHistory = () => {
  const [selectedDate, setSelectedDate] = useState('');
  const [historicalGames, setHistoricalGames] = useState([]);
  const [historicalStats, setHistoricalStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Get today's date in YYYY-MM-DD format using EST timezone
  const getTodayDate = () => {
    const today = new Date();
    // Convert to EST
    const estDate = new Date(today.toLocaleString("en-US", {timeZone: "America/New_York"}));
    const year = estDate.getFullYear();
    const month = String(estDate.getMonth() + 1).padStart(2, '0');
    const day = String(estDate.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  // Get earliest allowed date (November 23, 2025)
  const getEarliestDate = () => {
    return '2025-11-23';
  };

  const fetchHistoricalPredictions = async (date) => {
    if (!date) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/historical-predictions/${date}`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch historical predictions' }));
        throw new Error(errorData.detail || 'Failed to fetch historical predictions');
      }
      const data = await response.json();
      setHistoricalGames(data.games || []);
      setHistoricalStats(data.stats || null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching historical predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDateChange = (e) => {
    const date = e.target.value;
    setSelectedDate(date);
    if (date) {
      fetchHistoricalPredictions(date);
    }
  };


  const formatDate = (dateString) => {
    const date = new Date(dateString + 'T00:00:00');
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      timeZone: 'America/New_York'
    });
  };

  // Ensure max date is not before earliest date
  const getMaxDate = () => {
    const today = getTodayDate();
    const earliest = getEarliestDate();
    return today >= earliest ? today : earliest;
  };

  return (
    <div className="prediction-history">
      <h1 className="history-page-title">Historical Analysis</h1>
      
      {/* Date Selector */}
      <div className="date-selector-section">
        <label htmlFor="date-selector" className="date-label">
          Select Date:
        </label>
        <input
          id="date-selector"
          type="date"
          value={selectedDate}
          onChange={handleDateChange}
          min={getEarliestDate()}
          max={getMaxDate()}
          className="date-input"
        />
        {selectedDate && (
          <button 
            className="clear-date-button"
            onClick={() => {
              setSelectedDate('');
              setHistoricalGames([]);
              setHistoricalStats(null);
            }}
          >
            Clear
          </button>
        )}
      </div>

      {/* Historical Predictions */}
      {loading && (
        <div className="history-loading">
          <div className="loading-spinner"></div>
          <p>Loading predictions for {selectedDate}...</p>
        </div>
      )}

      {error && (
        <div className="history-error">
          <p>Error: {error}</p>
          <button onClick={() => selectedDate && fetchHistoricalPredictions(selectedDate)} className="retry-button">
            Retry
          </button>
        </div>
      )}

      {!loading && !error && selectedDate && (
        <>
          {historicalStats && (
            <div className="history-stats">
              <h2 className="stats-title">Predictions for {formatDate(selectedDate)}</h2>
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-value">{historicalStats.total_games}</div>
                  <div className="stat-label">Total Games</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{historicalStats.completed_games}</div>
                  <div className="stat-label">Completed Games</div>
                </div>
                {historicalStats.accuracy !== null && (
                  <div className="stat-card highlight">
                    <div className="stat-value">{historicalStats.accuracy.toFixed(1)}%</div>
                    <div className="stat-label">Accuracy</div>
                  </div>
                )}
                {historicalStats.correct_predictions !== undefined && (
                  <div className="stat-card highlight">
                    <div className="stat-value">{historicalStats.correct_predictions}/{historicalStats.completed_games}</div>
                    <div className="stat-label">Correct Predictions</div>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="history-list">
            <h2 className="list-title">Game Predictions</h2>
            {historicalGames.length === 0 ? (
              <div className="no-predictions">
                <p>No games found for this date.</p>
              </div>
            ) : (
              <div className="predictions-grid">
                {historicalGames.map((game, index) => {
                  const homeLogo = getTeamLogo(game.home_team);
                  const awayLogo = getTeamLogo(game.away_team);
                  const isCorrect = game.is_correct;
                  const hasResult = game.actual_winner !== null;

                  return (
                    <div 
                      key={index} 
                      className={`history-card ${!hasResult ? 'future' : isCorrect ? 'correct' : 'incorrect'}`}
                    >
                      <div className="card-header">
                        <span className="game-date">{formatDate(selectedDate)}</span>
                        {hasResult && isCorrect !== null && (
                          <span className={`result-badge ${isCorrect ? 'correct' : 'incorrect'}`}>
                            {isCorrect ? '✓ Correct' : '✗ Incorrect'}
                          </span>
                        )}
                        {!hasResult && (
                          <span className="result-badge future">{game.status || 'Pending'}</span>
                        )}
                      </div>

                      <div className="card-teams">
                        <div className="team-row">
                          <div className="team-info">
                            {homeLogo && (
                              <img src={homeLogo} alt={game.home_team} className="team-logo-small" />
                            )}
                            <span className="team-name-text">{getTeamName(game.home_team)}</span>
                          </div>
                          {game.actual_home_score !== null && (
                            <span className="score">{game.actual_home_score}</span>
                          )}
                        </div>

                        <div className="vs-divider-small">VS</div>

                        <div className="team-row">
                          <div className="team-info">
                            {awayLogo && (
                              <img src={awayLogo} alt={game.away_team} className="team-logo-small" />
                            )}
                            <span className="team-name-text">{getTeamName(game.away_team)}</span>
                          </div>
                          {game.actual_away_score !== null && (
                            <span className="score">{game.actual_away_score}</span>
                          )}
                        </div>
                      </div>

                      <div className="card-prediction">
                        {game.predicted_winner && (
                          <div className="prediction-info">
                            <span className="pred-label">Predicted:</span>
                            <span className="pred-winner">{getTeamName(game.predicted_winner)}</span>
                            <span className="pred-confidence">({(game.confidence * 100).toFixed(1)}%)</span>
                          </div>
                        )}
                        {game.actual_winner && (
                          <div className="actual-info">
                            <span className="actual-label">Actual:</span>
                            <span className="actual-winner">{getTeamName(game.actual_winner)}</span>
                          </div>
                        )}
                        {!game.predicted_winner && (
                          <div className="prediction-info">
                            <span className="pred-label">Prediction unavailable</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}

      {!selectedDate && (
        <div className="no-date-selected">
          <p>Select a date to view historical predictions and compare with actual results.</p>
          <p className="date-hint">Earliest date: November 23, 2025</p>
        </div>
      )}
    </div>
  );
};

export default PredictionHistory;

