/**
 * PredictionHistory Component
 * Displays past predictions and their accuracy
 */

import React, { useState, useEffect } from 'react';
import { getTeamLogo, getTeamName } from '../utils/teamLogos';
import './PredictionHistory.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const PredictionHistory = () => {
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/prediction-history?limit=100`);
      if (!response.ok) {
        throw new Error('Failed to fetch prediction history');
      }
      const data = await response.json();
      setPredictions(data.predictions || []);
      setStats(data.stats || null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching prediction history:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const isFutureGame = (prediction) => {
    const gameDate = new Date(prediction.game_date);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return gameDate >= today;
  };

  if (loading) {
    return (
      <div className="history-loading">
        <div className="loading-spinner"></div>
        <p>Loading prediction history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="history-error">
        <p>Error loading history: {error}</p>
        <button onClick={fetchHistory} className="retry-button">Retry</button>
      </div>
    );
  }

  return (
    <div className="prediction-history">
      {/* Statistics Section */}
      {stats && (
        <div className="history-stats">
          <h2 className="stats-title">Prediction Statistics</h2>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{stats.total_predictions}</div>
              <div className="stat-label">Total Predictions</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.completed_predictions}</div>
              <div className="stat-label">Completed Games</div>
            </div>
            <div className="stat-card highlight">
              <div className="stat-value">{stats.accuracy}%</div>
              <div className="stat-label">Overall Accuracy</div>
            </div>
            <div className="stat-card highlight">
              <div className="stat-value">{stats.recent_accuracy}%</div>
              <div className="stat-label">Recent Accuracy (Last 50)</div>
            </div>
          </div>
        </div>
      )}

      {/* Predictions List */}
      <div className="history-list">
        <h2 className="list-title">Past Predictions</h2>
        {predictions.length === 0 ? (
          <div className="no-predictions">
            <p>No predictions found. Make some predictions to see them here!</p>
          </div>
        ) : (
          <div className="predictions-grid">
            {predictions.map((pred) => {
              const isFuture = isFutureGame(pred);
              const homeLogo = getTeamLogo(pred.home_team);
              const awayLogo = getTeamLogo(pred.away_team);

              return (
                <div key={pred.id} className={`history-card ${isFuture ? 'future' : pred.is_correct ? 'correct' : 'incorrect'}`}>
                  <div className="card-header">
                    <span className="game-date">{formatDate(pred.game_date)}</span>
                    {!isFuture && pred.is_correct !== null && (
                      <span className={`result-badge ${pred.is_correct ? 'correct' : 'incorrect'}`}>
                        {pred.is_correct ? '✓ Correct' : '✗ Incorrect'}
                      </span>
                    )}
                    {isFuture && (
                      <span className="result-badge future">Upcoming</span>
                    )}
                  </div>

                  <div className="card-teams">
                    <div className="team-row">
                      <div className="team-info">
                        {homeLogo && (
                          <img src={homeLogo} alt={pred.home_team} className="team-logo-small" />
                        )}
                        <span className="team-name-text">{getTeamName(pred.home_team)}</span>
                      </div>
                      {pred.actual_score_home !== null && (
                        <span className="score">{pred.actual_score_home}</span>
                      )}
                    </div>

                    <div className="vs-divider-small">VS</div>

                    <div className="team-row">
                      <div className="team-info">
                        {awayLogo && (
                          <img src={awayLogo} alt={pred.away_team} className="team-logo-small" />
                        )}
                        <span className="team-name-text">{getTeamName(pred.away_team)}</span>
                      </div>
                      {pred.actual_score_away !== null && (
                        <span className="score">{pred.actual_score_away}</span>
                      )}
                    </div>
                  </div>

                  <div className="card-prediction">
                    <div className="prediction-info">
                      <span className="pred-label">Predicted:</span>
                      <span className="pred-winner">{getTeamName(pred.predicted_winner)}</span>
                      <span className="pred-confidence">({(pred.confidence * 100).toFixed(1)}%)</span>
                    </div>
                    {pred.actual_winner && (
                      <div className="actual-info">
                        <span className="actual-label">Actual:</span>
                        <span className="actual-winner">{getTeamName(pred.actual_winner)}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionHistory;

