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
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState(null);
  const [updating, setUpdating] = useState(false);
  const [viewMode, setViewMode] = useState('history'); // 'history' or 'date'

  // Get today's date in YYYY-MM-DD format
  const getTodayDate = () => {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
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

  useEffect(() => {
    if (viewMode === 'history') {
      fetchHistory();
    }
  }, [viewMode]);

  const triggerUpdate = async () => {
    setUpdating(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/trigger-prediction-update`, {
        method: 'POST'
      });
      if (!response.ok) {
        throw new Error('Failed to trigger update');
      }
      const data = await response.json();
      // Refresh history after update
      await fetchHistory();
      alert(`Update complete! Updated ${data.updated} out of ${data.total} predictions.`);
    } catch (err) {
      setError(err.message);
      console.error('Error triggering update:', err);
    } finally {
      setUpdating(false);
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
      {/* View Mode Toggle */}
      <div className="view-mode-toggle">
        <button 
          className={viewMode === 'history' ? 'active' : ''}
          onClick={() => setViewMode('history')}
        >
          Saved Predictions
        </button>
        <button 
          className={viewMode === 'date' ? 'active' : ''}
          onClick={() => setViewMode('date')}
        >
          Historical Analysis
        </button>
      </div>

      {viewMode === 'date' ? (
        <>
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
              max={getTodayDate()}
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
        </>
      ) : (
        <>
          {/* Statistics Section */}
          {stats && (
            <div className="history-stats">
              <div className="stats-header">
                <h2 className="stats-title">Prediction Statistics</h2>
                <button 
                  className="update-button"
                  onClick={triggerUpdate}
                  disabled={updating}
                >
                  {updating ? (
                    <>
                      <span className="button-spinner-small"></span>
                      <span>Updating...</span>
                    </>
                  ) : (
                    <>
                      <span>🔄</span>
                      <span>Update Results</span>
                    </>
                  )}
                </button>
              </div>
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
        </>
      )}
    </div>
  );
};

export default PredictionHistory;

