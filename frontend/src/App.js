import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [modelStats, setModelStats] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [predictingGames, setPredictingGames] = useState(new Set());

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

  const fetchDailyGames = async (date = null) => {
    const targetDate = date || selectedDate;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/daily-games`);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setGames(data.games || []);
      
      // Fetch predictions for each game
      const predictionPromises = (data.games || []).map(async (game) => {
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
            throw new Error('Prediction failed');
          }
          
          return predResponse.json();
        } catch (err) {
          console.error(`Error predicting game ${gameKey}:`, err);
          return null;
        } finally {
          setPredictingGames(prev => {
            const newSet = new Set(prev);
            newSet.delete(gameKey);
            return newSet;
          });
        }
      });
      
      const predResults = await Promise.all(predictionPromises);
      const predMap = {};
      (data.games || []).forEach((game, idx) => {
        if (predResults[idx]) {
          predMap[`${game.home_team}-${game.away_team}`] = predResults[idx];
        }
      });
      setPredictions(predMap);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDailyGames();
    fetchModelStats();
    // Refresh every 5 minutes
    const interval = setInterval(() => {
      fetchDailyGames();
      fetchModelStats();
    }, 300000);
    return () => clearInterval(interval);
  }, []);

  const getTeamLogo = (teamAbbr) => {
    // Use a more reliable logo source
    return `https://cdn.nba.com/logos/nba/${teamAbbr}/primary/L/logo.svg`;
  };

  const getTeamName = (teamAbbr) => {
    const teamNames = {
      'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'BRK': 'Nets',
      'CHA': 'Hornets', 'CHI': 'Bulls', 'CLE': 'Cavaliers',
      'DAL': 'Mavericks', 'DEN': 'Nuggets', 'DET': 'Pistons',
      'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
      'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies',
      'MIA': 'Heat', 'MIL': 'Bucks', 'MIN': 'Timberwolves',
      'NOP': 'Pelicans', 'NYK': 'Knicks', 'OKC': 'Thunder',
      'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
      'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs',
      'TOR': 'Raptors', 'UTA': 'Jazz', 'WAS': 'Wizards'
    };
    return teamNames[teamAbbr] || teamAbbr;
  };

  const getWinProbability = (game) => {
    const key = `${game.home_team}-${game.away_team}`;
    const prediction = predictions[key];
    if (!prediction) return null;
    return prediction.winner === game.home_team ? prediction.confidence : 1 - prediction.confidence;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>🏀 NBA Game Predictor</h1>
          <p className="subtitle">AI-Powered Game Predictions</p>
        </div>
        {modelStats && (
          <div className="model-badge">
            <span className="accuracy-label">Model Accuracy:</span>
            <span className="accuracy-value">{(modelStats.model_accuracy * 100).toFixed(1)}%</span>
          </div>
        )}
      </header>

      <main className="App-main">
        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <span>Error: {error}</span>
            <button onClick={() => fetchDailyGames()} className="retry-button">Retry</button>
          </div>
        )}
        
        <div className="controls-bar">
          <button 
            className="refresh-button"
            onClick={() => fetchDailyGames()} 
            disabled={loading}
          >
            {loading ? '🔄 Refreshing...' : '🔄 Refresh Games'}
          </button>
          {modelStats && (
            <div className="stats-info">
              <span>📊 {modelStats.data_statistics.total_games.toLocaleString()} games analyzed</span>
            </div>
          )}
        </div>

        {loading && games.length === 0 ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading today's games...</p>
          </div>
        ) : games.length === 0 ? (
          <div className="no-games">
            <div className="no-games-icon">📅</div>
            <h2>No Games Scheduled</h2>
            <p>There are no NBA games scheduled for today.</p>
          </div>
        ) : (
          <>
            <div className="games-header">
              <h2>Today's Games</h2>
              <p className="games-date">{formatDate(selectedDate)}</p>
            </div>
            <div className="games-grid">
              {games.map((game, index) => {
                const gameKey = `${game.home_team}-${game.away_team}`;
                const isPredicting = predictingGames.has(gameKey);
                const homeWinProb = getWinProbability(game);
                const awayWinProb = homeWinProb !== null ? 1 - homeWinProb : null;
                const prediction = predictions[gameKey];
                
                return (
                  <div key={index} className="game-card">
                    <div className="game-status">{game.status || 'Scheduled'}</div>
                    
                    <div className="teams-container">
                      <div className={`team home ${prediction?.winner === game.home_team ? 'predicted-winner' : ''}`}>
                        <div className="team-logo-container">
                          <img 
                            src={getTeamLogo(game.home_team)} 
                            alt={game.home_team}
                            onError={(e) => {
                              e.target.src = `https://via.placeholder.com/64/1a202c/ffffff?text=${game.home_team}`;
                            }}
                          />
                        </div>
                        <div className="team-info">
                          <span className="team-abbr">{game.home_team}</span>
                          <span className="team-name">{getTeamName(game.home_team)}</span>
                        </div>
                        {homeWinProb !== null && (
                          <div className="win-probability">
                            <div className="probability-bar">
                              <div 
                                className="probability-fill home"
                                style={{ width: `${homeWinProb * 100}%` }}
                              ></div>
                            </div>
                            <span className="probability-text">
                              {(homeWinProb * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                        {isPredicting && (
                          <div className="predicting-indicator">⏳</div>
                        )}
                      </div>
                      
                      <div className="vs-divider">
                        <span className="vs">VS</span>
                      </div>
                      
                      <div className={`team away ${prediction?.winner === game.away_team ? 'predicted-winner' : ''}`}>
                        <div className="team-logo-container">
                          <img 
                            src={getTeamLogo(game.away_team)} 
                            alt={game.away_team}
                            onError={(e) => {
                              e.target.src = `https://via.placeholder.com/64/1a202c/ffffff?text=${game.away_team}`;
                            }}
                          />
                        </div>
                        <div className="team-info">
                          <span className="team-abbr">{game.away_team}</span>
                          <span className="team-name">{getTeamName(game.away_team)}</span>
                        </div>
                        {awayWinProb !== null && (
                          <div className="win-probability">
                            <div className="probability-bar">
                              <div 
                                className="probability-fill away"
                                style={{ width: `${awayWinProb * 100}%` }}
                              ></div>
                            </div>
                            <span className="probability-text">
                              {(awayWinProb * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                        {isPredicting && (
                          <div className="predicting-indicator">⏳</div>
                        )}
                      </div>
                    </div>

                    {prediction && (
                      <div className="prediction-details">
                        <div className="prediction-summary">
                          <span className="prediction-label">Predicted Winner:</span>
                          <span className="prediction-winner">
                            {prediction.winner} ({(prediction.confidence * 100).toFixed(1)}% confidence)
                          </span>
                        </div>
                        <div className="team-stats">
                          <div className="team-stat-item">
                            <span className="stat-label">Home:</span>
                            <span className="stat-value">
                              {getTeamName(prediction.home_team)} - 
                              Win%: {(prediction.home_team_features.win_pct_season * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="team-stat-item">
                            <span className="stat-label">Away:</span>
                            <span className="stat-value">
                              {getTeamName(prediction.away_team)} - 
                              Win%: {(prediction.away_team_features.win_pct_season * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </main>

      <footer className="App-footer">
        <p>Powered by Machine Learning • Built with FastAPI & React</p>
        {modelStats && (
          <p className="footer-stats">
            Model trained on {modelStats.data_statistics.total_games.toLocaleString()} games
            from {modelStats.data_statistics.date_range.start} to {modelStats.data_statistics.date_range.end}
          </p>
        )}
      </footer>
    </div>
  );
}

export default App;
