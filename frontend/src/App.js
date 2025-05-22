import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState({});

  const fetchDailyGames = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/daily-games');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setGames(data.games);
      
      // Fetch predictions for each game
      const predictionPromises = data.games.map(async (game) => {
        const today = new Date().toISOString().split('T')[0];
        const predResponse = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            home_team: game.home_team,
            away_team: game.away_team,
            game_date: today
          })
        });
        return predResponse.json();
      });
      
      const predResults = await Promise.all(predictionPromises);
      const predMap = {};
      data.games.forEach((game, idx) => {
        predMap[`${game.home_team}-${game.away_team}`] = predResults[idx];
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
    // Refresh every 5 minutes
    const interval = setInterval(fetchDailyGames, 300000);
    return () => clearInterval(interval);
  }, []);

  const getTeamLogo = (teamAbbr) => {
    return `https://cdn.nba.com/logos/nba/${teamAbbr.toLowerCase()}/primary/L/logo.svg`;
  };

  const getWinProbability = (game) => {
    const key = `${game.home_team}-${game.away_team}`;
    const prediction = predictions[key];
    if (!prediction) return null;
    return prediction.winner === game.home_team ? prediction.confidence : 1 - prediction.confidence;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>NBA Game Predictor</h1>
        <p className="subtitle">Today's Games & Predictions</p>
      </header>

      <main className="App-main">
        {error && <div className="error-message">Error: {error}</div>}
        
        {loading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading today's games...</p>
          </div>
        ) : games.length === 0 ? (
          <div className="no-games">
            <p>No games scheduled for today</p>
          </div>
        ) : (
          <div className="games-grid">
            {games.map((game, index) => {
              const homeWinProb = getWinProbability(game);
              const awayWinProb = homeWinProb !== null ? 1 - homeWinProb : null;
              
              return (
                <div key={index} className="game-card">
                  <div className="game-time">{game.status}</div>
                  
                  <div className="teams-container">
                    <div className="team home">
                      <img src={getTeamLogo(game.home_team)} alt={game.home_team} />
                      <span className="team-name">{game.home_team}</span>
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
                    </div>
                    
                    <div className="vs">VS</div>
                    
                    <div className="team away">
                      <img src={getTeamLogo(game.away_team)} alt={game.away_team} />
                      <span className="team-name">{game.away_team}</span>
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
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        
        <button 
          className="refresh-button"
          onClick={fetchDailyGames} 
          disabled={loading}
        >
          {loading ? 'Refreshing...' : 'Refresh Games'}
        </button>
      </main>
    </div>
  );
}

export default App;
