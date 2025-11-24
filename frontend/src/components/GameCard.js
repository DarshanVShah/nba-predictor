/**
 * GameCard Component
 * Displays a single NBA game with teams, logos, and predictions
 */

import React from 'react';
import { getTeamLogo, getTeamName, getShortTeamName } from '../utils/teamLogos';
import './GameCard.css';

const GameCard = ({ game, prediction, isPredicting, onPredict }) => {
  const gameKey = `${game.home_team}-${game.away_team}`;
  const homeWinProb = prediction 
    ? (prediction.winner === game.home_team ? prediction.confidence : 1 - prediction.confidence)
    : null;
  const awayWinProb = homeWinProb !== null ? 1 - homeWinProb : null;

  const homeLogo = getTeamLogo(game.home_team);
  const awayLogo = getTeamLogo(game.away_team);

  return (
    <div className="game-card">
      <div className="game-status-badge">
        <span className="status-text">{game.status || 'Scheduled'}</span>
      </div>

      <div className="teams-matchup">
        {/* Home Team */}
        <div className={`team-card home-team ${prediction?.winner === game.home_team ? 'predicted-winner' : ''}`}>
          <div className="team-logo-wrapper">
            {homeLogo ? (
              <img 
                src={homeLogo} 
                alt={getTeamName(game.home_team)}
                className="team-logo"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'flex';
                }}
              />
            ) : null}
            <div className="team-logo-fallback" style={{ display: homeLogo ? 'none' : 'flex' }}>
              <span className="fallback-text">{game.home_team}</span>
            </div>
          </div>
          <div className="team-details">
            <h3 className="team-abbreviation">{game.home_team}</h3>
            <p className="team-name">{getShortTeamName(game.home_team)}</p>
          </div>
          {homeWinProb !== null && (
            <div className="win-probability">
              <div className="probability-bar-container">
                <div 
                  className="probability-bar home-bar"
                  style={{ width: `${homeWinProb * 100}%` }}
                ></div>
              </div>
              <span className="probability-percentage">
                {(homeWinProb * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {isPredicting && (
            <div className="predicting-overlay">
              <div className="spinner"></div>
              <span>Analyzing...</span>
            </div>
          )}
        </div>

        {/* VS Divider */}
        <div className="vs-container">
          <div className="vs-line"></div>
          <span className="vs-text">VS</span>
          <div className="vs-line"></div>
        </div>

        {/* Away Team */}
        <div className={`team-card away-team ${prediction?.winner === game.away_team ? 'predicted-winner' : ''}`}>
          <div className="team-logo-wrapper">
            {awayLogo ? (
              <img 
                src={awayLogo} 
                alt={getTeamName(game.away_team)}
                className="team-logo"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'flex';
                }}
              />
            ) : null}
            <div className="team-logo-fallback" style={{ display: awayLogo ? 'none' : 'flex' }}>
              <span className="fallback-text">{game.away_team}</span>
            </div>
          </div>
          <div className="team-details">
            <h3 className="team-abbreviation">{game.away_team}</h3>
            <p className="team-name">{getShortTeamName(game.away_team)}</p>
          </div>
          {awayWinProb !== null && (
            <div className="win-probability">
              <div className="probability-bar-container">
                <div 
                  className="probability-bar away-bar"
                  style={{ width: `${awayWinProb * 100}%` }}
                ></div>
              </div>
              <span className="probability-percentage">
                {(awayWinProb * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {isPredicting && (
            <div className="predicting-overlay">
              <div className="spinner"></div>
              <span>Analyzing...</span>
            </div>
          )}
        </div>
      </div>

      {/* Prediction Summary */}
      {prediction && (
        <div className="prediction-summary">
          <div className="prediction-header">
            <span className="prediction-icon">🎯</span>
            <span className="prediction-label">Predicted Winner</span>
          </div>
          <div className="prediction-result">
            {prediction.confidence >= 0.45 && prediction.confidence <= 0.55 ? (
              <span className="no-clear-winner">No Clear Winner - Too Close to Call</span>
            ) : (
              <>
                <span className="winner-team">{getTeamName(prediction.winner)}</span>
                <span className="confidence-badge">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Error State */}
      {!prediction && !isPredicting && (
        <div className="prediction-error">
          <span className="error-icon">⚠️</span>
          <span className="error-text">Prediction unavailable</span>
        </div>
      )}
    </div>
  );
};

export default GameCard;

