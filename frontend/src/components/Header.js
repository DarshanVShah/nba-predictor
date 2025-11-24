/**
 * Header Component
 * Displays the app header with title and model statistics
 */

import React from 'react';
import './Header.css';

const Header = ({ modelStats }) => {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-title-section">
          <h1 className="app-title">
            <span className="title-text">HoopsIQ</span>
          </h1>
          <p className="app-subtitle">AI-Powered Game Predictions</p>
        </div>
        {modelStats && (
          <div className="model-stats-badge">
            <div className="stat-item">
              <span className="stat-label">Accuracy</span>
              <span className="stat-value">
                {(modelStats.model_accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="stat-divider"></div>
            <div className="stat-item">
              <span className="stat-label">Games Analyzed</span>
              <span className="stat-value">
                {modelStats.data_statistics.total_games.toLocaleString()}
              </span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;

