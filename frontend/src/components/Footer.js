/**
 * Footer Component
 * Displays footer information and model statistics
 */

import React from 'react';
import './Footer.css';

const Footer = ({ modelStats }) => {
  return (
    <footer className="app-footer">
      <div className="footer-content">
        <div className="footer-main">
          <p className="footer-text">
            <span className="footer-icon">⚡</span>
            Powered by Machine Learning
          </p>
          <p className="footer-tech">
            Built with <span className="tech-highlight">FastAPI</span> & <span className="tech-highlight">React</span>
          </p>
        </div>
        {modelStats && (
          <div className="footer-stats">
            <p className="stats-text">
              Model trained on <strong>{modelStats.data_statistics.total_games.toLocaleString()}</strong> games
            </p>
            <p className="stats-date-range">
              {modelStats.data_statistics.date_range.start} → {modelStats.data_statistics.date_range.end}
            </p>
          </div>
        )}
      </div>
    </footer>
  );
};

export default Footer;

