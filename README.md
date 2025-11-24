# 🏀 NBA Game Predictor

An AI-powered web application that predicts NBA game winners using machine learning. Built with FastAPI (Python) and React.

![NBA Predictor](https://img.shields.io/badge/Python-3.8+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![React](https://img.shields.io/badge/React-19.1-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **AI-Powered Predictions**: Machine learning model trained on historical NBA game data
- **Real-Time Game Data**: Fetches today's NBA games automatically
- **Win Probability**: Shows confidence scores for each prediction with visual probability bars
- **Modern UI**: Beautiful, polished design with team logos, smooth animations, and gradient backgrounds
- **Model Statistics**: Displays model accuracy and training data information in the header
- **Future Game Support**: Can predict games that haven't been played yet
- **Responsive Design**: Fully responsive layout that works on desktop, tablet, and mobile
- **Team Logos**: Uses local team logo assets for fast loading and offline support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd nba-predictor
```

2. **Set up backend**
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your BALLDONTLIE_API_KEY
```

3. **Set up frontend**
```bash
cd frontend
npm install

# Create .env file
echo "REACT_APP_API_URL=http://localhost:8000" > .env
```

### Running the Application

1. **Start the backend** (from project root):
```bash
uvicorn backend.app:app --reload
```

2. **Start the frontend** (from frontend directory):
```bash
npm start
```

3. **Open your browser** to `http://localhost:3000`

## 📊 How It Works

### Data Pipeline

1. **Historical Data**: The model is trained on thousands of historical NBA games
2. **Feature Engineering**: Extracts 30+ features including:
   - Team statistics (FG%, 3P, rebounds, assists, etc.)
   - Advanced metrics (ORtg, DRtg, eFG%, etc.)
   - Contextual features (home court advantage, rest days, momentum)
   - Opponent strength metrics

3. **Model Training**: Uses Ridge Classifier with time-series cross-validation
4. **Prediction**: For future games, calculates rolling averages from recent games

### Model Architecture

- **Algorithm**: Ridge Classifier (L2-regularized logistic regression)
- **Features**: 30+ statistical and contextual features
- **Validation**: Time-series cross-validation to prevent data leakage
- **Accuracy**: ~63% on historical test data

## 🎯 API Endpoints

### `POST /predict`
Predict the winner of an NBA game

**Request:**
```json
{
  "home_team": "LAL",
  "away_team": "BOS",
  "game_date": "2024-01-15"
}
```

**Response:**
```json
{
  "winner": "LAL",
  "confidence": 0.72,
  "home_team": "LAL",
  "away_team": "BOS",
  "date": "2024-01-15",
  "home_team_features": {...},
  "away_team_features": {...}
}
```

### `GET /daily-games`
Get today's NBA games

### `GET /teams`
Get list of all NBA teams

### `GET /model-stats`
Get model evaluation statistics

### `GET /health`
Health check endpoint

## 🛠️ Project Structure

```
nba-predictor/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── config.py              # Configuration
│   ├── models/                # Saved ML models
│   ├── data/                  # Game data
│   └── utils/
│       ├── predictor.py        # Prediction logic
│       ├── team_stats_fetcher.py  # Stats fetching for future games
│       └── data_pipeline.py   # Data processing
├── frontend/
│   ├── public/
│   │   └── assets/            # Team logo images (PNG/GIF)
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Main application styles
│   │   ├── components/        # React components
│   │   │   ├── Header.js      # Header with model stats
│   │   │   ├── Footer.js      # Footer component
│   │   │   └── GameCard.js    # Individual game card
│   │   └── utils/
│   │       └── teamLogos.js  # Team logo mapping utilities
│   └── package.json
├── notebooks/                 # Jupyter notebooks for analysis
├── requirements.txt           # Python dependencies
└── README.md
```

## 🔧 Configuration

### Environment Variables

**Backend (.env):**
```
BALLDONTLIE_API_KEY=your_api_key_here
BACKEND_HOST=localhost
BACKEND_PORT=8000
```

**Frontend (.env):**
```
REACT_APP_API_URL=http://localhost:8000
```

## 📈 Model Performance

- **Accuracy**: ~63% on test set
- **Training Data**: 9,000+ historical games
- **Features**: 30+ statistical and contextual features
- **Validation**: Time-series cross-validation

## 🚢 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Quick Deploy

**Backend:**
```bash
gunicorn backend.app:app -w 4 -k uvicorn.workers.UvicornWorker
```

**Frontend:**
```bash
cd frontend
npm run build
# Serve the build/ directory
```

## 🎓 What This Project Demonstrates

- **Full-Stack Development**: FastAPI backend + React frontend
- **Machine Learning**: Feature engineering, model training, deployment
- **Data Engineering**: Web scraping, data processing, ETL pipelines
- **API Design**: RESTful APIs with proper error handling
- **Modern UI/UX**: Responsive design, loading states, error handling

## 🔮 Future Improvements

- [ ] Upgrade to ensemble models (XGBoost, Random Forest)
- [ ] Add player-level features (injuries, recent performance)
- [ ] Implement prediction history tracking
- [ ] Add comparison with betting odds
- [ ] Real-time game updates
- [ ] User accounts and prediction tracking

## 📝 License

MIT License

## 🙏 Acknowledgments

- [balldontlie.io](https://www.balldontlie.io/) for NBA game data API
- [Basketball Reference](https://www.basketball-reference.com/) for historical statistics

## 📧 Contact

For questions or issues, please open an issue on GitHub.
