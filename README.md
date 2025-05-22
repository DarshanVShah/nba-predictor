# NBA Game Predictor

This application predicts the winner of NBA games using machine learning. It uses historical game data and team statistics to make predictions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the NBA games dataset (`nba_games.csv`) in the `nba_predictor/data` directory.

## Running the API

Start the FastAPI server:
```bash
uvicorn nba_predictor.app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Predict Game Winner
```
POST /predict
```

Request body:
```json
{
    "home_team": "LAL",
    "away_team": "BOS",
    "game_date": "2024-01-01"
}
```

Response:
```json
{
    "winner": "LAL",
    "confidence": 0.75,
    "home_team": "LAL",
    "away_team": "BOS",
    "date": "2024-01-01"
}
```

### Get Available Teams
```
GET /teams
```

Response:
```json
{
    "teams": ["ATL", "BOS", "BRK", "CHA", "CHI", ...]
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
