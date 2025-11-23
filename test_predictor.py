from backend.utils.predictor import NBAPredictor
import traceback

try:
    p = NBAPredictor()
    p.load_data()
    print('SUCCESS: Data loaded')
    print('Predictors:', len(p.predictors))
    print('First 5 predictors:', p.predictors[:5])
    
    # Test a prediction
    result = p.predict_game('CLE', 'IND', '2025-11-22')
    print('Prediction result:', result)
except Exception as e:
    print('ERROR:', str(e))
    traceback.print_exc()



