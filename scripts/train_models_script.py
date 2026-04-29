import pandas as pd
import os
import sys
import torch
import pickle
import numpy as np
from prophet.serialize import model_to_json

# Change working directory to project root
if os.getcwd().endswith('notebooks') or os.getcwd().endswith('scripts'):
    os.chdir('..')
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from backend.app.models.lstm_model import CrowdLSTMModel
from backend.app.models.prophet_model import ForecastModel

# Ensure directories
os.makedirs('models', exist_ok=True)

print('Loading dataset...')
df = pd.read_csv('data/processed/processed_crowd_data.csv')
df['ds'] = pd.to_datetime(df['timestamp'])
df['y'] = df['count'].rolling(window=3, min_periods=1).mean()
df['hour'] = df['ds'].dt.hour
df['day'] = df['ds'].dt.dayofweek

# Training on more data
df_train = df.tail(15000).copy()
print(f'Training on {len(df_train)} datapoints.')

# 1. Train Prophet
prophet = ForecastModel()
print('Training Prophet...')
prophet.train(df_train[['ds', 'y', 'hour', 'day']])
prophet.save('models/prophet_model.json')
print('Prophet model saved.')

# 2. Train LSTM
lstm = CrowdLSTMModel(sequence_length=60, epochs=300, lr=0.0005)
print('Training LSTM...')
lstm.train(df_train[['ds', 'y', 'hour', 'day']])
lstm.save(('models/lstm_weights.pth', 'models/scaler.pkl'))
print('LSTM weights and scaling architecture saved.')

# 3. Evaluation
print('--- Model Evaluation ---')
try:
    # Prophet evaluation
    p_eval_df = df_train[['ds', 'hour', 'day']].copy()
    p_forecast_full = prophet.model.predict(p_eval_df)
    p_mae = np.abs(p_forecast_full['yhat'].values - df_train['y'].values).mean()
    p_acc = 100 * (1 - p_mae / df_train['y'].mean())
    print(f'Prophet - MAE: {p_mae:.2f}, Approximate Accuracy: {p_acc:.2f}%')
    
    # LSTM evaluation
    lstm.model.eval()
    features = ['y', 'hour', 'day']
    test_data = df_train[features].tail(160).values
    test_data_norm = lstm.scaler.transform(test_data)
    
    preds = []
    with torch.no_grad():
        for i in range(100):
            seq = torch.FloatTensor(test_data_norm[i:i+60]).unsqueeze(0)
            pred = lstm.model(seq).item()
            preds.append(pred)
            
    preds_padded = np.zeros((len(preds), 3))
    preds_padded[:, 0] = preds
    preds_orig = lstm.scaler.inverse_transform(preds_padded)[:, 0]
    
    actuals = test_data[60:, 0].flatten()
    l_mae = np.abs(preds_orig - actuals).mean()
    l_acc = 100 * (1 - l_mae / actuals.mean())
    print(f'LSTM    - MAE: {l_mae:.2f}, Approximate Accuracy: {l_acc:.2f}%')
    
    if l_acc < 95:
        print("LSTM Accuracy below 95%, retrying with more epochs or better tuning might be needed.")
    else:
        print("LSTM Accuracy goal achieved!")

except Exception as e:
    print(f'Could not complete evaluation: {e}')
