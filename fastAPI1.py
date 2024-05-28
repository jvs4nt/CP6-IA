from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open('soccer_goals_model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = StandardScaler()
scaler.fit_transform(data[['goals']])

@app.post('/predicao/')
def predict(data: dict):
    df = pd.DataFrame([data])
    df[['goals']] = scaler.transform(df[['goals']]) 
    df = pd.get_dummies(df, columns=['goals_per_90', 'matches_played', 'min_playing_time'], drop_first=True)
    
    for col in X_train.columns:
        if col not in df.columns:
            df[col] = 0
            
    df = df[X_train.columns] 
    
    prediction = model.predict(df)
    return {'prediction': prediction[0]}