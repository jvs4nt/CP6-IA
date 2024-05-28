from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open('baseball_wins_model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = StandardScaler()
scaler.fit_transform(data[['teams']])

@app.post('/predicao/')
def predict(data: dict):
    df = pd.DataFrame([data])
    df[['teams']] = scaler.transform(df[['teams']]) 
    df = pd.get_dummies(df, columns=['year', 'average_age', 'runs_per_game'], drop_first=True)
    
    for col in X_train.columns:
        if col not in df.columns:
            df[col] = 0
            
    df = df[X_train.columns] 
    
    prediction = model.predict(df)
    return {'prediction': prediction[0]}