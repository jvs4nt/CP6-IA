from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open('player_salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = StandardScaler()
scaler.fit_transform(data[['seson17_18']])

@app.post('/predicao/')
def predict(data: dict):
    df = pd.DataFrame([data])
    df[['seson17_18']] = scaler.transform(df[['seson17_18']]) 
    df = pd.get_dummies(df, columns=['Player', 'tm', 'seson17_18'], drop_first=True)
    
    for col in X_train.columns:
        if col not in df.columns:
            df[col] = 0
            
    df = df[X_train.columns] 
    
    prediction = model.predict(df)
    return {'prediction': prediction[0]}