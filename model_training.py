import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

np.random.seed(42)

data_size = 2000

amount = np.random.normal(250, 80, data_size)
frequency = np.random.normal(6, 2, data_size)
location_risk = np.random.randint(0, 2, data_size)
device_risk = np.random.randint(0, 2, data_size)

fraud = (
    (amount > 400) & 
    (location_risk == 1) | 
    (device_risk == 1) & 
    (frequency > 10)
)

df = pd.DataFrame({
    "amount": amount,
    "frequency": frequency,
    "location_risk": location_risk,
    "device_risk": device_risk,
    "fraud": fraud.astype(int)
})

X = df.drop("fraud", axis=1)
y = df["fraud"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

fraud_model = LogisticRegression()
fraud_model.fit(X_scaled, y)

anomaly_model = IsolationForest(contamination=0.05)
anomaly_model.fit(X_scaled)

pickle.dump(fraud_model, open("fraud_model.pkl", "wb"))
pickle.dump(anomaly_model, open("anomaly_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Models trained and saved successfully.")
