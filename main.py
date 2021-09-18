import pandas as pd

data = pd.read_csv("/Users/williammckeon/Sync/youtube videos/dataanalysis/housing.csv")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
new_scaler = scaler.fit_transform(data["total_bedrooms"].array.reshape(-1, 1))
data.drop(columns="total_bedrooms", inplace=True)
data["total_bedrooms"] = new_scaler
print(data["total_bedrooms"].head())
