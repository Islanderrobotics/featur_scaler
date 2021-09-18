import pandas as pd
import IslanderDataPreprocessing as IR

data = pd.read_csv("/Users/williammckeon/Sync/youtube videos/dataanalysis/housing.csv")
encoder = IR.Encoder(data)
data = encoder.Check()
scaler = IR.FeatureScaler(data)
data = scaler.Check()

print(data[scaler.next_one].head())
