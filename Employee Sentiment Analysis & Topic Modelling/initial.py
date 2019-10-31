import pandas as pd
path = 'data/train.csv'
df = pd.read_csv(path)
print(df.head(5))