def initial(): 
    import pandas as pd
    path = 'data/train.csv'
    df = pd.read_csv(path)
    print(df.head(5))

    for col in df.columns:
        print(f'Column {col} contains: {len(df[col].unique())} values')

    ratings = df.columns[10:]
    for rat in ratings:
        print(f"{rat} : {sorted(df[rat].unique())}")

if __name__ == "__main__":
    initial()
    return True