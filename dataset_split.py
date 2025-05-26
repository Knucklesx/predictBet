import pandas as pd

def split_dataset(input_csv='features_brasileirao.csv', train_file='train.csv',
                  val_file='val.csv', test_file='test.csv'):
    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'])
    
    train = df[df['date'] <= '2019-12-31']
    val = df[(df['date'] > '2019-12-31') & (df['date'] <= '2020-12-31')]
    test = df[df['date'] > '2020-12-31']
    
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)
