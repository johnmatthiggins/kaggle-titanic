#!/usr/bin/env python3
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

TEST_DATA = 'data/test.csv'
TRAIN_DATA = 'data/train.csv'

def main():
    train_data = pd.read_csv(TRAIN_DATA)

    train_X = pd.get_dummies(train_data.drop('Survived', axis=1).drop('Name', axis=1), dummy_na=True)

    columns = train_X.columns

    def print_nan(row, columns):
        for column in columns:
            if pd.isna(row[column]):
                print(f'{column} column has nan')

    print(train_X)

    train_Y = train_data[['Survived']]

    model = RandomForestClassifier(n_estimators=20)
    model.fit(train_X, train_Y)

    y_predictions = model.predict(train_X)
    print(sk.metrics.accuracy_score(train_Y, y_predictions))


if __name__ == '__main__':
    main()
