#!/usr/bin/env python3
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

TEST_DATA = 'data/test.csv'
TRAIN_DATA = 'data/train.csv'

def main():
    train_data = pd.read_csv(TRAIN_DATA)
    train_X = pd.get_dummies(train_data.drop('Survived', axis=1).drop('Name', axis=1), dummy_na=True)
    print(train_X)
    train_Y = train_data[['Survived']]

    model = RandomForestClassifier(n_estimators=20)
    model.fit(train_X, train_Y)

    y_predictions = model.predict(train_X)
    print(sk.metrics.accuracy_score(train_Y, y_predictions))


if __name__ == '__main__':
    main()
