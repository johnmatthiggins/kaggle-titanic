#!/usr/bin/env python3
import sys
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

TEST_DATA = 'data/test.csv'
TRAIN_DATA = 'data/train.csv'

nan_count = 0
def main():
    train_data = pd.read_csv(TRAIN_DATA)
    cleaned_data = clean_data(train_data)
    train_X = cleaned_data[0]
    train_Y = cleaned_data[1]

    print(non_numeric_ticket_blobs(train_X))
    print(a_to_z())
    exit()

    model, accuracy = train_model(train_X, train_Y)
    print(accuracy)

    if '--test' in sys.argv:
        file_path = sys.argv[2]
        print(f'writing answers to {file_path}')
        test_model(model, file_path)


# returns A - Z
def a_to_z():
    return list(map(lambda n: chr(97 + n).upper(), range(0, 26)))

def non_numeric_ticket_blobs(X):
    blobs = set()

    nondistinct_blobs = list(X.apply(grab_non_numeric_blobs, axis=1))
    
    for blob in nondistinct_blobs:
        for word in blob:
            blobs.add(word)

    return blobs


def grab_non_numeric_blobs(row):
    blob_set = set()
    ticket = row['Ticket']
    blobs = ticket.split(' ')

    for blob in blobs:
        if not blob.isnumeric():
            blob_set.add(blob)

    return blob_set


def test_model(model, results_file):
    test_X = pd.read_csv(TEST_DATA).drop('Name', axis=1)

    pred_Y = model.predict(test_X)
    test_X['Survived'] = pred_Y
    test_X[['Survived', 'PassengerId']].to_csv(results_file)


def clean_data(df):
    train_X = df.drop('Survived', axis=1).drop('Name', axis=1)

    columns = train_X.columns
    mean_age = train_X['Age'].mean()

    train_X = train_X.apply(lambda r: set_age_as_mean(r, mean_age), axis=1)

    train_X.to_csv('train_x.csv')
    train_Y = df['Survived']

    return (train_X, train_Y)


def train_model(train_X, train_Y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_X, train_Y)

    y_predictions = model.predict(train_X)
    accuracy = sk.metrics.accuracy_score(train_Y, y_predictions)

    return (model, accuracy)


def set_age_as_mean(row, mean):
    if pd.isna(row['Age']):
        row['Age'] = mean

    return row


if __name__ == '__main__':
    main()
