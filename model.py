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
    train(train_data)


def train(train_data):
    train_X = clean_data(train_data.drop('Survived', axis=1))
    train_Y = train_data['Survived']

    model, accuracy = train_model(train_X, train_Y)
    print(accuracy)

    if '--test' in sys.argv:
        file_path = sys.argv[2]
        print(f'writing answers to {file_path}')
        test_model(model, file_path)


# returns A - H
def a_to_h():
    return list(map(lambda n: chr(97 + n).upper(), range(0, 8)))


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
    test_X = clean_data(pd.read_csv(TEST_DATA))
    print(test_X)

    test_X.to_csv('test_x.csv')

    pred_Y = model.predict(test_X)
    test_X['Survived'] = pred_Y
    test_X[['Survived', 'PassengerId']].to_csv(results_file)


def clean_data(X):
    train_X = X.drop('Name', axis=1).drop('Ticket', axis=1)

    columns = train_X.columns
    mean_age = train_X['Age'].mean()
    mean_fare = train_X['Fare'].mean()

    train_X = train_X.apply(lambda r: set_age_as_mean(r, mean_age), axis=1)\
                     .apply(lambda r: set_fare_as_mean(r, mean_fare), axis=1)\
                     .join(create_cabin_columns(X)).drop('Cabin', axis=1)

    train_X = pd.get_dummies(train_X)

    train_X.to_csv('train_x.csv')

    return train_X


# key: 'Cabin'
def create_cabin_columns(df):
    dictionaries = df.apply(create_cabin_column_dict, axis=1)
    result_dict = dict()

    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key in result_dict:
                result_dict[key].append(value)
            else:
                result_dict[key] = [value]

    return pd.DataFrame.from_dict(result_dict)


def create_cabin_column_dict(row):
    letters = a_to_h()
    cabin_column_values = dict()

    for letter in letters:
        if letter in str(row['Cabin']):
            cabin_column_values[f'Cabin_{letter}'] = 1
        else:
            cabin_column_values[f'Cabin_{letter}'] = 0

    return cabin_column_values


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


def set_fare_as_mean(row, mean):
    if pd.isna(row['Fare']):
        row['Fare'] = mean

    return row


if __name__ == '__main__':
    main()
