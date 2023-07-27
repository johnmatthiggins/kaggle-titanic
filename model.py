#!/usr/bin/env python3
import matplotlib.pyplot as plt
import missingno as msno
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

    test_X.to_csv('test_x.csv')

    pred_Y = model.predict(test_X)
    test_X['Survived'] = pred_Y
    test_X[['Survived', 'PassengerId']].to_csv(results_file, index=False)


def clean_data(X):
    msno.matrix(X)
    plt.show()
    ticket_columns = set(build_ticket_columns())
    print(ticket_columns)
    ticket_values = X[['Ticket']]
    ticket_values['Ticket'].transform(lambda e: e.replace('STON', 'SOTON'))

    train_X = X.drop('Name', axis=1).drop('Ticket', axis=1)

    columns = train_X.columns
    mean_age = train_X['Age'].median()
    mean_fare = train_X['Fare'].median()

    ticket_column_values = ticket_values\
            .apply(lambda r: derive_ticket_columns(r, ticket_columns), axis=1)\
            .drop('Ticket', axis=1)

    train_X = train_X.apply(lambda r: set_age_as_mean(r, mean_age), axis=1)\
                     .apply(lambda r: set_fare_as_mean(r, mean_fare), axis=1)\
                     .join(create_cabin_columns(X)).drop('Cabin', axis=1)\
                     .join(ticket_column_values)

    train_X = pd.get_dummies(train_X)
    train_X.to_csv('train_x.csv')

    return train_X


def derive_ticket_columns(row, column_names):
    numbers = list(filter(lambda s: s.isnumeric() and len(s) > 3, str(row['Ticket']).split(' ')))

    if len(numbers) == 0:
        row['TicketNumber'] = -1
    else:
        row['TicketNumber'] = float(numbers[0])
    for name in column_names:
        if name in row['Ticket']:
            row[str(name)] = 1
        else:
            row[str(name)] = 0

    return row


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


def cabin_number(row):
    segments = str(row['Cabin']).split(' ')
    numeric_segments = []
    
    for segment in segments:
        numeric_segment = ''
        for c in segment:
            if c.isnumeric():
                numeric_segment += c

        if len(numeric_segment) != 0:
            numeric_segments.append(numeric_segment)

    if len(numeric_segments) != 0:
        return float(numeric_segments[0])
    else:
        return -1


def create_cabin_column_dict(row):
    letters = a_to_h()
    cabin_column_values = dict()
    number = cabin_number(row)
    cabin_column_values['CabinNumber'] = number

    for letter in letters:
        if letter in str(row['Cabin']):
            cabin_column_values[f'Cabin_{letter}'] = 1
        else:
            cabin_column_values[f'Cabin_{letter}'] = 0

    return cabin_column_values


def train_model(train_X, train_Y):
    model = RandomForestClassifier(n_estimators=512)
    model.fit(train_X, train_Y)

    y_predictions = model.predict(train_X)
    accuracy = sk.metrics.accuracy_score(train_Y, y_predictions)

    return (model, accuracy)


def build_ticket_columns():
    test = pd.read_csv(TEST_DATA)[['Ticket']]
    train = pd.read_csv(TRAIN_DATA)[['Ticket']]
    data = pd.concat((test, train), axis=1)

    def extract_destinations(row):
        ticket_text = row['Ticket']

        if not pd.isna(ticket_text[0]):
            chunks = str(ticket_text[0]).split(' ')
        else:
            chunks = []

        return list(filter(lambda s: not s.isnumeric(), chunks))

    chunk_list = list(data.apply(extract_destinations, axis=1))
    
    return np.concatenate(chunk_list)


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
