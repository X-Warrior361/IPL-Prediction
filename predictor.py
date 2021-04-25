import joblib
import pandas as pd
from itertools import combinations
import random
import numpy as np


def predictRuns(testInput):
    # Loading the joblib files
    with open('model_bat.joblib', 'rb') as f:
        model_bat = joblib.load(f)
    with open('model_bowl.joblib', 'rb') as f:
        model_bowl = joblib.load(f)
    with open('venue.joblib', 'rb') as f:
        venue = joblib.load(f)
    with open('player.joblib', 'rb') as f:
        player = joblib.load(f)

    # Preparing the data
    data = pd.read_csv(testInput)
    data_bat = data[['venue', 'innings', 'batsmen']]
    data_bowl = data[['venue', 'innings', 'bowlers']]

    # Transforming the venue and batsmen
    data_bat = (data_bat.set_index(['venue', 'innings']).apply(lambda x: x.str.split(',').explode()).reset_index())
    data_bat['venue'] = venue.transform(data_bat['venue'])
    data_bat['batsmen'] = player.transform(data_bat['batsmen'])

    # Predicting the batsmen score
    ball = (38 - data_bat.shape[0]) / data_bat.shape[0]
    test_case_bat = data_bat.to_numpy()
    prediction_bat = model_bat.predict(test_case_bat)
    print(prediction_bat )
    prediction_bat *= ball

    # Transforming the venue and bowler
    data_bowl = (data_bowl.set_index(['venue', 'innings']).apply(lambda x: x.str.split(',').explode()).reset_index())
    data_bowl['bowlers'] = player.transform(data_bowl['bowlers'])
    data_bowl['venue'] = venue.transform(data_bowl['venue'])

    # Predicting the bowler score
    test_case_bowl = data_bowl.to_numpy()
    prediction_bowl = model_bowl.predict(test_case_bowl)
    n = 6 % data_bowl.shape[0]
    if n == 0:
        prediction_bowl *= 6 / data_bowl.shape[0]
    else:
        choice = random.choice(list(combinations(prediction_bowl, n)))
        prediction_bowl = np.append(prediction_bowl, choice)

    print(prediction_bat)
    print(prediction_bowl)
    print(prediction_bat.sum())
    print(prediction_bowl.sum())
    prediction = (prediction_bowl.sum() + prediction_bat.sum()) / 2     # Getting the average
    return round(prediction)
