import json
import os

import numpy as np
import pandas as pd


def json_to_list(filename):
    with open(filename) as trainFile:
        data = json.load(trainFile)

    df = pd.DataFrame()

    frame = df.from_dict(data['comments'], orient='columns')

    # dataframe conversion into list type to feed it into CountVectorizer fit_transform
    return frame[['body']].values.flatten().tolist()


def json_to_csv(input, output, label):
    df = pd.DataFrame()

    for file in os.listdir(input):
        if file.endswith(".json"):
            with open(os.path.join(input, file), 'r', encoding='utf-8') as infile:
                data = json.load(infile)

            frame = pd.DataFrame.from_dict(data['comments'], orient='columns')
            df = df.append([[v, label] for v in frame[['body']].values.flatten().tolist()], ignore_index=True)

    df.to_csv(output, index=False)


def merge_and_shuffle(input, output):
    df = pd.DataFrame()

    for file in input:
        _file = pd.read_csv(file, index_col=None, header=None)
        df = df.append(_file, ignore_index=True)

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(output, index=False)
