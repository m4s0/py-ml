import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases/iris/iris.data', header=None)
    print(df.tail())

    #############################################################################
    print(50 * '=')
    print('Plotting the Iris data')
    print(50 * '-')

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    # plt.tight_layout()
    # plt.savefig('./images/02_06.png', dpi=300)
    plt.show()

    return X, y
