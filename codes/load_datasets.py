import numpy as np
import pandas as pd
import random


def load_dataset(dataset_name):

    path = "C:\\Users\Ashkan\Documents\PycharmProjects\\active_learning\datasets\\"+dataset_name

    if dataset_name == 'iris':
        iris = pd.read_csv(path+'.data', header=None)
        iris = iris.sample(frac=1).reset_index(drop=True)

        X_iris = iris.iloc[:, 0:4].values
        y_iris = iris.iloc[:, 4].values
        for i in range(len(y_iris)):
            if(y_iris[i] == 'Iris-setosa'):
                y_iris[i] = 0
            elif(y_iris[i] == 'Iris-versicolor'):
                y_iris[i] = 1
            else:
                y_iris[i] = 2

        x = X_iris
        y = y_iris

    elif dataset_name == 'ecoli':
        ecoli = pd.read_csv(path+'.data', header=None, delim_whitespace=True)
        x = ecoli.iloc[:, 1:8].values
        y = ecoli.iloc[:, 8].values
        for i in range(len(y)):
            if y[i] == 'imL':
                y[i] = 0
            elif y[i] == 'imU':
                y[i] = 1
            elif y[i] == 'pp':
                y[i] = 2
            elif y[i] == 'om':
                y[i] = 3
            elif y[i] == 'omL':
                y[i] = 4
            elif y[i] == 'im':
                y[i] = 5
            elif y[i] == 'cp':
                y[i] = 6
            else: y[i] = 7
    elif dataset_name == 'glass':
        glass = pd.read_csv(path+'.data', header=None)
        x = glass.iloc[:, 1:10].values
        y = glass.iloc[:, 10].values

    elif dataset_name == 'pima':
        pimaIndiansDiabetes = pd.read_csv(path+'.data', header=None)
        x = pimaIndiansDiabetes.iloc[:, 0:8].values
        y = pimaIndiansDiabetes.iloc[:, 8].values

    elif dataset_name == 'sonar':
        sonarAll = pd.read_csv(path+'.data', header=None)
        x = sonarAll.iloc[:, 0:60].values
        y = sonarAll.iloc[:, 60].values
        for i in range(len(y)):
            if y[i] == 'R':
                y[i] = 0
            elif y[i] == 'M':
                y[i] = 1
    elif dataset_name == 'wine':
        wine = pd.read_csv(path+'.data', header=None)
        x = wine.iloc[:, 1:14].values
        y = wine.iloc[:, 0].values
    elif dataset_name == 'soybean': #### it has 1 more feature than the project document
        soybeanSmall = pd.read_csv(path + '.data', header=None)
        x = soybeanSmall.iloc[:, 0:35].values
        y = soybeanSmall.iloc[:, 35].values
        for i in range(len(y)):
            if y[i] == 'D1':
                y[i] = 0
            elif y[i] == 'D2':
                y[i] = 1
            elif y[i] == 'D3':
                y[i] = 2
            elif y[i] == 'D4':
                y[i] = 3
    elif dataset_name == 'ionosphere': #### it has 3 data fewer than doc and 2 feature more
        ionosphere = pd.read_csv(path + '.data', header=None)
        x = ionosphere.iloc[:, 0:-2].values
        y = ionosphere.iloc[:, -1].values
        for i in range(len(y)):
            if y[i] == 'g':
                y[i] = 0
            elif y[i] == 'b':
                y[i] = 1
    elif dataset_name == 'balance':
        balance = pd.read_csv(path+'.data', header=None)
        x = balance.iloc[:, 1:].values
        y = balance.iloc[:, 0].values
        for i in range(len(y)):
            if y[i] == 'B':
                y[i] = 0
            elif y[i] == 'L':
                y[i] = 1
            elif y[i] == 'R':
                y[i] = 2

    elif dataset_name == 'breast':
        breast = pd.read_csv(path+'.data', header=None)
        x = breast.iloc[:, 0:-1].values
        y = breast.iloc[:, -1].values
        # removed_index = []
        n = len(x)-16
        for i in range(n):
            if x[i,6] == '?':
                x = np.delete(x, (i), axis=0)
                y = np.delete(y, (i), axis=0)

                # removed_index.append(i)

            # else:
            #     x[i,6] = float(x[i,6])
        # print(removed_index)

        # for i in removed_index:
        #     x = np.delete(x,(i),axis=0)
        #     y = np.delete(y,(i),axis=0)

        for i in range(len(x)):
            x[i,6] = float(x[i,6])

        for i in range(len(y)):
            if y[i] == 2:
                y[i] = 0
            elif y[i] == 4:
                y[i] = 1

    k = len(set(y))

    return x, y, k

