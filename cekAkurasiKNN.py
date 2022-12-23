import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def klasifikasiKNN(xx):
    path = xx
    dataset = pd.read_excel(path, header=None)
    X = dataset.iloc[1:, :40].values
    y = dataset.iloc[1:, 40].values
    ujiCoba = 0.05
    iters=1
    while(iters<=3):
        print("")
        formating = int(ujiCoba*100)
        print(f'Data uji {formating}% dari dataset')
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ujiCoba)
        print(f'(Jumlah data latih, Jumlah fitur perdata) = {x_train.shape}')
        print(f'(Jumlah data test, Jumlah fitur perdata) = {x_test.shape}')
        iterss=1
        n_tetangga = 5
        while(iterss<=3):
            grid_params = {
                'n_neighbors': [n_tetangga],
                'weights': ['distance'],
                'metric': ['euclidean']
            }
            model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
            model.fit(x_train, y_train)
            print(f'Akurasi n={n_tetangga}\t: {model.score(x_test, y_test)}')
            n_tetangga+=4
            iterss+=1
        iters+=1
        ujiCoba=ujiCoba+0.05

    print("Done")

cekAkurasi = klasifikasiKNN('./model/WFCCTrain.xlsx')
cekAkurasi
