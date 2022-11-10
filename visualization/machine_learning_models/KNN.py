import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier


# Django 3.0+
from django.templatetags.static import static

url = static('../../static/classifiers/KNN.png')


data_path = os.path.join(Path(__file__).resolve().parent.parent, "data/data.csv")
df = pd.read_csv(data_path, delimiter=',', encoding='utf-8', header=1)



def KNN(X_train, X_test, y_train, y_test):

    accuracy_manhattan = []
    accuracy_euclidean = []

    for i in range(1, 26):

        # Manhattan
        knn_manhattan = KNeighborsClassifier(i, metric='manhattan')
        knn_manhattan_model = knn_manhattan.fit(X_train, y_train)
        accuracy_manhattan.append(knn_manhattan_model.score(X_test, y_test))

        # Euclidean
        knn_euclidean = KNeighborsClassifier(i, metric='euclidean')
        knn_euclidean_model = knn_euclidean.fit(X_train, y_train)
        accuracy_euclidean.append(knn_euclidean_model.score(X_test, y_test))

    print(accuracy_manhattan)
    print(accuracy_euclidean)

    maximum_accuracy_manhattan = max(accuracy_manhattan)
    maximum_accuracy_euclidean = max(accuracy_euclidean)

    if maximum_accuracy_manhattan >= maximum_accuracy_euclidean:
        best_metric = "manhattan"
        best_accuracy = maximum_accuracy_manhattan
        best_k = accuracy_manhattan.index(best_accuracy) + 1
    else:
        best_metric = "euclidean"
        best_accuracy = maximum_accuracy_euclidean
        best_k = accuracy_euclidean.index(best_accuracy) + 1

    # KNN on best metrics
    knn = KNeighborsClassifier(best_k, metric=best_metric)
    knn_model = knn.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    KNN_accuracy_training_set = knn_model.score(X_train, y_train)
    KNN_accuracy_test_set = knn_model.score(X_test, y_test)

    data = {
        "Manhattan_data": accuracy_manhattan,
        "Euclidean_data": accuracy_euclidean,
        "Best_metric": best_metric,
        "Best_accuracy": best_accuracy,
        "Best_k": best_k,
        "Accuracy of K-NN classifier on training set": KNN_accuracy_training_set,
        "Accuracy of K-NN classifier on test set": KNN_accuracy_test_set,
    }

    print(best_accuracy)
    print(KNN_accuracy_test_set)
    return data

# Resultat
# accuracy_manhattan = [0.7184, 0.7793, 0.7677, 0.7924, 0.7894, 0.7997, 0.7977, 0.8010, 0.8026, 0.8030, 0.8028, 0.8041,0.8044, 0.8026, 0.8030, 0.8016, 0.8033, 0.8044, 0.8050, 0.8058, 0.8061, 0.8053, 0.8066, 0.8054,0.8048]
# accuracy_euclidean = [0.7143, 0.7797, 0.7659, 0.7937, 0.7903, 0.8019, 0.7979, 0.8019, 0.8016, 0.8044, 0.8045, 0.8036,0.8041, 0.8037, 0.8059, 0.8049, 0.8063, 0.8043, 0.806, 0.8051, 0.8056, 0.8052, 0.8047, 0.8044,0.8049]
