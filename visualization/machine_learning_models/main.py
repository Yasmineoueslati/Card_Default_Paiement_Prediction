import os
from pathlib import Path

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# Best parameters
# from .machine_learning_models.KNN import KNN
# from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

# ROC Curve
from sklearn.metrics import roc_curve, auc
# Classification report
from sklearn.metrics import classification_report



data_path = os.path.join(Path(__file__).resolve().parent.parent, "data\data.csv")
df = pd.read_csv(data_path, delimiter=',', encoding='utf-8', header=1)
X = df.iloc[:, 0:24].values
Y = df.iloc[:, 24].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Get best parameters KNN
# KNN_parameters = KNN(X_train, X_test, y_train, y_test)
# KNeighborsClassifier(KNN_parameters["Best_k"], metric=KNN_parameters["Best_metric"]),

# Get best parameters DecisionTreeClassifier
# param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1,10)}
# grid = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid=param_grid, cv=5)
# grid.fit(X_train, y_train)
# print(grid.best_params_)


def main(X_train, X_test, y_train, y_test):
    # LogisticRegression, DecisionTreeClassifier, SVC, RandomForestClassifier, KNN, Naïve Bayesian classifier (NB)
    Classifiers = [LogisticRegression(solver='lbfgs', max_iter=500),
                   DecisionTreeClassifier(),
                   SVC(),
                   RandomForestClassifier(n_estimators=100),
                   KNeighborsClassifier(23, metric='manhattan'),
                   GaussianNB(),
                   DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=2)]
    Result = []
    Different_Accuracies = []

    for Classifier in Classifiers:
        print(Classifier, " : Start")
        accuray_scores = cross_val_score(Classifier, X_train, y_train)
        Mean_Accuracy = np.average(accuray_scores)

        rounded_accuray_scores = [round(elem, 4) for elem in accuray_scores]
        rounded_Mean_Accuracy = round(Mean_Accuracy, 4)
        Different_Accuracies.append(rounded_Mean_Accuracy)

        classifier_method = str(Classifier).split("(")[0].replace("Classifier","")
        if classifier_method == "SVC":
            classifier_method = "SVM"
        elif classifier_method =="GaussianNB":
            classifier_method = "Naïve Bayesian classifier(NB)"

        # ROC Curve
        y_pred_knn = Classifier.fit(X_train, y_train).predict(X_test)
        fpr2, tpr2, threshold2 = roc_curve(y_test, y_pred_knn)
        # Classification report
        report_info = classification_report(y_test, y_pred_knn, output_dict=True)
        for k1, v1 in report_info.items():
            try:
                report_info[k1] = round(v1, 4)
            except:
                for k2, v2 in report_info[k1].items():
                    report_info[k1][k2] = round(v2, 4)

        Classifier_info = dict(Classifier_Method=classifier_method,
                               Accuracy_Scores=rounded_accuray_scores,
                               Mean_Accuracy=rounded_Mean_Accuracy,
                               ROC_Curve_Data=[[round(elem, 4) for elem in fpr2.tolist()], [round(elem, 4) for elem in tpr2.tolist()]],
                               Classification_Report=report_info)

        Result.append(Classifier_info)
        
    print(Result)
    print(Result[Different_Accuracies.index(max(Different_Accuracies))]['Classifier_Method'])

    Result = [{'Classifier_Method': 'LogisticRegression', 'Accuracy_Scores': [0.809, 0.809, 0.805, 0.7981, 0.8062], 'Mean_Accuracy': 0.8055, 'ROC_Curve_Data': [[0.0, 0.0195, 1.0], [0.0, 0.2052, 1.0]], 'Classification_Report': {'0': {'precision': 0.8178, 'recall': 0.9805, 'f1-score': 0.8918, 'support': 7060}, '1': {'precision': 0.7425, 'recall': 0.2052, 'f1-score': 0.3215, 'support': 1940}, 'accuracy': 0.8133, 'macro avg': {'precision': 0.7802, 'recall': 0.5928, 'f1-score': 0.6066, 'support': 9000}, 'weighted avg': {'precision': 0.8016, 'recall': 0.8133, 'f1-score': 0.7689, 'support': 9000}}}, {'Classifier_Method': 'DecisionTree', 'Accuracy_Scores': [0.7136, 0.7255, 0.709, 0.7169, 0.7317], 'Mean_Accuracy': 0.7193, 'ROC_Curve_Data': [[0.0, 0.3911, 1.0], [0.0, 0.418, 1.0]], 'Classification_Report': {'0': {'precision': 0.792, 'recall': 0.6089, 'f1-score': 0.6885, 'support': 7060}, '1': {'precision': 0.227, 'recall': 0.418, 'f1-score': 0.2943, 'support': 1940}, 'accuracy': 0.5678, 'macro avg': {'precision': 0.5095, 'recall': 0.5135, 'f1-score': 0.4914, 'support': 9000}, 'weighted avg': {'precision': 0.6702, 'recall': 0.5678, 'f1-score': 0.6035, 'support': 9000}}}, {'Classifier_Method': 'SVM', 'Accuracy_Scores': [0.8195, 0.815, 0.8138, 0.8088, 0.8124], 'Mean_Accuracy': 0.8139, 'ROC_Curve_Data': [[0.0, 0.03, 1.0], [0.0, 0.2485, 1.0]], 'Classification_Report': {'0': {'precision': 0.8245, 'recall': 0.97, 'f1-score': 0.8913, 'support': 7060}, '1': {'precision': 0.6945, 'recall': 0.2485, 'f1-score': 0.366, 'support': 1940}, 'accuracy': 0.8144, 'macro avg': {'precision': 0.7595, 'recall': 0.6092, 'f1-score': 0.6287, 'support': 9000}, 'weighted avg': {'precision': 0.7965, 'recall': 0.8144, 'f1-score': 0.7781, 'support': 9000}}}, {'Classifier_Method': 'RandomForest', 'Accuracy_Scores': [0.8179, 0.8167, 0.8129, 0.811, 0.8174], 'Mean_Accuracy': 0.8151, 'ROC_Curve_Data': [[0.0, 0.0431, 1.0], [0.0, 0.2402, 1.0]], 'Classification_Report': {'0': {'precision': 0.8209, 'recall': 0.9569, 'f1-score': 0.8837, 'support': 7060}, '1': {'precision': 0.6052, 'recall': 0.2402, 'f1-score': 0.3439, 'support': 1940}, 'accuracy': 0.8024, 'macro avg': {'precision': 0.713, 'recall': 0.5986, 'f1-score': 0.6138, 'support': 9000}, 'weighted avg': {'precision': 0.7744, 'recall': 0.8024, 'f1-score': 0.7674, 'support': 9000}}}, {'Classifier_Method': 'KNeighbors', 'Accuracy_Scores': [0.8086, 0.8102, 0.8024, 0.7998, 0.8057], 'Mean_Accuracy': 0.8053, 'ROC_Curve_Data': [[0.0, 0.0409, 1.0], [0.0, 0.2577, 1.0]], 'Classification_Report': {'0': {'precision': 0.8246, 'recall': 0.9591, 'f1-score': 0.8868, 'support': 7060}, '1': {'precision': 0.6337, 'recall': 0.2577, 'f1-score': 0.3664, 'support': 1940}, 'accuracy': 0.8079, 'macro avg': {'precision': 0.7292, 'recall': 0.6084, 'f1-score': 0.6266, 'support': 9000}, 'weighted avg': {'precision': 0.7835, 'recall': 0.8079, 'f1-score': 0.7746, 'support': 9000}}}, {'Classifier_Method': 'Naïve Bayesian classifier(NB)', 'Accuracy_Scores': [0.6005, 0.6126, 0.5238, 0.7081, 0.5343], 'Mean_Accuracy': 0.5959, 'ROC_Curve_Data': [[0.0, 0.4948, 1.0], [0.0, 0.7624, 1.0]], 'Classification_Report': {'0': {'precision': 0.8856, 'recall': 0.5052, 'f1-score': 0.6434, 'support': 7060}, '1': {'precision': 0.2975, 'recall': 0.7624, 'f1-score': 0.428, 'support': 1940}, 'accuracy': 0.5607, 'macro avg': {'precision': 0.5915, 'recall': 0.6338, 'f1-score': 0.5357, 'support': 9000}, 'weighted avg': {'precision': 0.7588, 'recall': 0.5607, 'f1-score': 0.597, 'support': 9000}}}, {'Classifier_Method': 'DecisionTree', 'Accuracy_Scores': [0.8176, 0.8233, 0.8174, 0.8121, 0.8143], 'Mean_Accuracy': 0.817, 'ROC_Curve_Data': [[0.0, 0.0381, 1.0], [0.0, 0.3314, 1.0]], 'Classification_Report': {'0': {'precision': 0.8396, 'recall': 0.9619, 'f1-score': 0.8966, 'support': 7060}, '1': {'precision': 0.705, 'recall': 0.3314, 'f1-score': 0.4509, 'support': 1940}, 'accuracy': 0.826, 'macro avg': {'precision': 0.7723, 'recall': 0.6467, 'f1-score': 0.6738, 'support': 9000}, 'weighted avg': {'precision': 0.8106, 'recall': 0.826, 'f1-score': 0.8005, 'support': 9000}}}]

# main(X_train, X_test, y_train, y_test)

