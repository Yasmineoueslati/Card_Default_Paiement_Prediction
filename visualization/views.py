import os
from pathlib import Path
from pprint import pprint

from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# from main.machine_learning_models.KNN import KNN


# from main.machine_learning_models.main import main

data_path = os.path.join(Path(__file__).resolve().parent, "data/data.csv")
df = pd.read_csv(data_path, delimiter=',', encoding='utf-8', header=1)

# X = df.iloc[:, 0:24].values
# Y = df.iloc[:, 24].values
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# data = main(X_train, X_test, y_train, y_test)
Result = [{'Classifier_Method': 'LogisticRegression',
           'Accuracy_Scores': [0.809, 0.809, 0.805, 0.7981, 0.8062],
           'Mean_Accuracy': 0.8055,
           'ROC_Curve_Data': [[0.0, 0.0195, 1.0], [0.0, 0.2052, 1.0]],
           'Classification_Report': {'0': {'precision': 0.8178, 'recall': 0.9805, 'f1-score': 0.8918, 'support': 7060},
                                     '1': {'precision': 0.7425, 'recall': 0.2052, 'f1-score': 0.3215, 'support': 1940},
                                     'accuracy': 0.8133,
                                     'macro avg': {'precision': 0.7802, 'recall': 0.5928, 'f1-score': 0.6066,
                                                   'support': 9000},
                                     'weighted avg': {'precision': 0.8016, 'recall': 0.8133, 'f1-score': 0.7689,
                                                      'support': 9000}}},
          {'Classifier_Method': 'DecisionTree', 'Accuracy_Scores': [0.7136, 0.7255, 0.709, 0.7169, 0.7317],
           'Mean_Accuracy': 0.7193,
           'ROC_Curve_Data': [[0.0, 0.3911, 1.0], [0.0, 0.418, 1.0]],
           'Classification_Report': {'0': {'precision': 0.792, 'recall': 0.6089, 'f1-score': 0.6885, 'support': 7060},
                                     '1': {'precision': 0.227, 'recall': 0.418, 'f1-score': 0.2943, 'support': 1940},
                                     'accuracy': 0.5678,
                                     'macro avg': {'precision': 0.5095, 'recall': 0.5135, 'f1-score': 0.4914,
                                                   'support': 9000},
                                     'weighted avg': {'precision': 0.6702, 'recall': 0.5678, 'f1-score': 0.6035,
                                                      'support': 9000}}},
          {'Classifier_Method': 'SVM', 'Accuracy_Scores': [0.8195, 0.815, 0.8138, 0.8088, 0.8124],
           'Mean_Accuracy': 0.8139,
           'ROC_Curve_Data': [[0.0, 0.03, 1.0], [0.0, 0.2485, 1.0]],
           'Classification_Report': {'0': {'precision': 0.8245, 'recall': 0.97, 'f1-score': 0.8913, 'support': 7060},
                                     '1': {'precision': 0.6945, 'recall': 0.2485, 'f1-score': 0.366, 'support': 1940},
                                     'accuracy': 0.8144,
                                     'macro avg': {'precision': 0.7595, 'recall': 0.6092, 'f1-score': 0.6287,
                                                   'support': 9000},
                                     'weighted avg': {'precision': 0.7965, 'recall': 0.8144, 'f1-score': 0.7781,
                                                      'support': 9000}}},
          {'Classifier_Method': 'RandomForest', 'Accuracy_Scores': [0.8179, 0.8167, 0.8129, 0.811, 0.8174],
           'Mean_Accuracy': 0.8151,
           'ROC_Curve_Data': [[0.0, 0.0431, 1.0], [0.0, 0.2402, 1.0]],
           'Classification_Report': {'0': {'precision': 0.8209, 'recall': 0.9569, 'f1-score': 0.8837, 'support': 7060},
                                     '1': {'precision': 0.6052, 'recall': 0.2402, 'f1-score': 0.3439, 'support': 1940},
                                     'accuracy': 0.8024,
                                     'macro avg': {'precision': 0.713, 'recall': 0.5986, 'f1-score': 0.6138,
                                                   'support': 9000},
                                     'weighted avg': {'precision': 0.7744, 'recall': 0.8024, 'f1-score': 0.7674,
                                                      'support': 9000}}},
          {'Classifier_Method': 'KNeighbors', 'Accuracy_Scores': [0.8086, 0.8102, 0.8024, 0.7998, 0.8057],
           'Mean_Accuracy': 0.8053,
           'ROC_Curve_Data': [[0.0, 0.0409, 1.0], [0.0, 0.2577, 1.0]],
           'Classification_Report': {'0': {'precision': 0.8246, 'recall': 0.9591, 'f1-score': 0.8868, 'support': 7060},
                                     '1': {'precision': 0.6337, 'recall': 0.2577, 'f1-score': 0.3664, 'support': 1940},
                                     'accuracy': 0.8079,
                                     'macro avg': {'precision': 0.7292, 'recall': 0.6084, 'f1-score': 0.6266,
                                                   'support': 9000},
                                     'weighted avg': {'precision': 0.7835, 'recall': 0.8079, 'f1-score': 0.7746,
                                                      'support': 9000}}},
          {'Classifier_Method': 'Naïve Bayesian classifier(NB)',
           'Accuracy_Scores': [0.6005, 0.6126, 0.5238, 0.7081, 0.5343], 'Mean_Accuracy': 0.5959,
           'ROC_Curve_Data': [[0.0, 0.4948, 1.0], [0.0, 0.7624, 1.0]],
           'Classification_Report': {'0': {'precision': 0.8856, 'recall': 0.5052, 'f1-score': 0.6434, 'support': 7060},
                                     '1': {'precision': 0.2975, 'recall': 0.7624, 'f1-score': 0.428, 'support': 1940},
                                     'accuracy': 0.5607,
                                     'macro avg': {'precision': 0.5915, 'recall': 0.6338, 'f1-score': 0.5357,
                                                   'support': 9000},
                                     'weighted avg': {'precision': 0.7588, 'recall': 0.5607, 'f1-score': 0.597,
                                                      'support': 9000}}},
          {'Classifier_Method': 'DecisionTree', 'Accuracy_Scores': [0.8176, 0.8233, 0.8174, 0.8121, 0.8143],
           'Mean_Accuracy': 0.817,
           'ROC_Curve_Data': [[0.0, 0.0381, 1.0], [0.0, 0.3314, 1.0]],
           'Classification_Report': {'0': {'precision': 0.8396, 'recall': 0.9619, 'f1-score': 0.8966, 'support': 7060},
                                     '1': {'precision': 0.705, 'recall': 0.3314, 'f1-score': 0.4509, 'support': 1940},
                                     'accuracy': 0.826,
                                     'macro avg': {'precision': 0.7723, 'recall': 0.6467, 'f1-score': 0.6738,
                                                   'support': 9000},
                                     'weighted avg': {'precision': 0.8106, 'recall': 0.826, 'f1-score': 0.8005,
                                                      'support': 9000}}}]

from operator import attrgetter

columns = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

histogram_data = {}
for column in columns:
    dataValues = []
    dataLabels = []
    count = pd.cut(df[column], [0,1,2,3]).value_counts().sort_index()
    for i, row in count.items():
        dataValues.append(row)
        dataLabels.extend((i.left, i.right))
        dataLabels = list(dict.fromkeys(dataLabels))

    histogram_data[column] = [dataValues, dataLabels]

# pprint(histogram_data)


# Create your views here.
def home(request):
    accuracy_manhattan = [0.7184, 0.7793, 0.7677, 0.7924, 0.7894, 0.7997, 0.7977, 0.8010, 0.8026, 0.8030, 0.8028, 0.8041, 0.8044, 0.8026, 0.8030, 0.8016, 0.8033, 0.8044, 0.8050, 0.8058, 0.8061, 0.8053, 0.8066, 0.8054, 0.8048]
    accuracy_euclidean = [0.7143, 0.7797, 0.7659, 0.7937, 0.7903, 0.8019, 0.7979, 0.8019, 0.8016, 0.8044, 0.8045, 0.8036, 0.8041, 0.8037, 0.8059, 0.8049, 0.8063, 0.8043, 0.806, 0.8051, 0.8056, 0.8052, 0.8047, 0.8044, 0.8049]
    context = {
        'data': df[:200],
        'accuracy_manhattan': accuracy_manhattan,
        'accuracy_euclidean': accuracy_euclidean,
        'result': Result,
        'histogram_data': histogram_data,
    }
    # print(context["result"])
    return render(request, 'dashboard.html', context)


def predict_form(request):
    return render(request, 'predict.html')


def prediction(request):

    # Save the model in the notebook
    # import pickle
    # filename = 'finalized_model.sav'
    # pickle.dump(final_model, open(filename, 'wb'))

    import pickle
    from sklearn.tree import DecisionTreeClassifier
    model_path = os.path.join(Path(__file__).resolve().parent, "finalized_model.sav")

    X1 = int(request.GET.get('X1'))
    X2 = int(request.GET.get('X2'))
    X3 = int(request.GET.get('X3'))
    X4 = int(request.GET.get('X4'))
    X5 = int(request.GET.get('X5'))
    X6 = int(request.GET.get('X6'))
    X7 = int(request.GET.get('X7'))
    X8 = int(request.GET.get('X8'))
    X9 = int(request.GET.get('X9'))
    X10 = int(request.GET.get('X10'))
    X11 = int(request.GET.get('X11'))
    X12 = int(request.GET.get('X12'))
    X13 = int(request.GET.get('X13'))
    X14 = int(request.GET.get('X14'))
    X15 = int(request.GET.get('X15'))
    X16 = int(request.GET.get('X16'))
    X17 = int(request.GET.get('X17'))
    X18 = int(request.GET.get('X18'))
    X19 = int(request.GET.get('X19'))
    X20 = int(request.GET.get('X20'))
    X21 = int(request.GET.get('X21'))
    X22 = int(request.GET.get('X22'))
    X23 = int(request.GET.get('X23'))

    # Load Model
    final_model = pickle.load(open(model_path, 'rb'))
    # Remove X17 : BILL_AMT61
    result = final_model.predict([[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X18,X19,X20,X21,X22,X23]])


    result2 = False
    if X1 == 1:
        result2 = True

    # Result : True ==> The client is Default
    return JsonResponse({'Result': bool(result[0])})


# array2 = []
# array = [0.7142666666666667, 0.7797333333333333, 0.7658666666666667, 0.7937333333333333, 0.7902666666666667, 0.8018666666666666, 0.7978666666666666, 0.8018666666666666, 0.8016, 0.8044, 0.8045333333333333, 0.8036, 0.8041333333333334, 0.8037333333333333, 0.8058666666666666, 0.8049333333333333, 0.8062666666666667, 0.8042666666666667, 0.806, 0.8050666666666667, 0.8056, 0.8052, 0.8046666666666666, 0.8044, 0.8049333333333333]
# for item in array:
#     array2.append(round(item,4))
# print(array2)


DecisionTree_data = {'Classifier_Method': 'DecisionTree',
                     'Accuracy_Scores': [0.7136, 0.7255, 0.709, 0.7169, 0.7317],
                     'Mean_Accuracy': 0.7193,
                     'Error_Rate': 0.2807,
                     'ROC_Curve_Data': [[0.0, 0.3911, 1.0], [0.0, 0.418, 1.0]],
                     'ROC_AUC': 5,
                     }

LogisticRegression_data = {'Classifier_Method': 'Logistic Regression',
                           'Symbol': 'LR',
                           'Accuracy_Scores': "",
                           'Mean_Accuracy': "",
                           'ROC_Curve_Data_fpr': [0.0, 0.0891123439667129, 0.1967753120665742, 0.31119972260748957, 0.4429611650485437, 0.5996879334257975, 0.842753120665742, 1],
                           'ROC_Curve_Data_tpr': [0.0, 0.19466975666280417, 0.35979142526071844, 0.5127462340672074, 0.6593279258400927, 0.8030127462340672, 0.9490150637311703, 1],
                           'ROC_AUC': 0.651,
                           'Score': 0.7648269945667716,
                           'Error_Rate': 0.2351730054332284}
