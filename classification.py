import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from svm import SVM

def perform_svm_classification(file_path, split_ratio, c_value):
    data = pd.read_csv(file_path)

    features = data.drop(columns=['Rev_Type']).values
    target = data['Rev_Type'].values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=split_ratio, random_state=2)

    clf = SVM(C=c_value)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    TP, TN, FP, FN = confusion_matrix(y_test, y_pred)

    accuracy, precision, recall, f1_score = calculate_metrics(TP, TN, FP, FN)

    return TP, TN, FP, FN, accuracy, precision, recall, f1_score

def confusion_matrix(y_true, y_pred):
  TP = np.sum((y_true == 1) & (y_pred == 1))
  TN = np.sum((y_true == -1) & (y_pred == -1))
  FP = np.sum((y_true == -1) & (y_pred == 1))
  FN = np.sum((y_true == 1) & (y_pred == -1))

  return TP, TN, FP, FN

def calculate_metrics(TP, TN, FP, FN):
    accuracy_score = (TP + TN) / (TP + TN + FP + FN)
    precision_score = TP / (TP + FP)
    recall_score = TP / (TP + FN)
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    return accuracy_score, precision_score, recall_score, f1_score
