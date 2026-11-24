import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

def avaliar_regressao(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred)
    }

def avaliar_classificacao_multiclasse(y_true, y_probs, classes=[0, 1, 2, 3]):
    y_pred = np.argmax(y_probs, axis=1)
    y_true_bin = label_binarize(y_true, classes=classes)
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ROC_AUC_macro": roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"),
        "PR_AUC_macro": average_precision_score(y_true_bin, y_probs, average="macro"),
        "Conf_Matrix": confusion_matrix(y_true, y_pred).tolist()
    }