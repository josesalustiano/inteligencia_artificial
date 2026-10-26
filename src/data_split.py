import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/processed', exist_ok=True)

# 1. Divisão da Base de Regressão (Alvo: Mental_Wellbeing_Score)
df_reg = pd.read_csv('data/processed/dataset_regression.csv')
X_reg = df_reg.drop(columns=['Mental_Wellbeing_Score'])
y_reg = df_reg['Mental_Wellbeing_Score']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# Salva os splits de regressão
X_train_reg.join(y_train_reg).to_csv('data/processed/train_regression.csv', index=False)
X_test_reg.join(y_test_reg).to_csv('data/processed/test_regression.csv', index=False)
print(f"Regressão - Treino: {X_train_reg.shape[0]} linhas | Teste: {X_test_reg.shape[0]} linhas")

# 2. Divisão da Base de Classificação (Alvo: Addiction_Level)
df_clf = pd.read_csv('data/processed/dataset_classification.csv')
X_clf = df_clf.drop(columns=['Addiction_Level'])
y_clf = df_clf['Addiction_Level']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

X_train_clf.join(y_train_clf).to_csv('data/processed/train_classification.csv', index=False)
X_test_clf.join(y_test_clf).to_csv('data/processed/test_classification.csv', index=False)
print(f"Classificação - Treino: {X_train_clf.shape[0]} linhas | Teste: {X_test_clf.shape[0]} linhas")