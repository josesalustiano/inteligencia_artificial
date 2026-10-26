import pandas as pd
import time
import os
import psutil
from sklearn.tree import DecisionTreeClassifier
from codecarbon import EmissionsTracker

X_train = pd.read_csv('data/processed/train_classification.csv').drop(columns=['Addiction_Level'])
y_train = pd.read_csv('data/processed/train_classification.csv')['Addiction_Level']

# Inicializa o rastreador de hardware e CO2
tracker = EmissionsTracker(project_name="decision_tree", output_dir="logs/")
tracker.start()
start_time = time.time()

# Treina o modelo vencedor (Modelo 2 Podado)
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=10, min_samples_split=2, random_state=42)
clf.fit(X_train, y_train)

# Captura métricas pós-treino
end_time = time.time()
emissions = tracker.stop()
ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

print(f"Tempo de Treinamento: {end_time - start_time:.4f} segundos")
print(f"Uso de RAM (Pico): {ram_usage:.2f} MB")
print(f"Emissões de CO2: {emissions:.8f} kg")