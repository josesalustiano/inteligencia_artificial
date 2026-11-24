import os
import time
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.tree import DecisionTreeClassifier
from codecarbon import EmissionsTracker

# ==============================================================================
# FUNÇÕES DA ATIVIDADE ATUAL (REDES NEURAIS / PYTORCH)
# ==============================================================================

def medir_inferencia(modelo, X_tensor, n_rep=30, n_aquecimento=5):
    """Mede tempo, vazão e memória apenas da etapa de inferência (PyTorch)."""
    modelo.eval()
    proc = psutil.Process(os.getpid())
    
    with torch.inference_mode():
        # Warm-up (aquecimento exigido no PDF)
        for _ in range(n_aquecimento):
            modelo(X_tensor)
            
        mem_antes = proc.memory_info().rss
        tempos = []
        
        for _ in range(n_rep):
            inicio = time.perf_counter()
            modelo(X_tensor)
            tempos.append(time.perf_counter() - inicio)
            
        mem_depois = proc.memory_info().rss
        
    media = np.mean(tempos)
    desvio = np.std(tempos)
    
    return {
        "tempo_medio_s": media,
        "tempo_desvio_s": desvio,
        "tempo_por_amostra_ms": 1000 * media / len(X_tensor),
        "vazao_amostras_s": len(X_tensor) / media,
        "delta_memoria_kb": abs(mem_depois - mem_antes) / 1024,
        "ram_total_mb": mem_depois / (1024 ** 2)
    }

def calcular_co2(tempo_horas, potencia_watts=65):
    """Calcula energia (kWh) e emissão de CO2 (g) usando o fator do Brasil (45,8 gCO2/kWh)."""
    energia_kwh = (potencia_watts * tempo_horas) / 1000
    emissao_co2_g = energia_kwh * 45.8
    return energia_kwh, emissao_co2_g


# ==============================================================================
# CÓDIGO DA ATIVIDADE ANTERIOR (ÁRVORE DE DECISÃO)
# Só executa se você rodar diretamente: python src/profiling.py
# ==============================================================================

if __name__ == "__main__":
    X_train = pd.read_csv('data/processed/train_classification.csv').drop(columns=['Addiction_Level'])
    y_train = pd.read_csv('data/processed/train_classification.csv')['Addiction_Level']

    tracker = EmissionsTracker(project_name="decision_tree", output_dir="logs/")
    tracker.start()
    start_time = time.time()

    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=10, min_samples_split=2, random_state=42)
    clf.fit(X_train, y_train)

    end_time = time.time()
    emissions = tracker.stop()
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    print(f"Tempo de Treinamento: {end_time - start_time:.4f} segundos")
    print(f"Uso de RAM (Pico): {ram_usage:.2f} MB")
    print(f"Emissões de CO2: {emissions:.8f} kg")