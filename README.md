# Inteligência Artificial - Análise de Dados e Avaliação Comparativa de Modelos (UFRN)

**Aluno:** José Salustiano Neto Júnior  
**Matrícula:** 20240062041  
**Disciplina:** DCT1401 - Inteligência Artificial  
**Professor:** Thommas K. S. Flores  
**Dataset:** [Social Media Addiction and Mental Wellbeing](https://www.kaggle.com/datasets/harpartapsingh13/social-media-addiction-and-mental-wellbeing-dataset/data)

---

## 📌 Descrição do Projeto

Este repositório centraliza o desenvolvimento prático da disciplina de Inteligência Artificial, dividido em duas etapas complementares utilizando o dataset *"Social Media Addiction and Mental Wellbeing"*:

* **Etapa 1 — Análise Exploratória e Pré-processamento:** Diagnóstico da qualidade dos dados brutos, imputação de nulos, tratamento de outliers, padronização (Z-score) e seleção dos alvos de Regressão (`Mental_Wellbeing_Score`) e Classificação (`Addiction_Level`).
* **Etapa 2 — Avaliação Comparativa de Modelos:** Construção, benchmark e avaliação comparativa entre três famílias de modelos:
  1. **Sistema Especialista:** Base de regras SE-ENTÃO com motor de inferência próprio e medição do esforço humano de elaboração.
  2. **Árvore de Decisão:** Modelos podados e sem restrição (para evidenciar overfitting) para regressão e classificação.
  3. **Ensemble Learning:** Random Forest (com análise de viés/variância) e Gradient Boosting.
* **Etapa 3 — Redes Neurais Artificiais (MLP) em PyTorch:** Implementação de arquiteturas MLP parametrizáveis (`nn.Module`) para regressão e classificação multiclasse, estudo sistemático e controlado de hiperparâmetros com múltiplas sementes (8 fatores isolados + mapa de calor 2D + *random search*), persistência segura de checkpoints (`.pt` com `weights_only=True`), script independente de inferência em ambiente limpo (`src/inference.py`) e análise completa de desempenho, generalização (5-Fold CV) e eficiência computacional (latência, vazão, RAM, MACs, energia e emissões de CO₂).

---

## 📂 Estrutura do Repositório

```text
inteligencia_artificial/
├── data/
│   ├── raw/                             # Dataset original bruto inviolado
│   └── processed/                       # Conjuntos limpos e divididos em treino/teste (80/20 estratificado)
├── docs/
│   └── relatorio_final.pdf              # Relatório técnico final em PDF (LaTeX)
├── logs/
│   ├── cronometro_regras.csv            # [Etapa 2] Registro temporal de elaboração do Sistema Especialista
│   ├── emissions.csv                    # [Etapa 2] Logs de energia e CO2 (CodeCarbon)
│   └── hiperparametros.csv              # [Etapa 3] Log completo de todas as execuções e sementes do estudo de hiperparâmetros
├── models/
│   ├── rf_regression_model.pkl          # [Etapa 2] Modelo Random Forest serializado
│   ├── modelo_regressao.pt              # [Etapa 3] Checkpoint final MLP Regressão (pesos + config + scalers)
│   └── modelo_classificacao.pt          # [Etapa 3] Checkpoint final MLP Classificação (pesos + config + scaler)
├── notebooks/
│   ├── 01_eda.ipynb                     # [Etapa 1] Análise exploratória inicial
│   ├── 02_preprocessing.ipynb           # [Etapa 1] Limpeza, imputação e divisão treino/teste
│   ├── 03_regression_analysis.ipynb     # [Etapa 1] Análise do alvo contínuo
│   ├── 04_classification_analysis.ipynb # [Etapa 1] Análise do alvo categórico
│   ├── 05_decision_tree.ipynb           # [Etapa 2] Árvores de decisão
│   ├── 06_ensemble_learning.ipynb       # [Etapa 2] Random Forest e Gradient Boosting
│   ├── 07_classification_plots.ipynb    # [Etapa 2] Visualizações de classificação
│   ├── 08_regression_models.ipynb       # [Etapa 2] Ajuste e treino de regressão
│   ├── 09_generalization.ipynb          # [Etapa 2] Gap de generalização e K-Fold CV
│   ├── 10_efficiency.ipynb              # [Etapa 2] Profiling de tempo, RAM e CO2
│   ├── 11_model_visualization.ipynb    # [Etapa 2] Curvas ROC, PR-AUC e resíduos
│   ├── 12_final_metrics.ipynb           # [Etapa 2] Consolidação das métricas
│   ├── 13_table.ipynb                   # [Etapa 2] Tabela-resumo da Etapa 2
│   ├── 14_mlp_regressao.ipynb           # [Etapa 3] Baseline MLP Regressão e checkpoint inicial
│   ├── 15_mlp_classificacao.ipynb       # [Etapa 3] Baseline MLP Classificação Multiclasse e checkpoint inicial
│   ├── 16_estudo_hiperparametros.ipynb  # [Etapa 3] Exploração inicial de variações
│   ├── 17_avaliacao_final_profiling.ipynb # [Etapa 3] Avaliação preliminar e curvas
│   ├── 18_estudo_hiperparametros_completo.ipynb # [Etapa 3] Ablation (8 fatores, 3 sementes), Heatmap 2D e Random Search
│   ├── 19_avaliacao_fidelidade_generalizacao.ipynb # [Etapa 3] Fidelidade (.pt), métricas Aula 08, gráficos e 5-Fold CV
│   └── 20_eficiencia_computacional.ipynb # [Etapa 3] Profiling (psutil, MACs, energia/CO2), sensibilidade e resumo final
├── reports/
│   ├── figures/                         # Curvas de aprendizado, heatmaps, ROC/PR, matriz de confusão e resíduos
│   └── tables/                          # Tabelas consolidadas em CSV (resumo_final.csv, sensibilidade, K-Fold, profiling)
├── rules/
│   ├── regras_classificacao.txt         # [Etapa 2] Base de regras SE-ENTÃO (Classificação)
│   └── regras_regressao.txt             # [Etapa 2] Base de regras SE-ENTÃO (Regressão)
├── src/
│   ├── __init__.py
│   ├── data_split.py                    # Divisão reprodutível treino/teste
│   ├── expert_system.py                 # Motor de inferência do Sistema Especialista
│   ├── models.py                        # Classe MLP parametrizável em PyTorch
│   ├── train.py                         # Laço de treinamento com validação e early stopping
│   ├── metrics.py                       # Funções de cálculo das métricas de regressão e classificação
│   ├── inference.py                     # Script independente para carregar .pt e realizar inferência
│   └── profiling.py                     # Medição de tempo, memória e recursos
├── README.md                            # Documentação principal do repositório
└── requirements.txt                     # Dependências do ambiente Python
```

---

## 🚀 Como Executar

### 1. Configuração do Ambiente
* **Python:** `3.11+`
* **PyTorch:** `2.6+` (execução em CPU com `weights_only=True` e `map_location="cpu"`)
* **Principais dependências:** `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `psutil`, `codecarbon`

```bash
# 1. Clone o repositório
git clone <URL_DO_SEU_REPOSITORIO>
cd inteligencia_artificial

# 2. Crie e ative um ambiente virtual (opcional, recomendado)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

### 2. Ordem de Execução dos Notebooks
Para reproduzir os experimentos integralmente sem vazamento de dados, execute os notebooks na pasta `notebooks/` na seguinte ordem:

1. **Etapa 1 (Pré-processamento):** `01_eda.ipynb` ➔ `02_preprocessing.ipynb` ➔ `03_regression_analysis.ipynb` ➔ `04_classification_analysis.ipynb`
2. **Etapa 2 (Modelos Clássicos):** `05_decision_tree.ipynb` ➔ `06_ensemble_learning.ipynb` ➔ `07_classification_plots.ipynb` ➔ `08_regression_models.ipynb` ➔ `09_generalization.ipynb` ➔ `10_efficiency.ipynb` ➔ `11_model_visualization.ipynb` ➔ `12_final_metrics.ipynb` ➔ `13_table.ipynb`
3. **Etapa 3 (Redes Neurais PyTorch):** `14_mlp_regressao.ipynb` ➔ `15_mlp_classificacao.ipynb` ➔ `18_estudo_hiperparametros_completo.ipynb` ➔ `19_avaliacao_fidelidade_generalizacao.ipynb` ➔ `20_eficiencia_computacional.ipynb` *(os notebooks `16` e `17` registram ensaios preliminares de arquitetura).*

---

## 🤖 Inferência com os Modelos Salvos (Ambiente Limpo)

Os arquivos `models/modelo_regressao.pt` e `models/modelo_classificacao.pt` armazenam o `state_dict`, os hiperparâmetros da arquitetura, os nomes das features e os parâmetros do `StandardScaler` ajustados exclusivamente no treino. Para realizar previsões em novas amostras sem retreinar:

```bash
# Inferência — Regressão (Mental_Wellbeing_Score)
python src/inference.py --modelo models/modelo_regressao.pt --entrada data/processed/test_regression.csv

# Inferência — Classificação Multiclasse (Addiction_Level)
python src/inference.py --modelo models/modelo_classificacao.pt --entrada data/processed/test_classification.csv
```

---

## 📊 Principais Resultados

### Etapa 1 — Análise Exploratória e Alvos

* **Regressão:** Alvo `Mental_Wellbeing_Score` (alta variabilidade e distribuição normal aproximada).
* **Classificação:** Alvo `Addiction_Level` (isolamento do comportamento e grupos de risco).
* **Features Principais:** `Daily_Usage_Hours`, `FOMO_Score`, `Scroll_Without_Purpose`, `Late_Night_Usage` e `Posts_Per_Week`.

### Etapa 2 — Comparativo entre Modelos

* **Desempenho Preditivo:** O **Random Forest** obteve o melhor desempenho no teste em ambos os problemas (Regressão: RMSE = 0.475; Classificação: F1-Score = 0.885, ROC-AUC = 0.936), reduzindo a variância das árvores isoladas.
* **Generalização:** O **Random Forest** apresentou o menor gap de generalização (0.019), enquanto a **Árvore de Decisão** sem restrições sofreu severo *overfitting* (gap de 0.279).
* **Eficiência Computacional:** A **Árvore de Decisão Podada** provou ser a mais viável para cenários embarcados/edge (inferência em 0.0027 ms/amostra, alocação de apenas 132.8 KB de RAM e consumo energético com emissão irrisória de CO₂).
* **Esforço Humano vs. Treino:** O **Sistema Especialista** exigiu 95 minutos de esforço humano contínuo (cronometrado em 6 sessões) para codificar 21 regras, enquanto os modelos de Machine Learning foram treinados computacionalmente em frações de segundo (0.11s a 0.58s).

### Etapa 3 — Redes Neurais MLP em PyTorch
* **Hiperparâmetros mais críticos:** Taxa de aprendizado (η), função de ativação (`GELU` na regressão e `Tanh` na classificação superaram a `ReLU` padrão) e regularização (`Dropout` + `Weight Decay`).
* **Persistência e Fidelidade:** Os modelos recarregados com `weights_only=True` e `model.eval()` reproduziram exatamente as saídas dos modelos originais em memória (`torch.allclose == True`, diferença máxima absoluta < 1e-6).
* **Eficiência e Implantação:** Modelos ultracompactos (12,70 KB e 29,69 KB em disco), com alta vazão em lote (> 1,6 milhão de amostras/s em CPU) e emissão irrisória de CO₂, plenamente viáveis para ambientes restritos ou dispositivos de borda (*edge*).

#### Tabela-Resumo Final (Etapa 3 — `reports/tables/resumo_final.csv`)

| Tarefa | Configuração Final | Métrica | Treino | Teste | Gap (Treino − Teste) | K-Fold 5 (Média ± DP) | Parâmetros | Disco (KB) | Treino (s) | Vazão Lote (amostras/s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão** | `[32, 32]`, GELU, AdamW, `lr=0.005`, `bs=32`, `drop=0.2`, `wd=1e-5` | RMSE | 0,3299 | 0,5142 | −0,1843 | 0,4673 ± 0,0124 | 2.049 | 12,70 | 2,16 | 1.632.653,1 |
| **Classificação** | `[64, 64]`, Tanh, RMSprop, `lr=0.003`, `bs=16`, `drop=0.3`, `wd=0.0` | F1-macro | 0,8112 | 0,7761 | +0,0351 | 0,7579 ± 0,0258 | 6.340 | 29,69 | 1,54 | 2.994.510,1 |

Os detalhamentos completos encontram-se em `reports/tables/` (sensibilidade dos hiperparâmetros, métricas completas da Aula 08, K-Fold e profiling) e `reports/figures/`.

---

## 📬 Contato

* **Aluno:** José Salustiano Neto Júnior
* **E-mail:** `jose.salustiano.166@ufrn.edu.br`