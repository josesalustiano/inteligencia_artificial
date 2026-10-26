import pandas as pd

def aplicar_regras(row):
    uso = row.get('Daily_Usage_Hours', 0)
    ansiedade = row.get('Anxiety_Score', 0)
    depressao = row.get('Depression_Score', 0)
    fomo = row.get('FOMO_Score', 0)
    sono = row.get('Sleep_Quality_Score', 10)
    bem_estar = row.get('Mental_Wellbeing_Score', 0)

    if uso > 6 and (ansiedade > 6 or depressao > 6):
        return 3
    elif uso > 4 and fomo > 5 and sono < 5:
        return 2
    elif uso < 3 and bem_estar > 7:
        return 0
    else:
        return 1

df_test = pd.read_csv('data/processed/test_classification.csv')
df_test['Addiction_Predicted'] = df_test.apply(aplicar_regras, axis=1)

acertos = (df_test['Addiction_Level'] == df_test['Addiction_Predicted']).sum()
total = len(df_test)
print(f"Acertos do Sistema Especialista (Sessão 3): {acertos} de {total} ({(acertos/total)*100:.2f}%)")