import argparse
import torch
import pandas as pd

try:
    from src.models import MLP
except ImportError:
    from models import MLP

def carregar_modelo(caminho_modelo):
    ckpt = torch.load(caminho_modelo, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    modelo = MLP(
        in_dim=cfg["in_dim"],
        out_dim=cfg["out_dim"],
        hidden=tuple(cfg["hidden"]),
        activation=cfg["activation"],
        dropout=cfg["dropout"]
    )
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo, ckpt

def prever_dataframe(caminho_modelo, df):
    modelo, ckpt = carregar_modelo(caminho_modelo)
    X_novo = df[ckpt["features"]].values
    
    x_tensor = torch.tensor(X_novo, dtype=torch.float32)
    x_padronizado = (x_tensor - ckpt["x_media"]) / ckpt["x_desvio"]

    with torch.inference_mode():
        saidas = modelo(x_padronizado)
        
        if ckpt["task"] == "regression":
            # Volta para a escala original do target
            preds = (saidas.squeeze(1) * ckpt["y_desvio"]) + ckpt["y_media"]
            return preds.numpy(), None
        else:
            # Classificação multiclasse: softmax para probabilidades
            probs = torch.softmax(saidas, dim=1)
            preds = torch.argmax(probs, dim=1)
            return preds.numpy(), probs.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência com modelo MLP salvo em PyTorch.")
    parser.add_argument("--modelo", required=True, help="Caminho para o arquivo .pt")
    parser.add_argument("--entrada", required=True, help="Caminho para o CSV com novas amostras")
    args = parser.parse_args()

    df_entrada = pd.read_csv(args.entrada)
    preds, probs = prever_dataframe(args.modelo, df_entrada)

    print(f"Inferência concluída para {len(preds)} amostras.")
    print("Primeiras 10 previsões:", preds[:10])