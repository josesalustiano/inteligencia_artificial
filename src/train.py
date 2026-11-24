import torch
import copy

def treinar(modelo, dl_tr, dl_val, loss_fn, otimizador, n_epocas=200, paciencia=20):
    hist = {"treino": [], "val": []}
    melhor_val = float("inf")
    espera = 0
    melhor_estado = None

    for ep in range(n_epocas):
        modelo.train()
        soma_treino = 0.0
        
        for xb, yb in dl_tr:
            otimizador.zero_grad()
            preds = modelo(xb)
            perda = loss_fn(preds, yb)
            perda.backward()
            otimizador.step()
            soma_treino += perda.item() * len(xb)
            
        hist["treino"].append(soma_treino / len(dl_tr.dataset))

        modelo.eval()
        with torch.no_grad():
            soma_val = sum(loss_fn(modelo(xb), yb).item() * len(xb) for xb, yb in dl_val)
            val_loss = soma_val / len(dl_val.dataset)
            hist["val"].append(val_loss)

        if val_loss < melhor_val - 1e-6:
            melhor_val = val_loss
            espera = 0
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1

        if espera >= paciencia:
            break

    modelo.load_state_dict(melhor_estado)
    return hist