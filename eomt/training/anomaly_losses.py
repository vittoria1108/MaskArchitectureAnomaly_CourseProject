# Questo file contiene le due metriche per migliorare l'anomaly segmentation di EoMT:
# 1) IsoMaxPlusHead -> sostituisce la testa di classificazione lineare con una basata su DISTANZE da prototipi (Estensione 1a del PDF)
#
# 2) logit_normalize -> funzione che "normalizza" i logit durante il training (Estensione 1b del PDF)

import torch
import torch.nn as nn
import torch.nn.functional as F

# PARTE 1 -- IsoMax+ : la testa di classificazione basata su DISTANZE

# Invece di moltiplicare le feature per una matrice di pesi (classico nn.Linear),
# confrontiamo ogni feature con un "prototipo" learnable di ciascuna classe e
# il logit diventa "minus distanza euclidea tra feature e prototipo".
# Un pixel anomalo (mai visto in training) avra' feature LONTANE da TUTTI i
# prototipi delle classi note. Quindi tutte le distanze saranno grandi e
# tutti i logit saranno molto negativi. Questo da' uno score di anomalia
# molto piu' netto rispetto al softmax classico.

class IsoMaxPlusHead(nn.Module):

    """
    Sostituisce un nn.Linear(in_features, num_classes_plus_one) e produce
    logit = - |distance_scale| * distanza_euclidea( feature_normalizzata,
                                                   prototipo_normalizzato )

    Input  : tensore (B, Q, D)
        B = batch size
        Q = numero di query di EoMT
        D = embed_dim (es. 768 per ViT-Base, 1024 per ViT-Large)

    Output : tensore (B, Q, C)
        C = num_classes_plus_one  (le 19 classi di Cityscapes + 1 "no-object")
    """

    def __init__(self, in_features: int, num_classes_plus_one: int):
        super().__init__()

        # Prototipi learnable per ogni classe: shape (C, D)
        self.prototypes = nn.Parameter(
            torch.empty(num_classes_plus_one, in_features)
        )

        # Fattore di scala per mappare correttamente le distanze [0, 2]
        self.distance_scale = nn.Parameter(torch.ones(1))

        # I prototipi partono da una gaussiana standard N(0, 1)
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x ha shape (B, Q, D). Le query di EoMT, una per ogni "slot" di
        # potenziale oggetto, ognuna con feature di dimensione D.

        # Questo è il trucco "isometric" di IsoMax+: distanza euclidea tra vettori
        # normalizzati è equivalente a coseno.
        # Normalizzazione L2 di feature e prototipi (proiezione su sfera unitaria)
        f = F.normalize(x, p=2.0, dim=-1)               # shape: (B, Q, D).
        p = F.normalize(self.prototypes, p=2.0, dim=-1)  # shape: (C, D)

        # Calcolo distanza euclidea tra feature e prototipi
        # Per riusare la versione batched, "appiattiamo" B*Q in una
        # dimensione sola, calcoliamo la cdist, poi riformiamo (B, Q, C).
        B, Q, D = f.shape
        f_flat = f.reshape(B * Q, D)                    # (B*Q, D)
        dist = torch.cdist(f_flat, p, p=2.0)            # (B*Q, C)
        dist = dist.reshape(B, Q, -1)                   # (B, Q, C)

        # Restituisce logit proporzionali alla distanza negativa.
        # Logit = - |scale| * distanza
        # - segno negativo: piu' la distanza e' grande, piu' il logit e'
        #     piccolo (basso), quindi quella classe e' "poco probabile".
        # - |scale| (torch.abs) per evitare scale negativi che invertirebbero
        #     la logica (un trucco standard del paper).
        return -torch.abs(self.distance_scale) * dist   # (B, Q, C)



#  PARTE 2 -- LogitNorm : normalizzazione della norma dei logit

#   Durante il training, la norma del vettore di logit ||f|| tende a crescere
#   indefinitamente, il modello diventa OVERCONFIDENT, anche su input sconosciuti.
#   Dividiamo i logit per la loro stessa norma (moltiplicata per una temperatura tau) PRIMA
#   di calcolare la cross-entropy.

def logit_normalize(logits: torch.Tensor, tau: float = 0.04) -> torch.Tensor:

    """Normalizza i logit alla loro norma L2, scalando per tau.

    Args:
        logits: tensore di forma (..., C). La normalizzazione viene fatta
                sull'ultima dimensione (quella delle classi). Funziona quindi
                sia per (B, C), sia per (B, Q, C) di EoMT.
        tau   : temperatura.
                Valori piu' piccoli = norme dei logit piu' forzate a essere
                piccole = piu' regolarizzazione contro l'overconfidence.

    Returns:
        Tensor della stessa shape, con norma L2 lungo l'ultima dimensione
        pari a 1 / tau.
    """

    norm = logits.norm(p=2, dim=-1, keepdim=True) + 1e-7 
    return logits / (norm * tau)