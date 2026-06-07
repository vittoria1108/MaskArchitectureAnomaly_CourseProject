# logit_normalize -> funzione che "normalizza" i logit durante il training (Estensione 1b del PDF)

import torch
import torch.nn as nn
import torch.nn.functional as F


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