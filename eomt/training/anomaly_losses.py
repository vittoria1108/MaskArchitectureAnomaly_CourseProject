# ============================================================================
# anomaly_losses.py
# ----------------------------------------------------------------------------
# Questo file contiene DUE strumenti per migliorare l'anomaly segmentation
# di EoMT senza usare dati di outlier:
#
#   1) IsoMaxPlusHead       -> sostituisce la testa di classificazione lineare
#                              con una basata su DISTANZE da prototipi.
#                              (Estensione 1a del PDF)
#
#   2) logit_normalize(...) -> piccola funzione che "normalizza" i logit
#                              durante il training, per evitare l'overconfidence.
#                              (Estensione 1b del PDF)
#
# Sono COMPLETAMENTE INDIPENDENTI: puoi usarne una sola, entrambe insieme,
# o nessuna. Quando nessuna delle due e' attiva, il modello si comporta
# esattamente come l'originale EoMT.
#
# Riferimenti:
#   IsoMax+   : Macedo & Ludermir, arXiv:2105.14399
#   LogitNorm : Wei et al., ICML 2022, arXiv:2205.09310
# ============================================================================


# ---- Importiamo le librerie che ci servono --------------------------------
# torch              -> la libreria principale di PyTorch (tensori, autograd).
# torch.nn (nn)      -> contiene i "mattoncini" delle reti neurali
#                       (es. nn.Module, nn.Parameter, nn.Linear...).
# torch.nn.functional (F) -> funzioni "stateless" come F.normalize, F.cross_entropy.
#                            "Stateless" = non hanno pesi propri, sono pure funzioni.
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  PARTE 1 -- IsoMax+ : la testa di classificazione basata su DISTANZE
# ============================================================================
#
# IDEA IN UNA FRASE:
#   Invece di moltiplicare le feature per una matrice di pesi (classico nn.Linear),
#   confrontiamo ogni feature con un "prototipo" learnable di ciascuna classe e
#   il logit diventa "minus distanza euclidea tra feature e prototipo".
#
# PERCHE' SERVE PER L'ANOMALY DETECTION?
#   Un pixel anomalo (mai visto in training) avra' feature LONTANE da TUTTI i
#   prototipi delle classi note. Quindi tutte le distanze saranno grandi e
#   tutti i logit saranno molto negativi. Questo da' uno score di anomalia
#   molto piu' netto rispetto al softmax classico (che spesso sbara' alta
#   confidenza anche su input mai visti).
#
# COME SI INSERISCE IN EoMT?
#   In `eomt/models/eomt.py` la "testa di classe" originale e':
#       self.class_head = nn.Linear(embed_dim, num_classes + 1)
#   La sostituiamo con IsoMaxPlusHead(embed_dim, num_classes + 1).
#   Input e output hanno la STESSA shape, quindi il resto del modello
#   non si accorge di nulla.
# ----------------------------------------------------------------------------

class IsoMaxPlusHead(nn.Module):
    """Testa di classificazione "distance-based" (IsoMax+ first part).

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
        # nn.Module richiede sempre questa chiamata al super().__init__()
        # come prima riga: registra i meccanismi interni di PyTorch.
        super().__init__()

        # ---- I PROTOTIPI ----
        # Un prototipo per ogni classe. Sono PARAMETRI LEARNABLE
        # (nn.Parameter -> il gradient li aggiorna come fossero pesi normali).
        # Shape: (C, D).  Ogni riga e' il "centroide" di una classe nello
        # spazio delle feature.
        self.prototypes = nn.Parameter(
            torch.empty(num_classes_plus_one, in_features)
        )

        # ---- LO SCALE DELLA DISTANZA ----
        # Uno scalare learnable. Lo useremo come |distance_scale| (valore
        # assoluto) per assicurare che lo scaling sia sempre positivo.
        # Inizializzato a 1.0. Serve perche' la distanza tra vettori
        # normalizzati ha range [0, 2], e con CE pura il segnale di gradiente
        # sarebbe troppo debole -> questo parametro impara "quanto urlare".
        self.distance_scale = nn.Parameter(torch.ones(1))

        # ---- INIZIALIZZAZIONE DEI PROTOTIPI ----
        # I prototipi partono da una gaussiana standard N(0, 1). Verranno
        # poi imparati dal training. Quella usata nel paper ufficiale.
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x ha shape (B, Q, D). Le query di EoMT, una per ogni "slot" di
        # potenziale oggetto, ognuna con feature di dimensione D.

        # PASSO 1 -- normalizziamo le feature a norma 1 (L2).
        #   Cosi' ogni feature giace sulla sfera unitaria. Questo e' il
        #   trucco "isometric" di IsoMax+: distanza euclidea tra vettori
        #   normalizzati e' equivalente a coseno, e diventa una vera metrica.
        # F.normalize(x, p=2.0, dim=-1) divide ogni vettore per la sua norma.
        f = F.normalize(x, p=2.0, dim=-1)               # shape: (B, Q, D)

        # PASSO 2 -- normalizziamo anche i prototipi sulla stessa sfera.
        p = F.normalize(self.prototypes, p=2.0, dim=-1)  # shape: (C, D)

        # PASSO 3 -- calcoliamo la distanza euclidea tra OGNI feature e
        # OGNI prototipo. torch.cdist vuole input 2D (oppure batch 3D).
        # Per riusare la versione batched, "appiattiamo" B*Q in una
        # dimensione sola, calcoliamo la cdist, poi riformiamo (B, Q, C).
        B, Q, D = f.shape
        f_flat = f.reshape(B * Q, D)                    # (B*Q, D)
        # cdist(f_flat, p) -> matrice (B*Q, C) con le distanze euclidee.
        dist = torch.cdist(f_flat, p, p=2.0)            # (B*Q, C)
        dist = dist.reshape(B, Q, -1)                   # (B, Q, C)

        # PASSO 4 -- ritorniamo i LOGIT.
        # Logit = - |scale| * distanza
        #   - segno negativo: piu' la distanza e' grande, piu' il logit e'
        #     piccolo (basso), quindi quella classe e' "poco probabile".
        #   - |scale| (torch.abs) per evitare scale negativi che invertirebbero
        #     la logica (un trucco standard del paper).
        # Nota: NON applichiamo qui il fattore "entropic_scale". Quello
        # verra' applicato nella LOSS (vedi mask_classification_loss.py).
        return -torch.abs(self.distance_scale) * dist   # (B, Q, C)


# ============================================================================
#  PARTE 2 -- LogitNorm : normalizzazione della norma dei logit
# ============================================================================
#
# IDEA IN UNA FRASE:
#   Durante il training, la norma del vettore di logit ||f|| tende a crescere
#   indefinitamente. Risultato: il modello diventa OVERCONFIDENT, cioe' urla
#   "sono certo!" anche su input sconosciuti. Soluzione: dividiamo i logit
#   per la loro stessa norma (moltiplicata per una temperatura tau) PRIMA
#   di calcolare la cross-entropy.
#
# FORMULA (vedi paper, eq. 4):
#
#                            exp( f_y / (tau * ||f||) )
#   L_LogitNorm = -log  ---------------------------------------
#                       sum_c  exp( f_c / (tau * ||f||) )
#
# COSA CAMBIA NELLA LOSS?
#   Solo l'ARGOMENTO della softmax: invece di passare i logit "f", passiamo
#   i logit normalizzati "f / (tau * ||f||)". L'inferenza usa i logit
#   ORIGINALI (non normalizzati), ma adesso hanno norme controllate, quindi
#   MSP / MaxLogit / MaxEntropy diventano molto piu' discriminativi tra
#   pixel inlier e pixel anomali.
#
# DOVE LO CHIAMIAMO?
#   In `mask_classification_loss.py`, dentro l'override di `loss_labels`,
#   PRIMA di passare i logit alla cross-entropy del padre Mask2FormerLoss.
# ----------------------------------------------------------------------------

def logit_normalize(logits: torch.Tensor, tau: float = 0.04) -> torch.Tensor:
    """Normalizza i logit alla loro norma L2, scalando per tau.

    Args:
        logits: tensore di forma (..., C). La normalizzazione viene fatta
                sull'ULTIMA dimensione (quella delle classi). Funziona quindi
                sia per (B, C), sia per (B, Q, C) di EoMT.
        tau   : temperatura. Il paper usa valori piccoli (default 0.04).
                Valori piu' piccoli = norme dei logit piu' forzate a essere
                piccole = piu' regolarizzazione contro l'overconfidence.

    Returns:
        Tensor della stessa shape, con norma L2 lungo l'ultima dimensione
        pari a 1 / tau.
    """
    # PASSO 1 -- calcoliamo ||f|| (norma euclidea) lungo l'asse delle classi.
    #   keepdim=True mantiene la dimensione cosi' la divisione fa broadcast
    #   correttamente. Aggiungiamo 1e-7 per evitare divisioni per zero
    #   (fondamentale a inizio training, quando i logit possono essere
    #   tutti vicini allo zero).
    norm = logits.norm(p=2, dim=-1, keepdim=True) + 1e-7   # shape (..., 1)

    # PASSO 2 -- dividiamo i logit per (tau * norm).
    #   Il risultato ha norma L2 lungo l'ultima dim pari a 1/tau.
    #   Adesso e' pronto per essere passato a F.cross_entropy.
    return logits / (norm * tau)


# ============================================================================
#  PARTE 3 (OPZIONALE) -- isomax_plus_ce
# ============================================================================
#
# Helper che racchiude la cross-entropy "in stile IsoMax+":
#   loss = CE( entropic_scale * logits_IsoMaxPlus,  targets )
#
# Nota: per integrare IsoMax+ in MaskClassificationLoss, abbiamo scelto di
# applicare lo scaling direttamente nell'override di `loss_labels` (vedi
# la guida nel messaggio precedente), riusando la CE del padre
# Mask2FormerLoss che gia' gestisce empty_weight, ignore_index e la
# convenzione di shape. Quindi questo helper qui sotto NON viene usato
# direttamente dalla pipeline EoMT: lo lasciamo come riferimento "stand-alone"
# (ad es. per debug, unit test, o se in futuro vuoi staccarlo dalla CE
# del padre).
#
# PUOI IGNORARE QUESTA FUNZIONE se non ti serve.
# ----------------------------------------------------------------------------

def isomax_plus_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    empty_weight: torch.Tensor = None,
    entropic_scale: float = 10.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy "IsoMax+ second part", versione stand-alone.

    Args:
        logits        : output di IsoMaxPlusHead, forma (B, Q, C). Sono gia'
                        = -|s|*distanza, quindi NEGATIVI.
        targets       : indici di classe ground-truth, forma (B, Q) con valori
                        in [0, C-1] oppure ignore_index per posizioni da
                        ignorare.
        empty_weight  : pesi per classe (per dare meno peso al "no-object").
                        Se None, tutte le classi hanno peso 1.
        entropic_scale: fattore moltiplicativo dentro la softmax. Il paper
                        suggerisce 10.0. Piu' alto = softmax piu' "piccata".
        ignore_index  : valore di target da saltare nel calcolo della loss.
    """
    # Moltiplichiamo i logit per entropic_scale (equivalente a softmax con
    # scaling). F.cross_entropy vuole shape (B, C, ...) -> facciamo transpose.
    scaled = logits * entropic_scale                    # (B, Q, C)
    return F.cross_entropy(
        scaled.transpose(1, 2),                          # (B, C, Q)
        targets,                                         # (B, Q)
        weight=empty_weight,
        ignore_index=ignore_index,
    )
