"""
src/loss.py
-----------
Funzioni di loss per il curve fitting con vincoli fisici.

Due modalità:
    1. mse_loss()              — MSE pura, per L-BFGS-B (vincoli nativi)
    2. make_constrained_loss() — MSE + penalizzazione barriera, per GD/ADAM/Lion

Tutte le loss hanno la stessa firma:
    loss_fn(pred, target, model) -> Tensor scalare
per essere intercambiabili nel trainer generico.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Loss 1 — MSE Pura (per L-BFGS-B)
# ---------------------------------------------------------------------------
def mse_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    model:  Optional[nn.Module] = None,  # ignorato, presente per uniformità
) -> torch.Tensor:
    """
    Mean Squared Error puro.

    Usato con L-BFGS-B che gestisce i vincoli box nativamente.
    Il parametro 'model' è accettato ma ignorato per uniformità di firma.
    """
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Loss 2 — MSE + Penalizzazione Barriera (per GD, ADAM, Lion)
# ---------------------------------------------------------------------------
def make_constrained_loss(penalty_weight: float = 1e3):
    """
    Factory che ritorna una funzione di loss con penalizzazione barriera.

    La penalizzazione si attiva solo quando i parametri scendono sotto zero,
    esplodendo quadraticamente per forzarli nel dominio ammissibile.

    Formula:
        L_total = MSE(pred, target)
                + lambda * [ReLU(-alpha)^2 + ReLU(-beta)^2 + ReLU(-gamma)^2]

    Parametri
    ----------
    penalty_weight : float
        Peso lambda della penalizzazione (default: 1000).
        Valori più alti = vincoli più rigidi ma possibile condizionamento peggiore.

    Ritorna
    -------
    Callable con firma: loss_fn(pred, target, model) -> Tensor scalare

    Esempio
    -------
    >>> loss_fn = make_constrained_loss(penalty_weight=1e3)
    >>> loss = loss_fn(pred, y, model)
    """
    def constrained_loss(
        pred:   torch.Tensor,
        target: torch.Tensor,
        model:  nn.Module,
    ) -> torch.Tensor:
        mse = F.mse_loss(pred, target)

        # Penalizzazione ReLU^2: zero quando param > 0, quadratica altrimenti
        penalty = (
            torch.relu(-model.alpha) ** 2
            + torch.relu(-model.beta)  ** 2
            + torch.relu(-model.gamma) ** 2
        )

        return mse + penalty_weight * penalty

    constrained_loss.__name__ = f"constrained_mse_loss(λ={penalty_weight})"
    return constrained_loss


# ---------------------------------------------------------------------------
# Istanza di default pronta all'uso (importabile direttamente)
# ---------------------------------------------------------------------------
constrained_mse_loss = make_constrained_loss(penalty_weight=1e3)
