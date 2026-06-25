"""
src/model.py
------------
Modello parametrico per il decadimento della capacità della batteria.

Equazione fisica:
    Capacity(x) = alpha * exp(-beta * x) + gamma

dove x è il numero di cicli (normalizzato in [0,1]).

I parametri alpha, beta, gamma sono nn.Parameter → PyTorch Autograd
calcola i gradienti automaticamente per tutti gli ottimizzatori.

Vincoli fisici (gestiti esternamente da loss.py / trainer.py):
    alpha > 0  — capacità degradabile iniziale
    beta  > 0  — tasso di decadimento chimico
    gamma > 0  — capacità residua asintotica
"""

import torch
import torch.nn as nn


class BatteryDecayModel(nn.Module):
    """
    Modello parametrico custom per il curve fitting del decadimento batteria.

    Parametri addestrabili (nn.Parameter):
        alpha : capacità iniziale degradabile  (vincolo: > 0)
        beta  : tasso di decadimento           (vincolo: > 0)
        gamma : asintoto residuo               (vincolo: > 0)

    Esempio di utilizzo
    -------------------
    >>> model = BatteryDecayModel()
    >>> x = torch.linspace(0, 1, 100)
    >>> y_pred = model(x)

    Inizializzazione
    ----------------
    I valori di default sono scelti in base alle statistiche del dataset
    NASA Battery (capacità media ~1.5 Ah, decadimento ~30% in vita utile):
        alpha_init = 0.5   (porzione degradabile)
        beta_init  = 2.0   (decadimento con x normalizzato in [0,1])
        gamma_init = 1.0   (asintoto residuo)
    """

    def __init__(
        self,
        alpha_init: float = 0.5,
        beta_init:  float = 2.0,
        gamma_init: float = 1.0,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta  = nn.Parameter(torch.tensor(float(beta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcola la capacità predetta per ogni valore di x.

        Parametri
        ----------
        x : Tensor shape (N,) — cicli normalizzati in [0, 1]

        Ritorna
        -------
        Tensor shape (N,) — capacità predetta in Ah
        """
        return self.alpha * torch.exp(-self.beta * x) + self.gamma

    # ------------------------------------------------------------------
    # Utility per trainer e notebook
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        """Ritorna un dizionario con i valori correnti dei parametri."""
        return {
            "alpha": self.alpha.item(),
            "beta":  self.beta.item(),
            "gamma": self.gamma.item(),
        }

    def get_param_vector(self) -> torch.Tensor:
        """Ritorna i parametri come tensore 1D [alpha, beta, gamma]."""
        return torch.tensor(
            [self.alpha.item(), self.beta.item(), self.gamma.item()],
            dtype=torch.float32,
        )

    def set_param_vector(self, params: torch.Tensor):
        """
        Imposta i parametri da un tensore 1D [alpha, beta, gamma].
        Usato dal bridge PyTorch ↔ SciPy in train_lbfgsb().
        """
        with torch.no_grad():
            self.alpha.copy_(params[0])
            self.beta.copy_(params[1])
            self.gamma.copy_(params[2])

    def reset(
        self,
        alpha_init: float = 0.5,
        beta_init:  float = 2.0,
        gamma_init: float = 1.0,
    ):
        """Reimposta i parametri ai valori iniziali (utile per i run multipli)."""
        with torch.no_grad():
            self.alpha.fill_(alpha_init)
            self.beta.fill_(beta_init)
            self.gamma.fill_(gamma_init)

    def __repr__(self) -> str:
        p = self.get_params()
        return (
            f"BatteryDecayModel("
            f"alpha={p['alpha']:.5f}, "
            f"beta={p['beta']:.5f}, "
            f"gamma={p['gamma']:.5f})"
        )


class MLPModel(nn.Module):
    """
    Rete Neurale Multi-Layer Perceptron per il decadimento batteria.
    Funge da baseline black-box per dimostrare l'utilità del modello fisico.
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametri
        ----------
        x : Tensor shape (N,) — cicli normalizzati
        
        Ritorna
        -------
        Tensor shape (N,)
        """
        # Trasforma da (N,) a (N, 1) per il layer lineare
        x_in = x.unsqueeze(1)
        y_pred = self.net(x_in)
        # Ritorna a (N,) per uniformità con la loss
        return y_pred.squeeze(1)

    def get_params(self) -> dict:
        """Ritorna dummy dict per uniformità col trainer."""
        return {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
