"""
src/trainer.py
--------------
Loop di training GENERICO e riutilizzabile.

Contiene due funzioni principali:
    train()         — per qualsiasi ottimizzatore PyTorch (GD, ADAM, Lion)
    train_lbfgsb()  — per L-BFGS-B via scipy (branch speciale)

Il pattern di utilizzo è identico per tutti gli step:

    model     = BatteryDecayModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config    = TrainConfig(epochs=5000, batch_size=32)
    loss_fn   = make_constrained_loss(config.penalty_weight)
    history   = train(model, optimizer, x, y, loss_fn, config)
"""

import time
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


# ============================================================
# DataClasses: Config e History
# ============================================================

@dataclass
class TrainConfig:
    """
    Parametri di configurazione del training.

    Attributi
    ---------
    epochs : int
        Numero massimo di epoch (o iterazioni per L-BFGS-B).
    batch_size : int
        Dimensione del mini-batch. -1 = full-batch.
    log_every : int
        Frequenza di logging (ogni N epoch).
    penalty_weight : float
        Peso della penalizzazione barriera per i vincoli (usato da make_constrained_loss).
    use_projection : bool
        Se True, applica clamp post-step per forzare i parametri > min_param_val.
    min_param_val : float
        Valore minimo per la proiezione (floor dei parametri).
    early_stop_tol : float
        Tolleranza per l'early stopping: stop se |Δloss| < tol per 'patience' epoch.
    patience : int
        Numero di epoch senza miglioramento prima dell'early stop.
    """
    epochs:          int   = 5000
    batch_size:      int   = -1       # -1 = full-batch
    log_every:       int   = 100
    penalty_weight:  float = 1e3
    use_projection:  bool  = True
    min_param_val:   float = 1e-6
    early_stop_tol:  float = 1e-10
    patience:        int   = 500


@dataclass
class TrainHistory:
    """
    Risultati di una sessione di training. Serializzabile con pickle.

    Attributi
    ---------
    losses : list[float]
        Valori della loss ad ogni log step.
    params : list[dict]
        Dizionari {'alpha', 'beta', 'gamma'} ad ogni log step.
    epochs_logged : list[int]
        Indici delle epoch in cui è stato fatto il log.
    elapsed_s : float
        Tempo totale di esecuzione in secondi.
    optimizer_name : str
        Nome dell'ottimizzatore (per plot e confronti).
    label : str
        Etichetta per i grafici (può essere più descrittiva di optimizer_name).
    n_func_evals : int
        Numero di valutazioni della funzione obiettivo (rilevante per L-BFGS-B).
    converged : bool
        True se è avvenuta convergenza prima del massimo di epoch.
    """
    losses:        list  = field(default_factory=list)
    params:        list  = field(default_factory=list)
    epochs_logged: list  = field(default_factory=list)
    elapsed_s:     float = 0.0
    optimizer_name: str  = "unknown"
    label:         str   = ""
    n_func_evals:  int   = 0
    converged:     bool  = False

    def save(self, path: str):
        """Salva la history su disco (pickle)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[history] Salvata in '{path}'")

    @staticmethod
    def load(path: str) -> "TrainHistory":
        """Carica una history salvata."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> str:
        """Ritorna una stringa di riepilogo per il notebook."""
        final_loss = self.losses[-1] if self.losses else float("nan")
        final_p = self.params[-1] if self.params else {}
        return (
            f"[{self.label}]\n"
            f"  Loss finale  : {final_loss:.6e}\n"
            f"  Parametri    : alpha={final_p.get('alpha', '?'):.5f}, "
            f"beta={final_p.get('beta', '?'):.5f}, "
            f"gamma={final_p.get('gamma', '?'):.5f}\n"
            f"  Tempo        : {self.elapsed_s:.2f}s\n"
            f"  Log steps    : {len(self.losses)}\n"
            f"  Func evals   : {self.n_func_evals if self.n_func_evals else 'N/A'}"
        )


# ============================================================
# Utility interne
# ============================================================

def _project_params(model: nn.Module, min_val: float = 1e-6):
    """Proietta tutti i parametri sopra min_val (box constraint enforcement)."""
    with torch.no_grad():
        for p in model.parameters():
            p.clamp_(min=min_val)


# ============================================================
# train() — loop generico per ottimizzatori PyTorch
# ============================================================

def train(
    model:          nn.Module,
    optimizer:      torch.optim.Optimizer,
    x:              torch.Tensor,
    y:              torch.Tensor,
    loss_fn:        Callable,
    config:         TrainConfig,
    optimizer_name: str = "",
    verbose:        bool = True,
) -> TrainHistory:
    """
    Loop di training generico per qualsiasi ottimizzatore PyTorch.

    Supporta full-batch (config.batch_size == -1) e mini-batch.
    La proiezione post-step (clamp) viene applicata se config.use_projection=True.

    Parametri
    ----------
    model : nn.Module
        Il modello da ottimizzare (BatteryDecayModel o qualsiasi nn.Module).
    optimizer : Optimizer
        Ottimizzatore PyTorch già configurato (SGD, Adam, Lion, ...).
    x : Tensor shape (N,)
        Input (cicli normalizzati).
    y : Tensor shape (N,)
        Target (capacità).
    loss_fn : Callable
        Funzione di loss con firma: loss_fn(pred, target, model) -> Tensor.
    config : TrainConfig
        Configurazione del training.
    optimizer_name : str
        Nome da usare nella history (default: tipo dell'ottimizzatore).
    verbose : bool
        Se True, stampa progressi durante il training.

    Ritorna
    -------
    TrainHistory con loss_curve, param_trajectory, elapsed_time.
    """
    name = optimizer_name or type(optimizer).__name__
    history = TrainHistory(optimizer_name=name, label=name)

    n = len(x)
    batch_size = n if config.batch_size == -1 else min(config.batch_size, n)

    best_loss   = float("inf")
    no_improve  = 0

    t_start = time.perf_counter()

    for epoch in range(config.epochs):
        # -- Shuffle per mini-batch --
        if batch_size < n:
            perm = torch.randperm(n)
            xs, ys = x[perm], y[perm]
        else:
            xs, ys = x, y

        # -- Loop sui batch --
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            xb = xs[start : start + batch_size]
            yb = ys[start : start + batch_size]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb, model)
            loss.backward()
            optimizer.step()

            # Proiezione post-step
            if config.use_projection:
                _project_params(model, config.min_param_val)

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_loss /= n_batches

        # -- Logging --
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            history.losses.append(epoch_loss)
            history.params.append(model.get_params())
            history.epochs_logged.append(epoch)

            if verbose and epoch % (config.log_every * 5) == 0:
                p = model.get_params()
                print(
                    f"  [{name}] epoch={epoch:>6d}  "
                    f"loss={epoch_loss:.6e}  "
                    f"alpha={p['alpha']:.4f} beta={p['beta']:.4f} gamma={p['gamma']:.4f}"
                )

        # -- Early stopping --
        if epoch_loss < best_loss - config.early_stop_tol:
            best_loss  = epoch_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.patience and epoch > config.log_every:
            # Logga l'ultimo punto e fermati
            if history.epochs_logged[-1] != epoch:
                history.losses.append(epoch_loss)
                history.params.append(model.get_params())
                history.epochs_logged.append(epoch)
            history.converged = True
            if verbose:
                print(f"  [{name}] Early stop @ epoch {epoch}  loss={epoch_loss:.6e}")
            break

    history.elapsed_s = time.perf_counter() - t_start

    if verbose:
        print(history.summary())

    return history


# ============================================================
# train_lbfgsb() — branch speciale per scipy L-BFGS-B
# ============================================================

def train_lbfgsb(
    model:    nn.Module,
    x:        torch.Tensor,
    y:        torch.Tensor,
    loss_fn:  Callable,
    config:   TrainConfig,
    bounds:   Optional[list] = None,
    verbose:  bool = True,
) -> TrainHistory:
    """
    Training con L-BFGS-B via scipy.optimize.minimize.

    Il gradiente è calcolato con PyTorch Autograd e passato a SciPy
    come array numpy. I vincoli sono gestiti nativamente come box constraints.

    Parametri
    ----------
    model : nn.Module
        Modello da ottimizzare.
    x, y : Tensor
        Dati di training (full-batch, L-BFGS-B è deterministico).
    loss_fn : Callable
        Loss function con firma: loss_fn(pred, target, model) -> Tensor.
    config : TrainConfig
        config.epochs = max iterazioni per scipy.
    bounds : list of (float, float)
        Box constraints: [(lb_alpha, ub_alpha), (lb_beta, ub_beta), ...].
        Default: [(min_param_val, None)] * 3
    verbose : bool
        Stampa l'output di scipy.

    Ritorna
    -------
    TrainHistory con la traiettoria di convergenza L-BFGS-B.
    """
    if bounds is None:
        bounds = [(config.min_param_val, None)] * len(list(model.parameters()))

    history = TrainHistory(optimizer_name="L-BFGS-B", label="L-BFGS-B")
    func_evals = [0]

    # ---- Funzione obiettivo per scipy (ritorna loss + gradiente) ----
    def objective(params_np: np.ndarray):
        func_evals[0] += 1

        # Carica i parametri nel modello
        params_t = torch.tensor(params_np, dtype=torch.float32)
        model.set_param_vector(params_t)

        # Azzera i gradienti manualmente
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Forward + backward
        pred = model(x)
        loss = loss_fn(pred, y, model)
        loss.backward()

        loss_val = loss.item()
        grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
        return loss_val, grad.detach().numpy().astype(np.float64)

    # ---- Callback per raccogliere la history ad ogni iterazione ----
    def callback(params_np: np.ndarray):
        params_t = torch.tensor(params_np, dtype=torch.float32)
        model.set_param_vector(params_t)
        with torch.no_grad():
            pred = model(x)
            loss_val = loss_fn(pred, y, model).item()
        history.losses.append(loss_val)
        history.params.append(model.get_params())
        history.epochs_logged.append(len(history.losses))

    # ---- Punto di partenza ----
    x0 = model.get_param_vector().detach().numpy().astype(np.float64)

    t_start = time.perf_counter()

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,           # objective ritorna (loss, grad)
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": config.epochs,
            "ftol":    1e-15,
            "gtol":    1e-10,
            "iprint":  -1,  # silenzia l'output interno di scipy
        },
    )

    history.elapsed_s   = time.perf_counter() - t_start
    history.n_func_evals = func_evals[0]
    history.converged    = result.success

    # Aggiorna il modello con i parametri finali trovati da scipy
    final_params = torch.tensor(result.x, dtype=torch.float32)
    model.set_param_vector(final_params)

    if verbose:
        print(f"\n[L-BFGS-B] {result.message}")
        print(f"  Iterazioni       : {result.nit}")
        print(f"  Valutazioni f(x) : {result.nfev}")
        print(history.summary())

    return history
