"""
src/utils.py
------------
Funzioni di visualizzazione e analisi per il notebook finale.

Tutte le funzioni accettano un parametro opzionale 'ax' per l'embedding
in subplot di matplotlib (perfetto per il notebook). Se ax=None, la funzione
crea e mostra una figura autonoma.

Palette colori degli ottimizzatori (consistente in tutti i plot):
    Gradient Descent : #ef4444  (rosso)
    L-BFGS-B         : #22c55e  (verde)
    ADAM             : #3b82f6  (blu)
    Lion             : #a855f7  (viola)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, List

import torch

# -- Import condizionale per evitare errori se usato senza i moduli src --
try:
    from .trainer import TrainHistory
    from .model   import BatteryDecayModel
except ImportError:
    from src.trainer import TrainHistory
    from src.model   import BatteryDecayModel


# ============================================================
# Palette e stile globale
# ============================================================

# Mappa nome ottimizzatore → colore (case-insensitive matching)
OPTIMIZER_COLORS = {
    "gradient descent": "#ef4444",
    "sgd":              "#ef4444",
    "gd":               "#ef4444",
    "l-bfgs-b":         "#22c55e",
    "lbfgsb":           "#22c55e",
    "adam":             "#3b82f6",
    "lion":             "#a855f7",
}

OPTIMIZER_MARKERS = {
    "sgd": "o", "gd": "o", "gradient descent": "o",
    "l-bfgs-b": "s", "lbfgsb": "s",
    "adam": "^",
    "lion": "D",
}

def _get_color(label: str) -> str:
    key = label.lower().split("(")[0].strip()
    for k, v in OPTIMIZER_COLORS.items():
        if k in key:
            return v
    # Fallback ciclico
    defaults = ["#f59e0b", "#06b6d4", "#ec4899", "#84cc16"]
    return defaults[hash(label) % len(defaults)]

def _get_marker(label: str) -> str:
    key = label.lower().split("(")[0].strip()
    for k, v in OPTIMIZER_MARKERS.items():
        if k in key:
            return v
    return "o"

def set_style():
    """Applica uno stile dark moderno a tutti i plot successivi."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "axes.facecolor":    "#1e1e2e",
        "figure.facecolor":  "#13131f",
        "axes.edgecolor":    "#44475a",
        "grid.color":        "#2d2d44",
        "grid.linewidth":    0.7,
        "axes.grid":         True,
        "legend.framealpha": 0.3,
        "lines.linewidth":   2.0,
    })

# Applica lo stile al momento dell'import
set_style()


# ============================================================
# 1. Loss Curve
# ============================================================

def plot_loss_curve(
    history: TrainHistory,
    ax:      Optional[plt.Axes] = None,
    title:   Optional[str]      = None,
    log_scale: bool             = True,
    color:   Optional[str]      = None,
    save_path: Optional[str]    = None,
) -> Figure:
    """
    Plotta la curva di convergenza della loss nel tempo.

    Parametri
    ----------
    history : TrainHistory
    ax : matplotlib Axes (opzionale, per embedding in subplot)
    title : str (opzionale)
    log_scale : bool — se True, asse y in scala logaritmica
    color : str — colore della linea (auto se None)
    save_path : str — salva la figura su file se specificato

    Ritorna
    -------
    Figure matplotlib
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    c = color or _get_color(history.label)
    epochs = history.epochs_logged
    losses = history.losses

    ax.plot(epochs, losses, color=c, linewidth=2, label=history.label, zorder=3)
    ax.fill_between(epochs, losses, alpha=0.08, color=c)

    if log_scale and min(losses) > 0:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch / Iterazione")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title or f"Curva di Convergenza — {history.label}")
    ax.legend()

    # Annotazione: loss finale
    ax.annotate(
        f"finale: {losses[-1]:.4e}",
        xy=(epochs[-1], losses[-1]),
        xytext=(-80, 15),
        textcoords="offset points",
        color=c,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
    )

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return fig


# ============================================================
# 2. Fitting Curve
# ============================================================

def plot_fitting_curve(
    model:      "BatteryDecayModel",
    x:          torch.Tensor,
    y:          torch.Tensor,
    meta:       Optional[dict]    = None,
    ax:         Optional[plt.Axes] = None,
    title:      Optional[str]      = None,
    label:      str                = "Modello fittato",
    color:      Optional[str]      = None,
    save_path:  Optional[str]      = None,
) -> Figure:
    """
    Sovrappone la curva fittata ai dati reali.

    Parametri
    ----------
    model : BatteryDecayModel — modello già ottimizzato
    x, y  : Tensor — dati reali
    meta  : dict (opzionale) — se fornito, colora i punti per batteria
    ax    : Axes (opzionale)
    title : str (opzionale)
    label : str — etichetta della curva fittata
    color : str — colore della curva (auto se None)
    save_path : str — salva su file se specificato

    Ritorna
    -------
    Figure matplotlib
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    c = color or _get_color(label)

    x_np = x.detach().numpy()
    y_np = y.detach().numpy()

    # -- Scatter dei dati reali --
    if meta and "df" in meta:
        # Colora per batteria (notebook view)
        df = meta["df"]
        batteries = df["battery_id"].unique()
        cmap = plt.get_cmap("tab20")
        for i, bid in enumerate(sorted(batteries)):
            mask = df["battery_id"] == bid
            xb = df.loc[mask, "x"].values
            yb = df.loc[mask, "capacity"].values
            ax.scatter(xb, yb, s=10, alpha=0.5, color=cmap(i % 20), label=bid, zorder=2)
    else:
        ax.scatter(x_np, y_np, s=8, alpha=0.4, color="#94a3b8",
                   label="Dati reali", zorder=2)

    # -- Curva fittata --
    x_line = torch.linspace(float(x.min()), float(x.max()), 300)
    with torch.no_grad():
        y_line = model(x_line).numpy()

    ax.plot(x_line.numpy(), y_line, color=c, linewidth=2.5, label=label, zorder=5)

    # -- Annotazione parametri --
    p = model.get_params()
    param_txt = (
        f"α = {p['alpha']:.4f}\n"
        f"β = {p['beta']:.4f}\n"
        f"γ = {p['gamma']:.4f}"
    )
    ax.text(
        0.97, 0.95, param_txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        color=c,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2e", alpha=0.8, edgecolor=c),
    )

    max_cycle = meta["max_cycle"] if meta else 1
    xlabel = "Ciclo normalizzato x" if (meta and meta.get("normalized")) else "Ciclo"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Capacità (Ah)")
    ax.set_title(title or f"Curve Fitting — {label}")

    if meta and len(meta.get("batteries", [])) <= 12:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.legend(fontsize=8)

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return fig


# ============================================================
# 3. Confronto ottimizzatori — Loss curves sovrapposte
# ============================================================

def compare_optimizers(
    histories:  List[TrainHistory],
    title:      str                = "Confronto Ottimizzatori",
    log_scale:  bool               = True,
    ax:         Optional[plt.Axes] = None,
    save_path:  Optional[str]      = None,
) -> Figure:
    """
    Sovrappone le loss curve di più ottimizzatori in un unico grafico.

    Parametri
    ----------
    histories : lista di TrainHistory
    title : str
    log_scale : bool
    ax : Axes (opzionale)
    save_path : str

    Ritorna
    -------
    Figure matplotlib
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.get_figure()

    for h in histories:
        c = _get_color(h.label)
        m = _get_marker(h.label)
        ax.plot(
            h.epochs_logged, h.losses,
            color=c, linewidth=2, label=h.label,
            marker=m, markevery=max(1, len(h.losses)//8),
            markersize=5, zorder=3,
        )

    if log_scale and all(min(h.losses) > 0 for h in histories):
        ax.set_yscale("log")

    ax.set_xlabel("Epoch / Iterazione")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend(fontsize=10)

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return fig


# ============================================================
# 4. Traiettoria dei parametri
# ============================================================

def plot_param_trajectory(
    histories:  List[TrainHistory],
    title:      str                = "Traiettoria dei Parametri",
    save_path:  Optional[str]      = None,
) -> Figure:
    """
    Plotta l'evoluzione di α, β, γ nel tempo per ogni ottimizzatore.
    Crea una figura con 3 subplot (uno per parametro).

    Ritorna
    -------
    Figure matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    param_names = ["alpha", "beta", "gamma"]
    param_labels = [r"$\alpha$", r"$\beta$", r"$\gamma$"]

    for h in histories:
        c = _get_color(h.label)
        for ax, pname, plabel in zip(axes, param_names, param_labels):
            values = [p[pname] for p in h.params]
            ax.plot(h.epochs_logged, values, color=c, linewidth=2, label=h.label)

    for ax, plabel in zip(axes, param_labels):
        ax.set_xlabel("Epoch / Iterazione")
        ax.set_ylabel(f"Valore {plabel}")
        ax.set_title(f"Evoluzione {plabel}")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Salvato in '{save_path}'")

    return fig


# ============================================================
# 5. Tabella riepilogativa per il notebook
# ============================================================

def print_summary_table(histories: List[TrainHistory]):
    """
    Stampa una tabella riepilogativa dei risultati di tutti gli ottimizzatori.
    Formattata per leggibilità nel notebook.
    """
    print("\n" + "=" * 80)
    print(f"{'Ottimizzatore':<20} {'Loss Finale':>14} {'alpha':>10} {'beta':>10} {'gamma':>10} {'Tempo':>10} {'F-Evals':>8}")
    print("=" * 80)
    for h in histories:
        if not h.losses:
            continue
        p = h.params[-1]
        fevals = str(h.n_func_evals) if h.n_func_evals else "N/A"
        print(
            f"{h.label:<20} "
            f"{h.losses[-1]:>14.6e} "
            f"{p['alpha']:>10.5f} "
            f"{p['beta']:>10.5f} "
            f"{p['gamma']:>10.5f} "
            f"{h.elapsed_s:>9.2f}s "
            f"{fevals:>8}"
        )
    print("=" * 80 + "\n")


# ============================================================
# 6. Dashboard completa (per il notebook)
# ============================================================

def plot_dashboard(
    histories:  List[TrainHistory],
    model_dict: dict,           # {'label': BatteryDecayModel}
    x:          torch.Tensor,
    y:          torch.Tensor,
    meta:       Optional[dict] = None,
    save_path:  Optional[str]  = None,
) -> Figure:
    """
    Dashboard completa 2x2 per il notebook finale:
    - [0,0] Loss curve confronto
    - [0,1] Traiettoria alpha
    - [1,0] Traiettoria beta & gamma
    - [1,1] Curve di fitting sovrapposte

    Parametri
    ----------
    histories   : lista di TrainHistory
    model_dict  : dizionario {label: BatteryDecayModel} con i modelli ottimizzati
    x, y        : dati
    meta        : metadati del dataset
    save_path   : path per il salvataggio

    Ritorna
    -------
    Figure matplotlib
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Analisi Comparativa degli Ottimizzatori — NASA Battery Dataset",
                 fontsize=15, y=1.01)

    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax_loss  = fig.add_subplot(gs[0, 0])
    ax_alpha = fig.add_subplot(gs[0, 1])
    ax_traj  = fig.add_subplot(gs[1, 0])
    ax_fit   = fig.add_subplot(gs[1, 1])

    # -- Loss curve --
    compare_optimizers(histories, ax=ax_loss, title="Loss di Convergenza")

    # -- Traiettoria alpha --
    for h in histories:
        c = _get_color(h.label)
        alphas = [p["alpha"] for p in h.params]
        ax_alpha.plot(h.epochs_logged, alphas, color=c, linewidth=2, label=h.label)
    ax_alpha.set_xlabel("Epoch")
    ax_alpha.set_ylabel(r"$\alpha$")
    ax_alpha.set_title(r"Evoluzione del parametro $\alpha$")
    ax_alpha.legend(fontsize=8)

    # -- Traiettoria beta e gamma --
    for h in histories:
        c = _get_color(h.label)
        betas  = [p["beta"]  for p in h.params]
        gammas = [p["gamma"] for p in h.params]
        ax_traj.plot(h.epochs_logged, betas,  color=c, linewidth=2,
                     label=f"β {h.label}", linestyle="-")
        ax_traj.plot(h.epochs_logged, gammas, color=c, linewidth=1.5,
                     label=f"γ {h.label}", linestyle="--")
    ax_traj.set_xlabel("Epoch")
    ax_traj.set_ylabel("Valore parametro")
    ax_traj.set_title(r"Evoluzione di $\beta$ (—) e $\gamma$ (--)")
    ax_traj.legend(fontsize=7, ncol=2)

    # -- Fitting curves --
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()
    ax_fit.scatter(x_np, y_np, s=6, alpha=0.3, color="#94a3b8", label="Dati reali", zorder=1)

    x_line = torch.linspace(float(x.min()), float(x.max()), 300)
    for label, model in model_dict.items():
        c = _get_color(label)
        with torch.no_grad():
            y_line = model(x_line).numpy()
        ax_fit.plot(x_line.numpy(), y_line, color=c, linewidth=2, label=label, zorder=3)

    ax_fit.set_xlabel("Ciclo normalizzato")
    ax_fit.set_ylabel("Capacità (Ah)")
    ax_fit.set_title("Curve di Fitting Comparate")
    ax_fit.legend(fontsize=8)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Dashboard salvata in '{save_path}'")

    return fig


# ============================================================
# 7. I/O History
# ============================================================

def save_history(history: TrainHistory, path: str):
    """Wrapper per history.save() — comodo da importare direttamente."""
    history.save(path)


def load_history(path: str) -> TrainHistory:
    """Wrapper per TrainHistory.load() — comodo da importare direttamente."""
    return TrainHistory.load(path)
