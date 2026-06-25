"""
run_lion.py — Step 5: Lion — EvoLved Sign Momentum
---------------------------------------------------
Esegue il training con Lion (libreria lion-pytorch).
Testa 2 learning rate: Lion richiede lr circa 10x inferiore ad ADAM.

Installazione: pip install lion-pytorch

Uso:
    python run_lion.py
"""

import torch
from pathlib import Path

try:
    from lion_pytorch import Lion
except ImportError:
    raise ImportError(
        "Libreria 'lion-pytorch' non trovata.\n"
        "Installala con: pip install lion-pytorch"
    )

from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import make_constrained_loss
from src.trainer import train, TrainConfig
from src.utils   import (plot_loss_curve, plot_fitting_curve,
                          save_history, compare_optimizers)

# ── Configurazione ────────────────────────────────────────────
CSV_PATH    = "battery_cycle_level_dataset_CLEAN_FINAL.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Lion richiede lr ~10x inferiore ad ADAM (aggiornamenti a magnitudine costante)
LEARNING_RATES = [1e-4, 3e-5]

# ── Dati ─────────────────────────────────────────────────────
x, y, meta = load_data(CSV_PATH)

# ── Training ─────────────────────────────────────────────────
histories = []

for lr in LEARNING_RATES:
    print(f"\n{'='*60}")
    print(f" Lion — lr = {lr}")
    print(f"{'='*60}")

    model     = BatteryDecayModel()
    optimizer = Lion(
        model.parameters(),
        lr           = lr,
        betas        = (0.9, 0.99),
        weight_decay = 0.0,     # Nessun weight decay (non ha senso fisico qui)
    )
    loss_fn = make_constrained_loss(penalty_weight=1e3)

    config = TrainConfig(
        epochs         = 5000,
        batch_size     = 32,         # Mini-batch (come ADAM per confronto equo)
        log_every      = 100,
        penalty_weight = 1e3,
        use_projection = True,       # Critico: Lion ignora la magnitudine
        min_param_val  = 1e-6,
        patience       = 600,
    )

    label         = f"Lion lr={lr}"
    history       = train(model, optimizer, x, y, loss_fn, config,
                          optimizer_name=label)
    history.label = label
    histories.append(history)
    save_history(history, str(RESULTS_DIR / f"lion_lr{lr}.pkl"))

    # Plot individuale
    plot_loss_curve(history,
                    title=f"Lion — lr={lr}",
                    save_path=str(RESULTS_DIR / f"lion_loss_lr{lr}.png"))

# Plot del fitting con il lr migliore (da determinare dai risultati)
best_history = min(histories, key=lambda h: h.losses[-1])
print(f"\n[Lion] Miglior lr: {best_history.label} (loss={best_history.losses[-1]:.6e})")

# Ri-addestramento per ottenere il modello finale
best_lr = LEARNING_RATES[histories.index(best_history)]
final_model = BatteryDecayModel()
final_opt   = Lion(final_model.parameters(), lr=best_lr, betas=(0.9, 0.99))
loss_fn     = make_constrained_loss()
config_full = TrainConfig(epochs=5000, batch_size=32, patience=600)
train(final_model, final_opt, x, y, loss_fn, config_full,
      optimizer_name=f"Lion lr={best_lr}", verbose=False)

plot_fitting_curve(final_model, x, y, meta,
                   label=f"Lion (lr={best_lr})",
                   title="Curve Fitting — Lion",
                   save_path=str(RESULTS_DIR / "lion_fitting.png"))

# ── Confronto learning rate ───────────────────────────────────
compare_optimizers(histories,
                   title="Lion — Confronto Learning Rate",
                   save_path=str(RESULTS_DIR / "lion_lr_comparison.png"))

print("\n[run_lion] Completato. Risultati in:", RESULTS_DIR)
