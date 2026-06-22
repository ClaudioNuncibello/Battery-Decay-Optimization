"""
run_adam.py — Step 4: ADAM (Lo Standard Industriale)
-----------------------------------------------------
Esegue il training con ADAM testando 3 diverse dimensioni di mini-batch
per analizzare il trade-off varianza/precisione.

Uso:
    python run_adam.py
"""

import torch
from pathlib import Path

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

BATCH_SIZES  = [8, 32, -1]          # -1 = full-batch
LEARNING_RATE = 1e-3

# ── Dati ─────────────────────────────────────────────────────
x, y, meta = load_data(CSV_PATH)

# ── Training ─────────────────────────────────────────────────
histories = []

for bs in BATCH_SIZES:
    bs_label = f"bs={bs}" if bs > 0 else "bs=full"
    print(f"\n{'='*60}")
    print(f" ADAM — lr={LEARNING_RATE}, {bs_label}")
    print(f"{'='*60}")

    model     = BatteryDecayModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr    = LEARNING_RATE,
        betas = (0.9, 0.999),
        eps   = 1e-8,
    )
    loss_fn = make_constrained_loss(penalty_weight=1e3)

    config = TrainConfig(
        epochs         = 5000,
        batch_size     = bs,
        log_every      = 100,
        penalty_weight = 1e3,
        use_projection = True,
        min_param_val  = 1e-6,
        patience       = 600,
    )

    label         = f"Adam {bs_label}"
    history       = train(model, optimizer, x, y, loss_fn, config,
                          optimizer_name=label)
    history.label = label
    histories.append(history)
    save_history(history, str(RESULTS_DIR / f"adam_{bs_label}.pkl"))

    # Plot individuale
    plot_loss_curve(history,
                    title=f"ADAM — lr={LEARNING_RATE}, {bs_label}",
                    save_path=str(RESULTS_DIR / f"adam_loss_{bs_label}.png"))

# Plot dell'ultimo modello fittato (bs=full come più preciso)
last_model = BatteryDecayModel()
optimizer  = torch.optim.Adam(last_model.parameters(), lr=LEARNING_RATE)
loss_fn    = make_constrained_loss()
config_full = TrainConfig(epochs=5000, batch_size=-1, patience=600)
train(last_model, optimizer, x, y, loss_fn, config_full,
      optimizer_name="Adam full-batch", verbose=False)
plot_fitting_curve(last_model, x, y, meta,
                   label="Adam (full-batch)",
                   title="Curve Fitting — ADAM",
                   save_path=str(RESULTS_DIR / "adam_fitting.png"))

# ── Confronto batch size ──────────────────────────────────────
compare_optimizers(histories,
                   title="ADAM — Confronto Batch Size",
                   save_path=str(RESULTS_DIR / "adam_bs_comparison.png"))

print("\n[run_adam] Completato. Risultati in:", RESULTS_DIR)
