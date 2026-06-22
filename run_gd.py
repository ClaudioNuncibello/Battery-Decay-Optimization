"""
run_gd.py — Step 2: Gradient Descent Classico (Baseline)
---------------------------------------------------------
Esegue il training con SGD puro (momentum=0) in modalità full-batch
per 3 valori di learning rate. Salva le history in results/.

Uso:
    python run_gd.py
"""

import torch
from pathlib import Path

from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import make_constrained_loss
from src.trainer import train, TrainConfig
from src.utils   import plot_loss_curve, plot_fitting_curve, save_history, compare_optimizers

# ── Configurazione ────────────────────────────────────────────
CSV_PATH     = "battery_cycle_level_dataset_CLEAN_FINAL.csv"
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LEARNING_RATES = [0.1, 0.01, 0.001]

# ── Dati ─────────────────────────────────────────────────────
x, y, meta = load_data(CSV_PATH)

# ── Training ─────────────────────────────────────────────────
histories = []

for lr in LEARNING_RATES:
    print(f"\n{'='*60}")
    print(f" Gradient Descent — lr = {lr}")
    print(f"{'='*60}")

    model     = BatteryDecayModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    loss_fn   = make_constrained_loss(penalty_weight=1e3)

    config = TrainConfig(
        epochs         = 10_000,
        batch_size     = -1,        # FULL-BATCH
        log_every      = 200,
        penalty_weight = 1e3,
        use_projection = True,
        min_param_val  = 1e-6,
        patience       = 800,
    )

    history       = train(model, optimizer, x, y, loss_fn, config,
                          optimizer_name=f"GD lr={lr}")
    history.label = f"GD lr={lr}"
    histories.append(history)
    save_history(history, str(RESULTS_DIR / f"gd_lr{lr}.pkl"))

    # Plot individuale
    plot_loss_curve(history, title=f"GD Classico — lr={lr}",
                    save_path=str(RESULTS_DIR / f"gd_loss_lr{lr}.png"))
    plot_fitting_curve(model, x, y, meta, title=f"Fitting GD — lr={lr}",
                       label=f"GD lr={lr}",
                       save_path=str(RESULTS_DIR / f"gd_fitting_lr{lr}.png"))

# ── Confronto fra i 3 learning rate ──────────────────────────
compare_optimizers(histories,
                   title="GD Classico — Confronto Learning Rate",
                   save_path=str(RESULTS_DIR / "gd_lr_comparison.png"))

print("\n[run_gd] Completato. Risultati in:", RESULTS_DIR)
