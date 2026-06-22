"""
run_lbfgsb.py — Step 3: L-BFGS-B (Il Campione Numerico)
---------------------------------------------------------
Esegue il training con L-BFGS-B via scipy. I vincoli fisici
(alpha, beta, gamma > 0) sono gestiti nativamente come box constraints.
Il gradiente è calcolato con PyTorch Autograd.

Uso:
    python run_lbfgsb.py
"""

import torch
from pathlib import Path

from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import mse_loss           # Loss PURA: niente penalizzazione
from src.trainer import train_lbfgsb, TrainConfig
from src.utils   import plot_loss_curve, plot_fitting_curve, save_history

# ── Configurazione ────────────────────────────────────────────
CSV_PATH    = "battery_cycle_level_dataset_CLEAN_FINAL.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Dati ─────────────────────────────────────────────────────
x, y, meta = load_data(CSV_PATH)

# ── Training ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(" L-BFGS-B — Quasi-Newton con Box Constraints nativi")
print(f"{'='*60}")

model = BatteryDecayModel()

config = TrainConfig(
    epochs        = 1000,       # Max iterazioni per scipy (solitamente < 100)
    batch_size    = -1,         # FULL-BATCH (L-BFGS-B è deterministico)
    log_every     = 1,          # Logga ogni iterazione (poche)
    use_projection = False,     # NON necessaria: vincoli gestiti nativamente
    min_param_val  = 1e-6,
)

# Box constraints: (lower_bound, upper_bound) per (alpha, beta, gamma)
bounds = [
    (1e-6, None),   # alpha > 0
    (1e-6, None),   # beta  > 0
    (1e-6, None),   # gamma > 0
]

history       = train_lbfgsb(model, x, y, mse_loss, config, bounds=bounds)
history.label = "L-BFGS-B"

save_history(history, str(RESULTS_DIR / "lbfgsb.pkl"))

# ── Plot ──────────────────────────────────────────────────────
plot_loss_curve(history,
                title="L-BFGS-B — Curva di Convergenza",
                log_scale=True,
                save_path=str(RESULTS_DIR / "lbfgsb_loss.png"))

plot_fitting_curve(model, x, y, meta,
                   label="L-BFGS-B",
                   title="Curve Fitting — L-BFGS-B",
                   save_path=str(RESULTS_DIR / "lbfgsb_fitting.png"))

print("\n[run_lbfgsb] Completato. Risultati in:", RESULTS_DIR)
