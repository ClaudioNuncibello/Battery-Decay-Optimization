"""
run_mlp.py — Baseline Deep Learning (MLP)
-----------------------------------------------------
Esegue il training di una classica rete neurale (MLP) sul dataset.
Questo serve come baseline per dimostrare perché il modello fisico
a 3 parametri (equazione esponenziale) è superiore in questo contesto.

Uso:
    python run_mlp.py
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from src.data    import load_data
from src.model   import MLPModel
from src.trainer import train, TrainConfig
from src.utils   import plot_loss_curve, save_history

# ── Configurazione ────────────────────────────────────────────
CSV_PATH    = "battery_cycle_level_dataset_CLEAN_FINAL.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LEARNING_RATE = 1e-3

# ── Dati ─────────────────────────────────────────────────────
x, y, meta = load_data(CSV_PATH)

# ── Training ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f" MLP (Deep Learning Baseline) — Adam lr={LEARNING_RATE}")
print(f"{'='*60}")

model     = MLPModel(hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# L'MLP non ha parametri fisici, quindi usiamo la MSE pura
def mse_loss_wrapper(pred, target, mod):
    return F.mse_loss(pred, target)

config = TrainConfig(
    epochs         = 5000,
    batch_size     = -1,       # full-batch come per gli altri
    log_every      = 100,
    use_projection = False,    # nessuna proiezione necessaria
    patience       = 600,
)

label   = "MLP (Black-box)"
history = train(model, optimizer, x, y, mse_loss_wrapper, config,
                optimizer_name=label)
history.label = label

save_history(history, str(RESULTS_DIR / "mlp.pkl"))

plot_loss_curve(history,
                title="MLP Baseline — Convergenza",
                save_path=str(RESULTS_DIR / "mlp_loss.png"))

# Salva anche i pesi del modello per il confronto
torch.save(model.state_dict(), str(RESULTS_DIR / "mlp_weights.pth"))

print("\n[run_mlp] Completato. Risultati in:", RESULTS_DIR)
