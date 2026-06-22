"""
compare_all.py — Confronto Finale dei 4 Ottimizzatori
------------------------------------------------------
Carica le history salvate dai 4 run script e produce
il confronto completo per la presentazione all'esame.

Eseguire DOPO aver lanciato tutti i run script:
    python run_gd.py
    python run_lbfgsb.py
    python run_adam.py
    python run_lion.py
    python compare_all.py

Uso:
    python compare_all.py
"""

import torch
from pathlib import Path

from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import mse_loss, make_constrained_loss
from src.trainer import train, train_lbfgsb, TrainConfig
from src.utils   import (
    load_history, compare_optimizers,
    plot_param_trajectory, plot_dashboard,
    print_summary_table, save_history,
)

RESULTS_DIR = Path("results")
CSV_PATH    = "battery_cycle_level_dataset_CLEAN_FINAL.csv"

# ── Carica dati ───────────────────────────────────────────────
x, y, meta = load_data(CSV_PATH)

# ── Carica history da disco ───────────────────────────────────
# Seleziona il "best" per ogni ottimizzatore
try:
    h_gd      = load_history(str(RESULTS_DIR / "gd" / "gd_lr0.01.pkl"))
except FileNotFoundError:
    print("[compare] gd_lr0.01.pkl non trovato, esegui run_gd.py prima.")
    h_gd = None

try:
    h_lbfgsb  = load_history(str(RESULTS_DIR / "lbfgsb" / "lbfgsb.pkl"))
except FileNotFoundError:
    print("[compare] lbfgsb.pkl non trovato, esegui run_lbfgsb.py prima.")
    h_lbfgsb = None

try:
    h_adam    = load_history(str(RESULTS_DIR / "adam" / "adam_bs=32.pkl"))
except FileNotFoundError:
    print("[compare] adam_bs=32.pkl non trovato, esegui run_adam.py prima.")
    h_adam = None

try:
    h_lion    = load_history(str(RESULTS_DIR / "lion" / "lion_lr0.0001.pkl"))
except FileNotFoundError:
    print("[compare] lion_lr0.0001.pkl non trovato, esegui run_lion.py prima.")
    h_lion = None

histories = [h for h in [h_gd, h_lbfgsb, h_adam, h_lion] if h is not None]

if not histories:
    raise RuntimeError("Nessuna history trovata. Esegui prima i run script.")

# ── Ri-addestra i modelli per il dashboard (serve il modello finale) ──
def retrain_model(history_label, optimizer_fn, loss_fn, config):
    """Helper per riaddestrare un modello dai parametri finali della history."""
    # (In un flusso completo si salverebbe anche il modello, non solo la history)
    model = BatteryDecayModel()
    if history_label in ["GD lr=0.01"]:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
    elif history_label == "L-BFGS-B":
        bounds = [(1e-6, None)] * 3
        cfg    = TrainConfig(epochs=1000, use_projection=False)
        train_lbfgsb(model, x, y, mse_loss, cfg, bounds=bounds, verbose=False)
        return model
    elif "Adam" in history_label:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        return model
    train(model, opt, x, y, loss_fn, config, verbose=False)
    return model

loss_fn     = make_constrained_loss()
config_fast = TrainConfig(epochs=5000, batch_size=-1, patience=600)

model_dict = {}
for h in histories:
    m = BatteryDecayModel()
    # Inizializza dai parametri finali della history
    import torch as _t
    final = h.params[-1]
    m.set_param_vector(_t.tensor([final["alpha"], final["beta"], final["gamma"]]))
    model_dict[h.label] = m

# ── Stampa tabella riepilogativa ──────────────────────────────
print_summary_table(histories)

# ── Confronto loss curves ─────────────────────────────────────
compare_optimizers(
    histories,
    title="Confronto Finale — 4 Ottimizzatori",
    save_path=str(RESULTS_DIR / "final_loss_comparison.png"),
)

# ── Traiettoria parametri ─────────────────────────────────────
plot_param_trajectory(
    histories,
    title="Traiettoria dei Parametri — 4 Ottimizzatori",
    save_path=str(RESULTS_DIR / "final_param_trajectory.png"),
)

# ── Dashboard completa ────────────────────────────────────────
plot_dashboard(
    histories,
    model_dict,
    x, y, meta,
    save_path=str(RESULTS_DIR / "dashboard_finale.png"),
)

print("\n[compare_all] Completato. Grafici in:", RESULTS_DIR)
