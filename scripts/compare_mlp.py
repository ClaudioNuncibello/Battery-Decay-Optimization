"""
compare_mlp.py — Confronto Finale: Modello Fisico vs Deep Learning (MLP)
-------------------------------------------------------------------------
Genera i grafici per dimostrare il comportamento dell'MLP rispetto al 
miglior modello fisico (L-BFGS-B).
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.data    import load_data
from src.model   import BatteryDecayModel, MLPModel
from src.utils   import load_history, _get_color, set_style

RESULTS_DIR = Path("results")
CSV_PATH    = "battery_cycle_level_dataset_CLEAN_FINAL.csv"

# ── Carica dati ───────────────────────────────────────────────
x, y, meta = load_data(CSV_PATH)

# ── Carica history da disco ───────────────────────────────────
try:
    h_lbfgsb = load_history(str(RESULTS_DIR / "lbfgsb" / "lbfgsb.pkl"))
except FileNotFoundError:
    h_lbfgsb = None
    print("[compare_mlp] lbfgsb.pkl non trovato.")

try:
    h_mlp = load_history(str(RESULTS_DIR / "mlp.pkl"))
except FileNotFoundError:
    h_mlp = None
    print("[compare_mlp] mlp.pkl non trovato.")

if not h_lbfgsb or not h_mlp:
    raise RuntimeError("Manca una delle history per il confronto.")

# ── Ricostruisci Modelli ──────────────────────────────────────
# 1. Modello Fisico
model_phys = BatteryDecayModel()
p_final = h_lbfgsb.params[-1]
model_phys.set_param_vector(torch.tensor([p_final["alpha"], p_final["beta"], p_final["gamma"]]))

# 2. MLP
model_mlp = MLPModel(hidden_dim=32)
model_mlp.load_state_dict(torch.load(str(RESULTS_DIR / "mlp_weights.pth")))
model_mlp.eval()

# ── Generazione Grafici ───────────────────────────────────────
set_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Curve Fitting ---
x_np = x.detach().numpy()
y_np = y.detach().numpy()

# Dati reali
ax1.scatter(x_np, y_np, s=8, alpha=0.3, color="#94a3b8", label="Dati reali", zorder=1)

x_line = torch.linspace(float(x.min()), float(x.max()), 300)
with torch.no_grad():
    y_phys = model_phys(x_line).numpy()
    y_mlp  = model_mlp(x_line).numpy()

ax1.plot(x_line.numpy(), y_phys, color=_get_color("L-BFGS-B"), linewidth=2.5, 
         label="Modello Fisico (L-BFGS-B)", zorder=3)
ax1.plot(x_line.numpy(), y_mlp, color=_get_color("MLP"), linewidth=2.5, linestyle="--", 
         label="Deep Learning (MLP)", zorder=4)

ax1.set_xlabel("Ciclo normalizzato")
ax1.set_ylabel("Capacità (Ah)")
ax1.set_title("Confronto Fitting: Fisica vs Deep Learning")
ax1.legend()

# --- Plot 2: Metriche a Barre ---
labels = ["Modello Fisico\n(3 parametri)", "MLP Black-box\n(~1000 parametri)"]
losses = [h_lbfgsb.losses[-1], h_mlp.losses[-1]]
times  = [h_lbfgsb.elapsed_s, h_mlp.elapsed_s]
colors = [_get_color("L-BFGS-B"), _get_color("MLP")]

# Grafico a doppia asse (Loss e Tempo)
ax2_loss = ax2
ax2_time = ax2.twinx()

x_pos = [0, 1]
width = 0.35

bar1 = ax2_loss.bar([p - width/2 for p in x_pos], losses, width, color=colors, alpha=0.8, label='MSE Loss')
bar2 = ax2_time.bar([p + width/2 for p in x_pos], times, width, color=colors, alpha=0.4, hatch='//', label='Tempo (s)')

ax2_loss.set_ylabel('Final MSE Loss', color='white')
ax2_time.set_ylabel('Tempo (secondi)', color='gray')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.set_title("Trade-off: Errore vs Efficienza")

# Valori sulle barre
for i, b in enumerate(bar1):
    ax2_loss.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001, 
                  f"{losses[i]:.4f}", ha='center', va='bottom', color='white', fontsize=10)

for i, b in enumerate(bar2):
    ax2_time.text(b.get_x() + b.get_width()/2, b.get_height() + (max(times)*0.02), 
                  f"{times[i]:.2f}s", ha='center', va='bottom', color='gray', fontsize=10)

fig.tight_layout()

save_path = RESULTS_DIR / "mlp_vs_physical_comparison.png"
fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
print(f"[compare_mlp] Grafico comparativo salvato in '{save_path}'")
