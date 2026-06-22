# Step 3 — L-BFGS-B (Il Campione Numerico)

## Obiettivo
Implementare il metodo **Quasi-Newton a memoria limitata con Box Constraints nativi**. Questo è l'algoritmo teoricamente più adatto al problema: sfrutta la curvatura approssimata della funzione di loss e gestisce i vincoli fisici senza alcun artificio (niente penalizzazioni, niente proiezioni).

> [!IMPORTANT]
> L-BFGS-B richiede un'interfaccia speciale: usa `scipy.optimize.minimize` come backend, ma il gradiente viene calcolato con **PyTorch Autograd**. Il trainer generico gestisce questo caso con un branch dedicato quando `optimizer=None`.

---

## File di Esecuzione: `run_lbfgsb.py`

```python
import torch
import numpy as np
from scipy.optimize import minimize
from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import mse_loss          # Loss PURA — niente penalizzazione
from src.trainer import train_lbfgsb, TrainConfig
from src.utils   import plot_loss_curve, plot_fitting_curve, save_history

x, y = load_data('battery_cycle_level_dataset_CLEAN_FINAL.csv')
model = BatteryDecayModel()

config = TrainConfig(
    epochs     = 1000,        # Max iterazioni (solitamente converge in < 100)
    batch_size = -1,          # FULL-BATCH (L-BFGS-B è deterministico)
    log_every  = 10,
    use_projection = False,   # Non necessaria: vincoli gestiti nativamente
)

# Box constraints: (lower_bound, upper_bound) per (alpha, beta, gamma)
bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]

history = train_lbfgsb(model, x, y, mse_loss, config, bounds)
save_history(history, 'results/lbfgsb.pkl')
plot_loss_curve(history, title='L-BFGS-B — Quasi-Newton')
plot_fitting_curve(model, x, y, title='Fitting L-BFGS-B')
```

---

## Dettagli Matematici

### L'Approssimazione dell'Hessiana (L-BFGS)
A differenza del GD classico (che usa solo il gradiente $g_t = \nabla \mathcal{L}$), L-BFGS approssima la matrice Hessiana inversa $H_t^{-1}$ usando gli ultimi $m$ coppie gradiente/spostamento:

$$\theta_{t+1} = \theta_t - H_t^{-1} \cdot g_t$$

La variante "Limited Memory" mantiene solo $m$ vettori in memoria (anzichè la matrice $n \times n$ completa).

### I Box Constraints (la "-B" di L-BFGS-B)
Il vincolo fisico $\alpha, \beta, \gamma > 0$ viene espresso come **box constraint** passato direttamente a SciPy:
```python
bounds = [
    (1e-6, None),   # alpha > 0
    (1e-6, None),   # beta  > 0
    (1e-6, None),   # gamma > 0
]
```
SciPy gestisce i vincoli tramite il metodo delle **proiezioni attive** interno all'algoritmo — nessun termine extra nella loss.

---

## Interfaccia PyTorch ↔ SciPy

L-BFGS-B richiede una funzione `f(params_numpy) -> (loss_scalar, gradient_numpy)`. Il bridge è costruito nel `trainer.py`:

```python
def _make_scipy_objective(model, x, y, loss_fn):
    def objective(params_np):
        # 1. Carica i parametri numpy nel modello PyTorch
        _set_params(model, torch.tensor(params_np, dtype=torch.float32))
        
        # 2. Forward pass e loss
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # 3. Backward pass per il gradiente (Autograd)
        loss.backward()
        
        # 4. Estrai il gradiente come numpy
        grad_np = _get_grad(model).numpy().astype(np.float64)
        model.zero_grad()
        
        return loss.item(), grad_np
    return objective
```

---

## Hyperparametri

| Parametro | Valore | Nota |
|---|---|---|
| Memoria $m$ | 10 | Numero di coppie (s,y) storiche mantenute |
| `ftol` | 1e-12 | Tolleranza sulla variazione relativa della loss |
| `gtol` | 1e-8 | Tolleranza sulla norma del gradiente proiettato |
| Max iterazioni | 1000 | Tipicamente converge in < 100 |
| Inizializzazione | Identica agli altri step | Confronto equo |

---

## Metriche da Raccogliere

1. **Numero di valutazioni della funzione** (function evaluations) — confronto con GD classico (ordini di grandezza inferiore)
2. **Numero di iterazioni** a convergenza
3. **Loss finale** e **parametri (α*, β*, γ*)** con alta precisione
4. **Norma del gradiente** finale (misura della qualità della soluzione)
5. **Tempo di esecuzione** wall-clock
6. **Verifica vincoli:** α, β, γ ≥ 1e-6 (sempre soddisfatti senza proiezione)

---

## Risultati Attesi e Interpretazione

| Metrica | L-BFGS-B | GD Classico (confronto) |
|---|---|---|
| Iterazioni | ~20–100 | ~5.000–10.000 |
| Function evaluations | ~50–200 | ~10.000 |
| Loss finale | Minima assoluta (o locale) | Dipende dall'lr |
| Precisione parametri | Alta (gradiente ~0) | Media |
| Gestione vincoli | Nativa, zero overhead | Proiezione post-step |

**Messaggio chiave per l'esame:** L-BFGS-B è il metodo di riferimento per problemi di ottimizzazione non lineare a pochi parametri. La sua conoscenza della curvatura lo rende **ordini di grandezza più efficiente** del GD classico. La variante "-B" dimostra come i vincoli fisici possano essere integrati nell'algoritmo anziché essere un'aggiunta esterna.

---

## Output da Salvare

```
results/
├── lbfgsb.pkl
└── plots/
    ├── lbfgsb_loss_curve.png
    └── lbfgsb_fitting.png
```

---

## Dipendenze

- `scipy.optimize.minimize` con `method='L-BFGS-B'`
- `torch.autograd` per il calcolo del gradiente
- `src/loss.py` → `mse_loss()` (senza penalizzazione)
- `src/trainer.py` → `train_lbfgsb()` (branch speciale del trainer)
