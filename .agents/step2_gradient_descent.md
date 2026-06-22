# Step 2 — Gradient Descent Classico (Baseline)

## Obiettivo
Implementare la **discesa del gradiente full-batch con passo rigido** come baseline storica. Lo scopo non è ottenere i risultati migliori, ma *dimostrare i limiti* di un metodo del primo ordine senza adattività e senza informazioni sulla curvatura.

> [!NOTE]
> Questo step riutilizza integralmente `src/trainer.py` — l'unica differenza rispetto agli altri step è l'ottimizzatore passato al trainer.

---

## File di Esecuzione: `run_gd.py`

```python
import torch
from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import constrained_mse_loss
from src.trainer import train, TrainConfig
from src.utils   import plot_loss_curve, plot_fitting_curve, save_history

# --- CONFIGURAZIONE OTTIMIZZATORE ---
LEARNING_RATES = [0.1, 0.01, 0.001]

x, y = load_data('battery_cycle_level_dataset_CLEAN_FINAL.csv')

for lr in LEARNING_RATES:
    model     = BatteryDecayModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # GD puro (no momentum)

    config = TrainConfig(
        epochs        = 10_000,
        batch_size    = -1,         # FULL-BATCH
        log_every     = 500,
        use_projection = True,
        min_param_val  = 1e-6,
    )

    history = train(model, optimizer, x, y, constrained_mse_loss, config)
    save_history(history, f'results/gd_lr{lr}.pkl')
    plot_loss_curve(history, title=f'GD Classico — lr={lr}')
    plot_fitting_curve(model, x, y, title=f'Fitting GD — lr={lr}')
```

---

## Dettagli Matematici

### Update Rule
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

### Gestione Vincoli — Proiezione Post-Step
Dopo ogni aggiornamento, i parametri vengono forzati nel dominio ammissibile:
$$\theta_{t+1} \leftarrow \max(\theta_{t+1},\ \epsilon) \quad \text{con } \epsilon = 10^{-6}$$

Implementazione nel trainer (attivata da `use_projection=True`):
```python
with torch.no_grad():
    for p in model.parameters():
        p.clamp_(min=config.min_param_val)
```

### Criterio di Stop
- **Max iterazioni:** 10.000 epoch
- **Early stopping:** $|\mathcal{L}_t - \mathcal{L}_{t-1}| < 10^{-8}$

---

## Hyperparametri da Esplorare

| Parametro | Valori testati | Motivazione |
|---|---|---|
| Learning rate $\eta$ | 0.1, 0.01, 0.001 | Analisi sensitività: lr troppo alto → divergenza, troppo basso → lentezza |
| Inizializzazione | Default (α=1.0, β=0.5, γ=0.5) | Punto di partenza comune a tutti gli step |
| Max epochs | 10.000 | Necessario per la convergenza (molto più lento degli altri metodi) |

---

## Metriche da Raccogliere

1. **Curva di Loss vs. iterazione** per ogni learning rate
2. **Traiettoria dei parametri** (α, β, γ) nel tempo
3. **Numero di iterazioni a convergenza** (o plateau)
4. **Loss finale** e **parametri trovati** (α*, β*, γ*)
5. **Tempo di esecuzione** wall-clock
6. **Analisi del rimbalzo sui vincoli:** quante volte la proiezione è stata attiva?

---

## Risultati Attesi e Interpretazione

| Scenario | Comportamento previsto |
|---|---|
| lr = 0.1 | Possibile divergenza o oscillazioni forti intorno al minimo |
| lr = 0.01 | Convergenza lenta ma stabile, possibile intrappolamento in plateau |
| lr = 0.001 | Convergenza molto lenta, richiederà tutte le 10.000 iterazioni |

**Messaggio chiave per l'esame:** il GD classico non ha informazioni sulla curvatura della loss surface → si muove sempre con la stessa "forza" indipendentemente dalla geometria locale. Questo giustifica l'introduzione dei metodi avanzati (Step 3–5).

---

## Output da Salvare

```
results/
├── gd_lr0.1.pkl      # TrainHistory serializzato
├── gd_lr0.01.pkl
├── gd_lr0.001.pkl
└── plots/
    ├── gd_loss_curves.png
    └── gd_fitting.png
```

---

## Dipendenze

- `torch.optim.SGD` (momentum=0 → GD puro)
- `src/trainer.py` → `train()` con `batch_size=-1`
- `src/utils.py` → `plot_loss_curve()`, `plot_fitting_curve()`
