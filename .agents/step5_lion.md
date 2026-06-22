# Step 5 — Lion — EvoLved Sign Momentum (L'Avanguardia)

## Obiettivo
Analizzare il comportamento di **Lion**, un algoritmo scoperto tramite ricerca automatizzata (Google Brain, 2023), applicato a un problema di regressione parametrica a soli 3 variabili. Lion usa esclusivamente il **segno** del gradiente per determinare la direzione dell'aggiornamento, ignorando completamente la sua magnitudine.

> [!NOTE]
> Lion è progettato per grandi modelli linguistici. Questo step ha anche uno scopo critico: valutare se e quanto questo algoritmo "overkill" si adatta a un problema con soli 3 parametri.

---

## File di Esecuzione: `run_lion.py`

```python
import torch
from lion_pytorch import Lion           # Libreria: pip install lion-pytorch
from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import constrained_mse_loss
from src.trainer import train, TrainConfig
from src.utils   import plot_loss_curve, plot_fitting_curve, save_history, compare_optimizers

x, y = load_data('battery_cycle_level_dataset_CLEAN_FINAL.csv')

# Lion richiede un LR molto più piccolo di ADAM (regola empirica: lr_lion ≈ lr_adam / 10)
LEARNING_RATES = [1e-4, 3e-5]

histories = []
for lr in LEARNING_RATES:
    model     = BatteryDecayModel()
    optimizer = Lion(
        model.parameters(),
        lr      = lr,
        betas   = (0.9, 0.99),
        weight_decay = 0.0,     # No weight decay per regressione fisica
    )

    config = TrainConfig(
        epochs         = 5000,
        batch_size     = 32,          # Mini-batch (come ADAM per confronto equo)
        log_every      = 100,
        penalty_weight = 1e3,
        use_projection = True,        # Critico per Lion (vedi sezione vincoli)
        min_param_val  = 1e-6,
    )

    history = train(model, optimizer, x, y, constrained_mse_loss, config)
    history.label = f'Lion lr={lr}'
    histories.append(history)
    save_history(history, f'results/lion_lr{lr}.pkl')

compare_optimizers(histories, title='Lion — Confronto Learning Rate')
```

---

## Dettagli Matematici

### Update Rule di Lion
Lion usa **un solo momento** (a differenza dei due di ADAM):

$$c_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$\theta_t = \theta_{t-1} - \eta \cdot \text{sign}(c_t)$$

$$m_t = \beta_2 \cdot m_{t-1} + (1 - \beta_2) \cdot g_t$$

**Cosa significa `sign()`:** ogni parametro si aggiorna esattamente di $\pm \eta$, indipendentemente da quanto sia grande o piccolo il gradiente. Questo rende Lion molto più aggressivo in direzione e molto più economico in memoria.

### Confronto con ADAM
| Aspetto | ADAM | Lion |
|---|---|---|
| Momenti mantenuti | 2 ($m_t$, $v_t$) | 1 ($m_t$) |
| Direzione aggiornamento | $\hat{m}_t / \sqrt{\hat{v}_t}$ | $\text{sign}(c_t)$ |
| Magnitudine aggiornamento | Adattiva per parametro | Costante ($\pm\eta$) |
| Memoria (parametri extra) | $2 \times |\theta|$ | $1 \times |\theta|$ |
| LR tipico | 1e-3 | 1e-4 (circa 1/10 di ADAM) |

---

## Gestione Vincoli — Caso Critico

### Perché i Vincoli sono Problematici con Lion
Lion applica aggiornamenti di magnitudine costante ($\pm\eta$) indipendentemente da quanto il parametro si avvicini al vincolo. Quando un parametro raggiunge il bordo del dominio ammissibile ($\alpha \approx 0$), Lion potrebbe continuare a "spingere" con la stessa forza.

**Doppia Strategia:**
1. **Penalizzazione nella loss** — il gradiente della penalizzazione aumenta vicino al vincolo, cercando di invertire la direzione
2. **Proiezione post-step** (critica con Lion) — safety net assoluta dopo ogni aggiornamento

```python
# Dopo ogni optimizer.step():
with torch.no_grad():
    for p in model.parameters():
        p.clamp_(min=1e-6)
```

> [!WARNING]
> Con Lion, la proiezione potrebbe essere attivata più frequentemente rispetto ad ADAM, proprio perché Lion ignora la magnitudine del gradiente vicino al vincolo. Questo è un comportamento interessante da documentare nell'analisi.

---

## Hyperparametri

| Parametro | Valore | Nota |
|---|---|---|
| Learning rate $\eta$ | 1e-4, 3e-5 | Molto più basso di ADAM (regola: lr_lion ≈ lr_adam/10) |
| $\beta_1$ | 0.9 | Momentum per il calcolo della direzione |
| $\beta_2$ | 0.99 | Momentum per l'aggiornamento dell'EMA |
| `weight_decay` | 0.0 | Nessuna regolarizzazione L2 (non ha senso fisico qui) |
| Batch size | 32 | Identico ad ADAM per confronto equo |
| Epochs | 5000 | Identico ad ADAM |

---

## Metriche da Raccogliere

1. **Stabilità della Loss:** confronto della varianza della loss curve con ADAM (stesse epoch, stesso batch size)
2. **Analisi dei rimbalzi sui vincoli:**
   - Frequenza di attivazione della proiezione
   - Comportamento di α, β, γ vicino al bordo 0
3. **Consumo di memoria** effettivo (Lion usa 1 momento vs. 2 di ADAM)
4. **Velocità di convergenza** (wall-clock time per epoch)
5. **Loss finale** vs. ADAM vs. L-BFGS-B vs. GD
6. **Sensibilità al Learning Rate:** analisi del comportamento con lr diversi

---

## Risultati Attesi e Analisi Critica

### Scenari Possibili

| Scenario | Comportamento |
|---|---|
| LR troppo alto (≥1e-3) | Oscillazioni forti, non converge — passo costante è troppo aggressivo |
| LR ottimale (~1e-4) | Convergenza stabile, loss comparabile ad ADAM |
| LR troppo basso (≤1e-6) | Convergenza lenta simile al GD classico |

### Analisi Critica per l'Esame
Lion è stato progettato per ottimizzare miliardi di parametri dove:
- La **memoria** è la risorsa critica (1 momento invece di 2 fa differenza su GPT-4)
- Il **rumore** del mini-batch è dominante (il segno è già un'astrazione del gradiente)

Su un problema con **3 parametri**:
- Il risparmio di memoria è trascurabile
- L'aggiornamento a magnitudine costante può essere meno preciso di ADAM vicino alla soluzione
- Il comportamento sui vincoli è meno controllabile

**Messaggio chiave per l'esame:** Lion mostra come un algoritmo vincitore in un contesto (LLM su miliardi di parametri) non sia necessariamente superiore in un contesto diverso (regressione parametrica a 3 variabili). La scelta dell'ottimizzatore deve essere guidata dalle caratteristiche del problema.

---

## Confronto Finale — Tutti e 4 gli Ottimizzatori

Il runner finale `compare_all.py` carica tutti i `TrainHistory` salvati e produce il confronto completo per la presentazione all'esame:

```python
# compare_all.py
from src.utils import load_history, compare_optimizers, plot_param_trajectories

histories = [
    load_history('results/gd_lr0.01.pkl'),
    load_history('results/lbfgsb.pkl'),
    load_history('results/adam_bs32.pkl'),
    load_history('results/lion_lr1e-4.pkl'),
]
compare_optimizers(histories, title='Confronto Finale — 4 Ottimizzatori')
plot_param_trajectories(histories)
```

---

## Output da Salvare

```
results/
├── lion_lr1e-4.pkl
├── lion_lr3e-5.pkl
└── plots/
    ├── lion_loss_comparison.png
    ├── lion_constraint_analysis.png
    └── final_comparison_all_optimizers.png
```

---

## Dipendenze

- `lion-pytorch` → `pip install lion-pytorch`
- `src/trainer.py` → `train()` (identico ad ADAM — solo l'ottimizzatore cambia)
- `src/loss.py` → `constrained_mse_loss()`
- `src/utils.py` → `compare_optimizers()`, `plot_fitting_curve()`
