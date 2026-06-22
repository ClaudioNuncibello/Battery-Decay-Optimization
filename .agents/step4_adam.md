# Step 4 — ADAM (Lo Standard Industriale)

## Obiettivo
Implementare **ADAM con mini-batching** per mostrare la robustezza di un metodo stocastico adattivo sul "rumore fisico" del dataset (i piccoli recuperi fisiologici della capacità della batteria). Confrontare full-batch vs. mini-batch per analizzare il trade-off varianza/velocità.

> [!NOTE]
> Questo step riutilizza `src/trainer.py` senza alcuna modifica — basta cambiare `optimizer` e `batch_size` nel `TrainConfig`.

---

## File di Esecuzione: `run_adam.py`

```python
import torch
from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import constrained_mse_loss
from src.trainer import train, TrainConfig
from src.utils   import plot_loss_curve, plot_fitting_curve, save_history, compare_optimizers

x, y = load_data('battery_cycle_level_dataset_CLEAN_FINAL.csv')

BATCH_SIZES = [8, 32, -1]   # -1 = full-batch

histories = []
for bs in BATCH_SIZES:
    model     = BatteryDecayModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    config = TrainConfig(
        epochs         = 5000,
        batch_size     = bs,
        log_every      = 100,
        penalty_weight = 1e3,
        use_projection = True,
        min_param_val  = 1e-6,
    )

    history = train(model, optimizer, x, y, constrained_mse_loss, config)
    history.label = f'Adam bs={bs if bs > 0 else "full"}'
    histories.append(history)
    save_history(history, f'results/adam_bs{bs}.pkl')

compare_optimizers(histories, title='ADAM — Confronto Batch Size')
```

---

## Dettagli Matematici

### Update Rule di ADAM
ADAM mantiene due momenti del gradiente:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(primo momento — media)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(secondo momento — varianza non centrata)}$$

Con correzione del bias (per compensare l'inizializzazione a zero):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Aggiornamento:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Gestione Vincoli — Doppia Strategia
1. **Penalizzazione nella Loss** (attiva durante il calcolo del gradiente):
   $$\mathcal{L}_{total} = \text{MSE} + \lambda \cdot \sum_{\theta \in \{\alpha,\beta,\gamma\}} \text{ReLU}(-\theta)^2$$
2. **Proiezione post-step** (safety net dopo l'aggiornamento):
   ```python
   param.data.clamp_(min=1e-6)
   ```

---

## Hyperparametri

| Parametro | Valore | Nota |
|---|---|---|
| Learning rate $\eta$ | 1e-3 | Standard per ADAM |
| $\beta_1$ | 0.9 | Decay del primo momento |
| $\beta_2$ | 0.999 | Decay del secondo momento |
| $\epsilon$ | 1e-8 | Stabilità numerica (evita divisione per zero) |
| `penalty_weight` $\lambda$ | 1e3 | Peso del termine di penalizzazione |
| Batch sizes | 8, 32, full | Confronto dell'effetto del rumore |
| Epochs | 5000 | Sufficiente per la convergenza |

---

## Metriche da Raccogliere

1. **Loss curve per ogni batch size** — evidenziare la varianza (oscillazioni) del mini-batch
2. **Confronto loss finale** vs. batch size (accuratezza vs. stocasticità)
3. **Traiettoria dei parametri** (α, β, γ) — analisi del momentum: come il "ricordo" aiuta a superare i picchi di capacità
4. **Analisi dell'impatto sui vincoli:**
   - Quante volte la proiezione è stata attiva?
   - Il termine di penalizzazione è stato determinante o la proiezione è bastata?
5. **Tempo di esecuzione** per epoch vs. batch size
6. **Confronto diretto con L-BFGS-B e GD:** Loss finale, iterazioni, tempo

---

## Risultati Attesi e Interpretazione

### Effetto del Mini-Batch
| Batch Size | Loss Curve | Precisione Finale |
|---|---|---|
| 8 | Alta varianza, oscillante | Buona (+ regularizzazione implicita) |
| 32 | Media varianza, più stabile | Buona |
| Full (-1) | Bassa varianza, monotona | Alta (deterministica) |

### Ruolo del Noise Fisico del Dataset
Le piccole oscillazioni della capacità (recuperi fisiologici delle batterie) creano una loss surface non perfettamente convessa. Con mini-batch piccoli, ADAM "vede" dati parziali ad ogni step → il rumore stocastico aiuta a sfuggire ai plateau e ai minimi locali superficiali.

**Messaggio chiave per l'esame:** ADAM non è il metodo "più preciso" su questo problema (L-BFGS-B lo supera), ma dimostra la **robustezza stocastica** e l'adattività del learning rate — qualità che lo rendono lo standard nel Deep Learning per dataset grandi e modelli con milioni di parametri.

---

## Output da Salvare

```
results/
├── adam_bs8.pkl
├── adam_bs32.pkl
├── adam_bs-1.pkl
└── plots/
    ├── adam_loss_comparison.png
    └── adam_fitting.png
```

---

## Dipendenze

- `torch.optim.Adam`
- `src/trainer.py` → `train()` con `batch_size` variabile (mini-batch loop interno)
- `src/loss.py` → `constrained_mse_loss()`
- `src/utils.py` → `compare_optimizers()`, `plot_fitting_curve()`
