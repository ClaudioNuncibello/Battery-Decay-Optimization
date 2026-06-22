# Step 1 — Creazione della Rete e Infrastruttura Comune

## Obiettivo
Costruire l'**architettura riutilizzabile** del progetto: data pipeline, modello fisico parametrico, loop di training generico (che accetta qualsiasi ottimizzatore), e sistema di logging/plotting.

Tutti gli step successivi (2–5) importeranno da questi moduli senza modificarli.

---

## Struttura dei File

```
ProgettoAnalisiNumerica/
│
├── battery_cycle_level_dataset_CLEAN_FINAL.csv
│
├── src/
│   ├── data.py          # Data loading e preprocessing
│   ├── model.py         # BatteryDecayModel (nn.Module)
│   ├── loss.py          # MSE Loss + penalizzazione vincoli
│   ├── trainer.py       # Loop di training GENERICO (riutilizzabile)
│   └── utils.py         # Plotting, metriche, salvataggio history
│
├── run_gd.py            # Step 2: Gradient Descent classico
├── run_lbfgsb.py        # Step 3: L-BFGS-B
├── run_adam.py          # Step 4: ADAM
└── run_lion.py          # Step 5: Lion
```

> [!IMPORTANT]
> Il file `trainer.py` è il cuore del progetto. Riceve il modello e l'ottimizzatore già configurati come parametri — non sa nulla di chi li ha creati. Ogni `run_*.py` si occupa solo di istanziare il proprio ottimizzatore e chiamare `trainer.train(model, optimizer, data)`.

---

## 1.1 — `src/data.py` — Data Pipeline

**Responsabilità:**
- Carica il CSV con `pandas`
- Filtra le batterie con ≥ 30 cicli (fitting globale)
- Normalizza i cicli: `x = cycle / max_cycle` per stabilità numerica
- Restituisce tensori PyTorch `(x_tensor, y_tensor)` pronti per il training

**Interfaccia pubblica:**
```python
def load_data(csv_path: str, min_cycles: int = 30) -> tuple[Tensor, Tensor]:
    """
    Ritorna (x, y) normalizzati come tensori float32.
    x = cicli normalizzati in [0, 1]
    y = capacità in Ah
    """
```

**Fitting globale:** tutti i dati delle batterie filtrate vengono concatenati in un unico tensore. Il modello troverà i parametri (α, β, γ) ottimali per l'intera distribuzione.

---

## 1.2 — `src/model.py` — BatteryDecayModel

**Equazione fisica:**
$$\text{Capacity}(x) = \alpha \cdot e^{-\beta x} + \gamma$$

**Implementazione:**
```python
class BatteryDecayModel(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=0.5, gamma_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta  = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x: Tensor) -> Tensor:
        return self.alpha * torch.exp(-self.beta * x) + self.gamma

    def get_params(self) -> dict:
        return {
            'alpha': self.alpha.item(),
            'beta':  self.beta.item(),
            'gamma': self.gamma.item()
        }
```

**Punto chiave:** i 3 parametri sono `nn.Parameter` → PyTorch Autograd calcola i gradienti automaticamente per tutti gli ottimizzatori.

---

## 1.3 — `src/loss.py` — Loss Function con Vincoli

Due modalità gestite dallo stesso file:

### Modalità A — Penalizzazione Barriera (per GD, ADAM, Lion)
$$\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda \cdot \left[\text{ReLU}(-\alpha)^2 + \text{ReLU}(-\beta)^2 + \text{ReLU}(-\gamma)^2\right]$$

```python
def constrained_mse_loss(pred, target, model, penalty_weight=1e3) -> Tensor:
```

### Modalità B — MSE Pura (per L-BFGS-B che gestisce i vincoli nativamente)
```python
def mse_loss(pred, target) -> Tensor:
```

---

## 1.4 — `src/trainer.py` — Loop di Training GENERICO ⭐

```python
def train(
    model:     nn.Module,
    optimizer: Optimizer | None,   # None = L-BFGS-B usa scipy direttamente
    x:         Tensor,
    y:         Tensor,
    loss_fn:   Callable,
    config:    TrainConfig,        # dataclass con epochs, batch_size, log_every
) -> TrainHistory:
    """
    Loop di training riutilizzabile.

    - Full-batch:  batch_size = len(x) (GD classico, L-BFGS-B)
    - Mini-batch:  batch_size < len(x) (ADAM, Lion)

    Ritorna TrainHistory con loss_curve, param_trajectory, elapsed_time.
    """
```

**`TrainConfig`** — dataclass con tutti i parametri di training:
```python
@dataclass
class TrainConfig:
    epochs:       int   = 5000
    batch_size:   int   = -1        # -1 = full-batch
    log_every:    int   = 100
    penalty_weight: float = 1e3
    use_projection: bool  = True    # clamp post-step
    min_param_val:  float = 1e-6   # floor per la proiezione
```

**`TrainHistory`** — dataclass con i risultati:
```python
@dataclass
class TrainHistory:
    losses:     list[float]         # loss ad ogni epoch loggata
    params:     list[dict]          # {'alpha':..., 'beta':..., 'gamma':...}
    elapsed_s:  float               # tempo totale in secondi
    optimizer_name: str
```

---

## 1.5 — `src/utils.py` — Plotting e Metriche

Funzioni di visualizzazione riutilizzabili da tutti gli step:

```python
def plot_loss_curve(history: TrainHistory, ax=None) -> None
def plot_fitting_curve(model, x, y, ax=None) -> None
def plot_param_trajectory(histories: list[TrainHistory]) -> None
def compare_optimizers(histories: list[TrainHistory]) -> None
def save_history(history: TrainHistory, path: str) -> None
def load_history(path: str) -> TrainHistory
```

---

## Pattern di Utilizzo (come ogni `run_*.py` usa il trainer)

```python
# Esempio: run_adam.py
from src.data    import load_data
from src.model   import BatteryDecayModel
from src.loss    import constrained_mse_loss
from src.trainer import train, TrainConfig
from src.utils   import plot_loss_curve, plot_fitting_curve

# 1. Carica dati
x, y = load_data('battery_cycle_level_dataset_CLEAN_FINAL.csv')

# 2. Istanzia modello
model = BatteryDecayModel()

# 3. Configura il TUO ottimizzatore
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Chiama il trainer GENERICO (identico per tutti gli step)
config  = TrainConfig(epochs=5000, batch_size=32)
history = train(model, optimizer, x, y, constrained_mse_loss, config)

# 5. Visualizza risultati
plot_loss_curve(history)
plot_fitting_curve(model, x, y)
```

---

## Dipendenze da Installare

```bash
pip install torch pandas numpy matplotlib lion-pytorch scipy
```

---

## Verifica Step 1

- [ ] `src/data.py` carica correttamente il CSV e restituisce tensori
- [ ] `BatteryDecayModel.forward()` produce output per un batch di x
- [ ] `constrained_mse_loss()` cresce quando α, β o γ < 0
- [ ] `trainer.train()` completa una run di 100 epoch senza errori
- [ ] `plot_fitting_curve()` visualizza correttamente i dati raw + curva fittata
