"""
src/data.py
-----------
Data pipeline per il dataset NASA Battery Degradation.
Carica il CSV, filtra le batterie con abbastanza cicli,
normalizza l'asse x e restituisce tensori PyTorch pronti per il training.

Usabile sia dagli script run_*.py sia direttamente nel notebook.
"""

import pandas as pd
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
DEFAULT_MIN_CYCLES = 30
DEFAULT_CSV = "battery_cycle_level_dataset_CLEAN_FINAL.csv"


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def load_data(
    csv_path: str = DEFAULT_CSV,
    min_cycles: int = DEFAULT_MIN_CYCLES,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Carica e prepara il dataset NASA Battery per il curve fitting.

    Parametri
    ----------
    csv_path : str
        Percorso al file CSV.
    min_cycles : int
        Numero minimo di cicli per includere una batteria.
    normalize : bool
        Se True, normalizza i cicli in [0, 1] dividendo per il massimo globale.
    dtype : torch.dtype
        Tipo dei tensori di output (default: float32).

    Ritorna
    -------
    x : Tensor shape (N,)
        Cicli (normalizzati se normalize=True).
    y : Tensor shape (N,)
        Capacità in Ah.
    meta : dict
        Informazioni sul dataset:
        - 'batteries'  : lista delle batterie incluse
        - 'n_samples'  : numero totale di campioni
        - 'max_cycle'  : massimo ciclo (usato per la normalizzazione)
        - 'normalized' : bool
        - 'df'         : DataFrame filtrato (utile per plot per-batteria)
    """
    df = pd.read_csv(csv_path)

    # --- Filtraggio batterie con abbastanza cicli ---
    cycle_counts = df.groupby("battery_id")["cycle"].max()
    valid_batteries = sorted(
        cycle_counts[cycle_counts >= min_cycles].index.tolist()
    )

    if not valid_batteries:
        raise ValueError(
            f"Nessuna batteria con almeno {min_cycles} cicli trovata in {csv_path}"
        )

    df_filtered = df[df["battery_id"].isin(valid_batteries)].copy()
    df_filtered = df_filtered.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

    # --- Normalizzazione ---
    max_cycle = int(df_filtered["cycle"].max())
    if normalize:
        df_filtered["x"] = df_filtered["cycle"].astype(float) / max_cycle
    else:
        df_filtered["x"] = df_filtered["cycle"].astype(float)

    # --- Tensori ---
    x = torch.tensor(df_filtered["x"].values, dtype=dtype)
    y = torch.tensor(df_filtered["capacity"].values, dtype=dtype)

    meta = {
        "batteries":  valid_batteries,
        "n_batteries": len(valid_batteries),
        "n_samples":  len(df_filtered),
        "max_cycle":  max_cycle,
        "normalized": normalize,
        "df":         df_filtered,          # utile per plot per-batteria nel notebook
    }

    print(f"[data] Caricate {meta['n_batteries']} batterie, "
          f"{meta['n_samples']} campioni totali "
          f"(cicli max={max_cycle}, normalize={normalize})")
    print(f"[data] Batterie: {valid_batteries}")

    return x, y, meta


# ---------------------------------------------------------------------------
# Utility per il notebook: dati per singola batteria
# ---------------------------------------------------------------------------
def get_battery_data(
    meta: dict,
    battery_id: str,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ritorna (x, y) per una singola batteria dal DataFrame memorizzato in meta.
    Utile per plot per-batteria nel notebook.
    """
    df = meta["df"]
    df_b = df[df["battery_id"] == battery_id]
    if df_b.empty:
        raise ValueError(f"Batteria '{battery_id}' non trovata nel dataset filtrato.")

    x = torch.tensor(df_b["x"].values, dtype=torch.float32)
    y = torch.tensor(df_b["capacity"].values, dtype=torch.float32)
    return x, y
