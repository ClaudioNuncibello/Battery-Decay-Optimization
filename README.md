# Battery Decay Optimization

Questo progetto analizza le performance di diversi algoritmi di ottimizzazione numerica (Gradient Descent, L-BFGS-B, Adam, Lion) applicati a un problema di curve fitting non lineare con vincoli fisici per la previsione del degrado delle batterie agli ioni di litio.

## 🗂 Navigazione della Repository

Il codice è stato ingegnerizzato separando la logica core dagli script di esecuzione per garantire massima pulizia e modularità.

### 1. `src/` (Core Library)
Contiene l'architettura logica e matematica del progetto. Non contiene script da eseguire direttamente.
- **`model.py`**: Definisce il Modello Parametrico Esponenziale in PyTorch (e la baseline MLP).
- **`loss.py`**: Definisce le funzioni di costo (Loss MSE) e la logica della *Barrier Penalty* per imporre i vincoli fisici.
- **`trainer.py`**: Il ciclo di training generico, compatibile con qualsiasi ottimizzatore (registrazione loss, aggiornamento parametri, proiezioni).
- **`utils.py`**: Funzioni grafiche e di salvataggio (dashboard, barchart, loss surface 3D, caricamento history).
- **`data.py`**: Script per il pre-processing e caricamento del dataset.

### 2. `scripts/` (Entry Points)
Contiene gli script eseguibili che avviano gli esperimenti.
> **Nota bene:** Affinché i percorsi funzionino correttamente, tutti gli script devono essere lanciati dalla *root directory* (cartella principale) usando il flag `-m` di Python.
- `python -m scripts.run_gd`: Avvia l'addestramento con Gradient Descent.
- `python -m scripts.run_lbfgsb`: Avvia l'addestramento con L-BFGS-B.
- `python -m scripts.run_adam`: Avvia l'addestramento con Adam.
- `python -m scripts.run_lion`: Avvia l'addestramento con Lion.
- `python -m scripts.run_mlp`: Avvia l'addestramento della baseline Deep Learning (Multi-Layer Perceptron).
- `python -m scripts.compare_all`: Genera tutti i grafici comparativi avanzati (Dashboard, 3D, BarChart) partendo dai risultati salvati.
- `python -m scripts.compare_mlp`: Genera il grafico di confronto testa-a-testa tra il Modello Fisico e l'MLP.

### 3. `results/` (Output)
Cartella auto-generata in cui vengono salvati:
- I file `.pkl` contenenti lo storico di ogni addestramento (valori loss, traiettorie dei parametri, tempi).
- Le immagini generate (`.png`) pronte per essere allegate alla relazione.

### 4. File nella Root Principale
- **`relazione.tex`**: Il file sorgente LaTeX per la relazione accademica finale.
- **`battery_cycle_level_dataset_CLEAN_FINAL.csv`**: Il dataset pulito (dati originali estratti dal repository NASA).
- **`data/TEORIA.md`**: Un documento markdown di supporto che contiene i fondamenti matematici e di analisi numerica a sostegno del progetto.
