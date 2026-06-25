# Indice della Relazione in LaTeX

## 1. Introduzione e Analisi del Problema
- **Contesto Operativo**: Il degrado delle batterie (NASA Dataset) e l'importanza della previsione del ciclo di vita.
- **Il Problema della Regressione**: Introduzione al metodo dei Minimi Quadrati, il vettore residuo e le difficoltà analitiche dei modelli non lineari.
- **Obiettivo dello Studio**: Dimostrare le performance di diversi ottimizzatori per fittare un'equazione parametrica non lineare $y = \alpha e^{-\beta x} + \gamma$.

## 2. Modello Fisico e Vincoli
- **L'Equazione Esponenziale**: Analisi dei tre parametri ($\alpha, \beta, \gamma$) e del loro significato fisico.
- **Spazio di Ottimizzazione e Vincoli**: Perché servono limiti fisici stretti ($\alpha, \beta, \gamma > 0$) e come la Barrier Penalty gestisce le violazioni.

## 3. Rassegna degli Algoritmi di Ottimizzazione Continua
*(Riferimento teoria: `data/TEORIA.md`)*
- **Metodi del Primo Ordine**: Gradient Descent (GD) classico e l'utilità del campionamento stocastico (SGD).
- **Metodi Adattivi e del Momento (Adam)**: L'inerzia, la stima adattiva del passo e la loro efficacia nel superare lo zig-zagging.
- **Metodi Quasi-Newton (L-BFGS-B)**: L'approssimazione dell'Hessiana per catturare la curvatura esatta dello spazio e il supporto nativo per i "Box Constraints".
- **Innovazioni Estreme (Lion)**: L'eliminazione della magnitudine in favore del segno per ottimizzare l'uso della memoria, e i suoi limiti su scale ridotte.

## 4. Architettura Software e Metodologia
- **Struttura del Codice**: Modularità dell'implementazione in PyTorch (separazione tra package `src/` e cartella script `scripts/`).
- **Pipeline Sperimentale**: L'estrazione dei dati, calcolo della loss e valutazione dell'efficienza temporale per ogni ottimizzatore.

## 5. Analisi dei Risultati
- **Confronto delle Metriche**: Analisi del Bar Chart (L-BFGS-B dominatore in $0.01$ s contro decine di secondi di Adam/Lion).
- **Dinamica di Convergenza Geometrica**: Analisi della Loss Surface 3D. Discussione della traiettoria diretta di L-BFGS-B rispetto all'andamento di Adam/Lion.

## 6. Il "No Free Lunch": Modello Fisico vs Deep Learning (MLP)
- **Implementazione della Baseline**: Costruzione di un Multi-Layer Perceptron (MLP) a 32 neuroni (~1100 parametri).
- **Confronto Teorico-Pratico**: L'MLP soffre della maledizione della sovra-parametrizzazione (fitta il rumore, leggero overfitting), annienta l'interpretabilità (scatola nera) e costa il 30000% di tempo in più di calcolo senza reali benefici predittivi asintotici rispetto al modello fisico L-BFGS-B.

## 7. Conclusioni
- **Sintesi dello Studio**: Riallineamento dei risultati pratici con l'Analisi Numerica. Sugli scenari a bassa dimensionalità ben posti fisicamente, il Machine Learning purista e gli approcci classici del secondo ordine stravincono contro il Deep Learning standard.
