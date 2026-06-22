# Progetto di Calcolo Numerico: Ottimizzazione Vincolata per il Curve Fitting

## 1. Contesto del Progetto
Questo progetto costituisce l'esame finale per il corso di *Numerical Methods for Scientific Computing* (Prof. Boscarino). L'obiettivo non è l'addestramento di un modello di Machine Learning "black-box" (come un MLP), ma l'**analisi numerica e comparativa di diversi metodi di ottimizzazione** applicati a un problema reale di *curve fitting* parametrico con vincoli fisici.

Il progetto mira a dimostrare la stretta connessione tra i metodi numerici classici (es. approcci Quasi-Newton) e le moderne tecniche di addestramento stocastico utilizzate nel Deep Learning, analizzandone le traiettorie di convergenza, la gestione dei vincoli e l'efficienza computazionale.

## 2. Il Dataset (NASA Battery Degradation)
Il progetto utilizza il dataset pubblico "NASA Battery Cycle-Level Dataset". 
* **Dominio:** Degradazione fisica dell'hardware (batterie agli ioni di litio).
* **Dati:** Log sperimentali estratti dai laboratori NASA che tracciano la capacità residua della batteria all'aumentare dei cicli di ricarica.
* **Caratteristiche Numeriche:** La mole di dati (migliaia di registrazioni su diverse batterie) e la presenza di "rumore fisico" (piccoli recuperi fisiologici della capacità) offrono una superficie della funzione di costo complessa e sfidante, ideale per confrontare algoritmi *Full-Batch* e *Stocastici* (Mini-batching).

## 3. Il Problema Matematico e Architettura
L'architettura del modello non utilizza strati neurali classici, ma implementa un **Modello Parametrico Custom** in PyTorch che modella l'equazione fisica del decadimento della batteria.

### L'Equazione di Decadimento (Funzione Obiettivo)
$$Capacity(x) = \alpha \cdot e^{-\beta x} + \gamma$$

Dove la variabile indipendente $x$ rappresenta il numero di cicli (Cycle), e la variabile dipendente è la Capacità (Capacity).

### Parametri Addestrabili
L'obiettivo degli ottimizzatori è scoprire i valori perfetti di 3 parametri (definiti come `nn.Parameter` in PyTorch):
1. **$\alpha$**: La porzione di capacità "sacrificabile" che si degrada nel tempo.
2. **$\beta$**: Il tasso chimico di degradazione.
3. **$\gamma$**: L'asintoto (la capacità residua "morta" a fine vita).

### Ottimizzazione Vincolata (Box Constraints)
Trattandosi di un sistema fisico, i parametri sono soggetti a vincoli rigidi invalicabili:
* $\alpha > 0$ (La capacità non può essere negativa).
* $\beta > 0$ (Il tasso di decadimento deve causare una discesa, non una salita).
* $\gamma > 0$ (L'energia residua non può scendere sotto lo zero).

## 4. Gli Algoritmi sotto Analisi
Il progetto mette a confronto 4 ottimizzatori appartenenti a famiglie matematiche profondamente diverse:

1. **Discesa del Gradiente Classico (Baseline):** * Metodo del primo ordine, *full-batch*, passo rigido. 
   * *Ruolo:* Dimostrare i limiti storici (lentezza, intrappolamento nei minimi locali) e giustificare la necessità di metodi avanzati.
2. **L-BFGS-B (Il Campione Numerico):** * Algoritmo Quasi-Newton a memoria limitata. Sfrutta l'approssimazione della matrice Hessiana (derivata seconda) per mappare la curvatura della funzione. La variante "-B" gestisce nativamente i *Box Constraints*.
   * *Ruolo:* Dimostrare la supremazia teorica e la precisione chirurgica dei metodi classici su problemi ben definiti.
3. **ADAM (Lo Standard Industriale):**
   * Metodo stocastico del primo ordine con passo adattivo e *momentum*.
   * *Ruolo:* Mostrare la robustezza e l'efficienza nel superare il "rumore fisico" del dataset tramite l'uso di mini-batch.
4. **Lion - EvoLved Sign Momentum (L'Avanguardia):**
   * Algoritmo scoperto tramite ricerca automatizzata (Google Brain). Usa l'operatore del `segno` per determinare la direzione, ignorando la magnitudo del gradiente.
   * *Ruolo:* Analizzare come un algoritmo nato per enormi modelli linguistici si comporti (in termini di stabilità della Loss e consumo di memoria) su un problema di regressione parametrica.

## 5. Gestione dei Vincoli nell'Implementazione
Poiché ADAM, Lion e il Gradiente Classico sono algoritmi nativamente "non vincolati", il rispetto dei limiti fisici ($\alpha, \beta, \gamma > 0$) sarà garantito matematicamente tramite uno dei seguenti approcci:
* **Metodi di Penalizzazione:** Modifica della funzione di costo (Loss) con l'aggiunta di termini di barriera che esplodono se i parametri tentano di uscire dallo spazio ammissibile.
* **Proiezione del Gradiente:** Forzatura dei pesi (clipping/projection) all'interno dei margini al termine di ogni passo di aggiornamento.

## 6. Metriche ed Estrazione Risultati
L'implementazione sarà curata per separare la logica di addestramento dall'estrazione delle metriche. Si utilizzeranno framework agili (es. PyTorch) esclusivamente per sfruttare l'Autograd. 
All'esame verranno presentati:
1. Grafici 2D/3D delle traiettorie di convergenza della Loss.
2. Curve di fitting sovrapposte ai dati reali sparsi.
3. Analisi della reazione degli algoritmi all'impatto con il "muro" dei vincoli fisici.