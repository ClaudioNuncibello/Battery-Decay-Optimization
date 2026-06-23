# Teoria Avanzata degli Ottimizzatori e Minimi Quadrati

Questa guida espande la teoria matematica dietro ai modelli di regressione e agli ottimizzatori implementati nel progetto, integrando i fondamenti dell'Analisi Numerica. Le formule sono state scritte in formato testo standard (Unicode) per la massima leggibilità.

---

## 1. Il Problema della Regressione e i Minimi Quadrati

Nel contesto del Machine Learning e del curve fitting, l'obiettivo della regressione lineare o non lineare è trovare una funzione `F(x; θ)`, dipendente da un vettore di parametri `θ`, che approssimi al meglio un set di dati misurati `(x_i, y_i)` per `i = 1, ..., m`.

### Il Vettore Residuo e la Funzione di Costo
Si definisce l'**errore o residuo** per ogni punto come la discrepanza tra il valore misurato e quello predetto dal modello:
`r_i = y_i - F(x_i; θ)`

Il **Metodo dei Minimi Quadrati** richiede che sia minima la lunghezza euclidea del vettore residuo (varianza). Questo si esplica minimizzando la funzione di costo (Loss) definita come la Somma degli Errori Quadratici:
`L(θ) = ||r||_2^2 = Σ (y_i - F(x_i; θ))^2`

### Sistemi Sovradeterminati ed Equazioni Normali
Nei casi lineari (es. `F(x; θ) = X θ`), se il numero di equazioni / dati `m` è maggiore del numero di parametri `n` (sistema sovradeterminato), il sistema `X θ = y` in genere non ammette una soluzione esatta.
Per minimizzare `||X θ - y||_2^2`, imponendo che il gradiente sia nullo (`∇L(θ) = 0`), si ricava il **sistema delle Equazioni Normali**:
`X^T X θ = X^T y`

**Criticità:** La matrice `X^T X` è spesso mal condizionata. Gli errori introdotti nei dati e nel processo di approssimazione tendono ad amplificarsi vertiginosamente. Sebbene si possano usare tecniche di decomposizione robusta (come la decomposizione ai valori singolari **SVD** o **QR**), per modelli vasti o non lineari (come il nostro decadimento batteria esponenziale) la via analitica diretta non è percorribile e si deve ricorrere ai **metodi iterativi di discesa**.

---

## 2. Fondamenti di Ottimizzazione Continua

I metodi iterativi (o metodi di discesa) generano una successione di vettori `θ^{(0)}, θ^{(1)}, ...` che converge al parametro ottimale `θ^*`. L'aggiornamento, al passo `k`, è della forma:
`θ^{(k+1)} = θ^{(k)} + α_k p^{(k)}`
dove `p^{(k)}` è la **direzione di ricerca** e `α_k` è lo **step size** (learning rate o passo di discesa). Affinché ci sia una riduzione della Loss, la direzione `p^{(k)}` deve formare un angolo ottuso col gradiente, garantendo `∇L(θ^{(k)})^T p^{(k)} < 0`.

### Espansione di Taylor
Molti algoritmi si basano sull'espansione in serie di Taylor della Loss `L` attorno a `θ^{(k)}`:
`L(θ^{(k)} + Δθ) ≈ L(θ^{(k)}) + ∇L(θ^{(k)})^T Δθ  +  (1/2) Δθ^T H Δθ`

Dove:
- **`∇L(θ)`** è il **Gradiente** (vettore delle derivate prime). La sua direzione opposta garantisce la massima decrescita locale.
- **`H`** è la matrice **Hessiana** (derivate seconde). Descrive la curvatura dello spazio dei parametri.

---

## 3. Gradient Descent (GD) e Stochastic Gradient Descent (SGD)

### Gradient Descent Classico
Il metodo del gradiente classico impone come direzione di discesa esattamente l'opposto del gradiente: `p^{(k)} = - ∇L(θ^{(k)})`. La formula di aggiornamento è:
`θ^{(k+1)} = θ^{(k)}  -  η ∇L(θ^{(k)})`

**Analisi Matematica e Limiti:**
Per minimizzare un'energia quadratica pura `(1/2) θ^T A θ - θ^T b` (con `A` simmetrica e definita positiva), il parametro ottimale `α_k` si ricava minimizzando analiticamente l'energia lungo la direzione scelta. Ponendo a zero la derivata rispetto ad `α_k`, si ottiene:
`α_k = (r^{(k)T} r^{(k)}) / (r^{(k)T} A r^{(k)})`
dove `r^{(k)}` è il residuo.
Tuttavia, il GD soffre enormemente se la matrice `A` ha un alto numero di condizionamento `κ(A)` (geometria a "valle stretta"): le iterazioni rimbalzano lungo i pendii ripidi (effetto **Zig-Zagging**) rallentando drasticamente la convergenza.

### Stochastic Gradient Descent (SGD)
Valutare `∇L(θ)` sull'intero set di dati (Full Batch) a ogni iterazione è computazionalmente oneroso. L'SGD o il Mini-batch GD valuta il gradiente solo su un sottoinsieme randomico di campioni. Il "rumore" stocastico introdotto dal mini-batch aiuta l'algoritmo a scavalcare falsi minimi locali e buche poco profonde della funzione Loss, generalizzando meglio il modello.

---

## 4. Oltre il GD: Gradiente Coniugato e Metodo dei Momenti

Per superare la lentezza e l'oscillazione del GD, l'Analisi Numerica ha sviluppato metodi più raffinati che includono memorie storiche e correzioni ortogonali.

### Metodo del Gradiente Coniugato (CG)
Invece di muoversi passivamente nella direzione opposta al gradiente a ogni passo (il che distrugge il lavoro fatto nei passi precedenti inducendo zig-zag), il CG sceglie direzioni di discesa `p^{(k)}` "coniugate" (A-ortogonali, ovvero `p^{(i)T} A p^{(j)} = 0`). 
L'aggiornamento iterativo definisce nuove direzioni come combinazione lineare tra il gradiente negativo e la direzione precedente:
`p^{(k+1)} = - ∇L(θ^{(k+1)}) + β_k p^{(k)}`
dove `β_k` è un parametro calcolato (es. formula di Fletcher-Reeves). Questo assicura teoricamente la convergenza per sistemi quadratici in un numero di iterazioni minore o uguale al numero totale di parametri `n`.

### Metodo dei Momenti
Introduce una "memoria" nell'algoritmo definendo un vettore di momentum `m^{(k)}` per accelerare la convergenza e smorzare lo zig-zagging.
`m^{(k)} = β_1 m^{(k-1)} + (1 - β_1) ∇L(θ^{(k)})`
`θ^{(k+1)} = θ^{(k)} - η m^{(k)}`
Il termine di momentum aggrega le informazioni dei gradienti passati: agisce come un filtro passa-basso, stabilizzando le traiettorie increspate e permettendo ai parametri di guadagnare inerzia lungo le direzioni a pendenza costante.

---

## 5. ADAM (Adaptive Moment Estimation)

Adam (2014) è l'ottimizzatore standard e dominante nel Deep Learning. Combina in modo geniale i benefici del Momentum (primo ordine) con una regolazione adattiva del learning rate tramite la varianza dei gradienti (secondo ordine).

### Le Formule di ADAM
Ad ogni iterazione, Adam aggiorna due "momenti":

1. **Primo Momento (Media esponenziale del gradiente / Inerzia):**
   `m^{(k)} = β_1 m^{(k-1)}  +  (1 - β_1) g^{(k)}`
2. **Secondo Momento (Media esponenziale del gradiente al quadrato / Varianza):**
   `v^{(k)} = β_2 v^{(k-1)}  +  (1 - β_2) (g^{(k)})^2`

Poiché `m^{(0)}` e `v^{(0)}` partono da 0, presentano un *bias* iniziale verso lo zero (specie se `β_1, β_2` sono vicini a 1, es. 0.9 e 0.999). Per correggerlo si introducono i correttori $\hat{m}$ e $\hat{v}$:
`\hat{m}^{(k)} = m^{(k)} / (1 - β_1^k) \quad , \quad \hat{v}^{(k)} = v^{(k)} / (1 - β_2^k)`

**Aggiornamento dei Parametri:**
`θ^{(k+1)} = θ^{(k)}  -  (η / (\sqrt{\hat{v}^{(k)}} + ε)) \hat{m}^{(k)}`
(dove `ε ≈ 10^{-8}` previene divisioni per zero).

### L'Approssimazione Continua alle Equazioni Differenziali (ODEs ADAM)
Una visione moderna dell'Analisi Numerica (Silva e Gazeau, 2020) dimostra che, nel limite in cui il learning rate `h` tende a zero, ADAM modella un **sistema di equazioni differenziali ordinarie (ODE)**. Con un tempo continuo `t`, la dinamica di ADAM è descritta dalla famiglia parametrica:
`θ'(t) = - m(t) / (\sqrt{v(t)} + ε)`
Analizzare ADAM tramite la sua "Formulazione Continua" fornisce strumenti teorici potentissimi per comprendere e tracciare la fluidodinamica dell'ottimizzazione e per ideare estensioni numeriche di time-stepping di ordine superiore, più efficienti su complessi scenari di regressione.

---

## 6. Metodi Quasi-Newton e L-BFGS-B

Se il Metodo di Newton esatto richiede di invertire l'Hessiana `H^{-1}`, un processo proibitivo per un gran numero di variabili (`O(n^3)` di costo computazionale per step), i metodi **Quasi-Newton** approssimano `H^{-1}` in modo iterativo basandosi solo sull'osservazione dei gradienti correnti e passati.

### L'Approssimazione BFGS
L'algoritmo BFGS costruisce una matrice `B_k` che fa da proxy per l'inversa dell'Hessiana. Monitora due vettori principali:
- La differenza dei parametri: `s_k = θ^{(k+1)} - θ^{(k)}`
- La differenza dei gradienti: `y_k = ∇L(θ^{(k+1)}) - ∇L(θ^{(k)})`

La matrice si aggiorna imponendo l'**Equazione della Secante**: `B_{k+1} y_k = s_k`.

### L-BFGS-B (Limited Memory & Box Constraints)
- **L (Limited-Memory):** Invece di allocare e calcolare interamente una matrice densa `n x n`, si memorizzano solo gli ultimi `m` step (tipicamente gli ultimi 10 vettori `s_k` e `y_k`). Questo permette un drastico taglio nei costi di memoria.
- **B (Box Constraints):** Permette l'imposizione di vincoli stretti sui parametri (es. costringere i parametri della batteria `α, β, γ > 0`) usando l'approccio matematico dell'**Active Set**. Se un parametro urta un vincolo ("muro"), la direzione di step viene congelata lungo tale dimensione finché il gradiente derivativo non punterà verso una regione ammissibile interna. 

---

## 7. Lion (EvoLved Sign Momentum)

Sviluppato da Google Brain per il pre-training di architetture massive, Lion introduce un approccio non-stazionario minimalista che converte le informazioni di momento in istruzioni binarie pure (il segno).

### L'Algoritmo
Lion calcola il momento classico:
`m^{(k)} = β_1 m^{(k-1)}  +  (1 - β_1) g^{(k)}`

L'innovazione estrema è rimuovere la magnitudine (lunghezza/valore assoluto) dello step, lasciando **esclusivamente il segno (+1 o -1)**:
`θ^{(k+1)} = θ^{(k)}  -  η \cdot sign(m^{(k)})`

### Considerazioni Analitiche
1. **Risparmio di Memoria:** Rispetto ad Adam, viene soppresso il tracciamento del secondo momento (la varianza `v^{(k)}`), il che dimezza il consumo di VRAM: un fattore vincente nell'era dei Large Language Models con 100+ miliardi di parametri.
2. **Natura dello Step Costante:** Essendo il modulo del passo obbligato da `sign(\cdot)` ad essere pari ad `η`, l'algoritmo non è matematicamente in grado di ridurre "dolcemente" le falcate mentre si cala nel minimo globale. Senza una strategia manuale rigida di smorzamento di `η` (learning rate decay), Lion andrà fatalmente a saltare all'infinito (rimbalzi perpetui in un range di ampiezza `± η`) attorno all'ottimo senza decantarvi.

---

## 8. Criteri di Arresto (Test di Convergenza)

Gli algoritmi iterativi sopra menzionati non dispongono di fine corsa logico ma necessitano di **stopping criteria** rigorosi rispetto a una tolleranza `tol` utente:

1. **Criterio dello Step:** Interrompe i calcoli alla `k+1`-esima iterazione se l'aggiornamento dello spazio dei parametri diventa trascurabile:
   `||θ^{(k+1)} - θ^{(k)}|| ≤ tol`
2. **Criterio del Residuo o del Gradiente:** Interrompe i calcoli qualora il versore della pendenza della Loss tocchi lo zero (verificando la natura stazionaria del punto):
   `||∇L(θ^{(k)})|| ≤ tol`

---

## 9. Analisi Sperimentale Finale e Osservazioni sul "No Free Lunch"

L'ecosistema dell'implementazione `Battery-Decay-Optimization` rispecchia limpidamente i dettami della Teoria Numerica di cui sopra:

1. **La Supremazia dell'Hessiana (L-BFGS-B):** Poiché lo spazio di simulazione consta di appena tre dimensioni puramente deterministiche, la matrice Hessiana approssimata mappa impeccabilmente il bacino quadratico e l'algoritmo plana dritto sul minimo analitico in meno di $0.02$ secondi complessivi. È l'apice del Curve Fitting Classico su spazi limitati.
2. **La Resilienza del Rumore Stocastico in ADAM:** Gestendo molteplici batterie affette da interferenze e rumori misurativi, il partizionamento a blocchi casuali (mini-batch) accoppiato con l'inerzia del momento permette ad ADAM di assorbire e filtrare i picchi anomali sfuggendo a sub-ottimi che il L-BFGS-B subirebbe. Ciò costa un aumento computazionale radicale (ordine dei decine di secondi), ben compensato però dalla stabilità del modello in fase test.
3. **Le Debolezze del Vettore Segno:** LION, dominatore nell'high-dimensional AI, ha fallito di fronte a una parametrizzazione a bassa entropia. L'assenza di percezione geometrica derivata dall'ignorare l'intensità puntuale (il modulo) del gradiente l'ha reso matematicamente inabile a interpretare con finezza l'andamento del "fondo cassa" della funzione di batteria, portando a oscillazioni irrisolte. Non c'è e non vi sarà mai un algoritmo universale superiore (*No Free Lunch Theorem*).
