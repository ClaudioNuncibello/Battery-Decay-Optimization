# Teoria Avanzata degli Ottimizzatori

Questa guida espande la teoria matematica dietro ai 4 ottimizzatori implementati nel progetto. Le formule sono state scritte in formato testo standard (Unicode) così che siano perfettamente leggibili su qualsiasi dispositivo, senza richiedere plugin per la matematica (LaTeX).

---

## 1. Fondamenti di Ottimizzazione Continua

L'obiettivo dell'ottimizzazione è trovare un vettore di parametri θ (nel nostro caso θ = [α, β, γ]) che minimizzi una funzione obiettivo (Loss) continua e differenziabile L(θ).

### Espansione di Taylor
Molti algoritmi si basano sull'espansione in serie di Taylor di L attorno a un punto θ_t:

```text
L(θ_t + Δθ) ≈ L(θ_t) + ∇L(θ_t)^T * Δθ  +  1/2 * Δθ^T * H * Δθ
```

Dove:
- **∇L(θ_t)** è il **Gradiente** (vettore delle derivate prime). Indica la direzione di massima pendenza.
- **H** è la matrice **Hessiana** (matrice quadrata delle derivate seconde parziali). L'Hessiana descrive la curvatura locale (es. se sei in una valle stretta o in una conca piatta).

---

## 2. Gradient Descent (GD) Classico

Il Gradient Descent (o Discesa del Gradiente) si basa solo sull'approssimazione al primo ordine dell'espansione di Taylor. Trascurando l'Hessiana (la curvatura), la direzione di discesa più ripida è semplicemente l'opposto del gradiente.

### Formula di Aggiornamento

```text
θ_{t+1} = θ_t  -  η * ∇L(θ_t)
```

Dove η (eta) > 0 è il **Learning Rate** (o passo di discesa).

### Limiti Matematici: Il Numero di Condizionamento
Se la funzione ha una geometria a "valle stretta", l'Hessiana è sbilanciata. Il GD soffre di un grave problema:
- Lungo l'asse ripido, il gradiente è alto e l'algoritmo "rimbalza" da un lato all'altro (oscillazioni).
- Lungo l'asse piatto, il gradiente è minuscolo e l'algoritmo avanza in modo esasperantemente lento.
Questo fenomeno è noto come **Zig-Zagging**. Inoltre, il GD non decelera in modo intelligente vicino al minimo a meno che il learning rate non venga scalato manualmente.

---

## 3. Metodi Quasi-Newton e L-BFGS-B

Per superare la cecità del GD rispetto alla curvatura, il metodo di Newton usa l'approssimazione al secondo ordine (guarda anche le derivate seconde).

### Metodo di Newton Esatto
L'aggiornamento perfetto minimizzerebbe esattamente l'approssimazione quadratica di Taylor, usando l'inversa dell'Hessiana ( H^(-1) ):

```text
θ_{t+1} = θ_t  -  H^(-1) * ∇L(θ_t)
```

Il problema è che calcolare e invertire H richiede troppa potenza di calcolo se ci sono milioni di parametri (non è il nostro caso, ma lo è per le reti neurali).

### L'approssimazione BFGS
I metodi Quasi-Newton, come BFGS, costruiscono gradualmente una matrice B_t che approssima H^(-1).
Ad ogni passo, misurano la differenza tra i parametri (s_t) e la differenza tra i gradienti (y_t):

```text
s_t = θ_{t+1} - θ_t
y_t = ∇L(θ_{t+1}) - ∇L(θ_t)
```

La matrice viene aggiornata in modo da soddisfare l'**Equazione della Secante**:

```text
B_{t+1} * y_t = s_t
```

### La Variante "L" (Limited-Memory)
In L-BFGS, la matrice B_t non viene mai salvata in memoria per intero. Invece, si tengono in memoria solo gli ultimi 10 passi di (s_t, y_t). Il calcolo dell'aggiornamento viene fatto al volo (algoritmo "two-loop recursion"), rendendolo leggerissimo.

### La Variante "B" (Box Constraints)
Il nostro modello fisico impone α, β, γ > 0. L-BFGS-B gestisce questi "muri" usando il metodo dell'**Active Set**.
Se un parametro sbatte contro il valore zero (limite), viene "congelato" (diventa una variabile attiva) finché il gradiente non punta nuovamente verso l'interno dell'area consentita.

---

## 4. ADAM (Adaptive Moment Estimation)

Adam nasce nel 2014 per superare le difficoltà del GD nelle reti neurali. Combina le idee del Momentum (Inerzia) e dello Scaling della Varianza.

### Il Mini-Batch e il Rumore
A differenza di L-BFGS che analizza tutti i dati assieme, Adam lavora spesso su **Mini-Batch** (sottoinsiemi casuali dei dati). Questo genera un gradiente "rumoroso" (g_t). Il rumore agisce come un regolarizzatore naturale, sballottando i parametri ed evitando che si incastrino in finti minimi locali (buche poco profonde).

### Le Medie Mobili Esponenziali
Adam calcola due memorie per ogni singolo parametro:

**1. Il Primo Momento (m_t, "Inerzia"):**
```text
m_t = β_1 * m_{t-1}  +  (1 - β_1) * g_t
```
Accumula le direzioni passate. Se il gradiente cambia direzione bruscamente a causa di un dato anomalo, l'inerzia impedisce deviazioni estreme. (Solitamente β_1 = 0.9).

**2. Il Secondo Momento (v_t, "Varianza"):**
```text
v_t = β_2 * v_{t-1}  +  (1 - β_2) * g_t^2
```
Se un parametro oscilla molto, v_t cresce in fretta. Adam usa v_t per frenare i parametri instabili e accelerare quelli che si muovono lentamente in una direzione costante. (Solitamente β_2 = 0.999).

### Bias Correction
All'inizio del training (t=1), m_t e v_t partono da zero, quindi sono "viziati" (Bias) verso lo zero. Adam applica una correzione (indicata con il simbolo "hat" o cappelletto):

```text
m_hat_t = m_t / (1 - β_1^t)
v_hat_t = v_t / (1 - β_2^t)
```

### L'Update Finale
La modifica finale applica l'inerzia e divide per la radice della varianza:

```text
θ_{t+1} = θ_t  -  [ η / (√v_hat_t + ε) ] * m_hat_t
```
Dove ε (epsilon) è un numero piccolissimo per evitare divisioni per zero.

---

## 5. Lion (EvoLved Sign Momentum)

Sviluppato da Google Brain, Lion ha un approccio radicale: ignora la magnitudine (il valore assoluto o grandezza) del gradiente. Guarda solo la direzione.

### L'Algoritmo
Lion mantiene una sola memoria esponenziale (Inerzia):

```text
m_t = β_1 * m_{t-1}  +  (1 - β_1) * g_t
```

Invece di dividere per la radice della varianza come Adam, Lion guarda il **segno** matematico (+ o -) dell'Inerzia e fa un passo di lunghezza identica e prestabilita (η):

```text
θ_{t+1} = θ_t  -  η * sign(m_t)
```

### Considerazioni su Lion
- **Memoria Dimezzata:** Salvare solo "m_t" invece di "m_t" e "v_t" significa dimezzare la RAM usata. Per ChatGPT (100 miliardi di parametri), è un risparmio enorme.
- **Passo Costante e Rimbalzi:** Siccome il passo è forzato da "sign()" ad essere sempre grande quanto η, Lion non decelera dolcemente quando si avvicina al bersaglio. Tende a "rimbalzare" indefinitamente attorno al minimo se non abbassi manualmente η.
- **Inadeguatezza per il nostro problema:** Nelle equazioni fisiche precise a 3 parametri come la nostra curva, perdere le informazioni sulla grandezza del gradiente rende Lion cieco rispetto a quanto è profondo il buco che sta esplorando.

---

## 6. L'Impatto Teorico nell'Analisi Sperimentale (Per la Relazione)

Durante la presentazione, puoi basare la comparazione su queste fondamenta:

1. **Il trionfo dell'Hessiana (L-BFGS-B):** Ha richiesto solo 0.01 secondi perché il nostro spazio ha solo 3 dimensioni. L'Hessiana approssimata mappa perfettamente la curvatura a "valle" del nostro decadimento batteria, lanciandosi dritta sul fondo in pochissimi passi.
2. **Il valore del rumore (Adam):** Adam in modalità mini-batch ha dimostrato che il rumore introdotto dai piccoli gruppi di dati lo ha aiutato a "scavalcare" piccole asperità create dalle differenze fisiche tra le 19 batterie, portandolo a un errore leggermente inferiore, anche se al costo di 27 secondi di calcolo.
3. **No Free Lunch Theorem:** Non esiste un algoritmo superiore per tutto. Lion, sebbene all'avanguardia assoluta per Intelligenza Artificiale e linguaggi naturali, si è dimostrato teoricamente sbilanciato per un problema deterministico di Curve Fitting classico.
