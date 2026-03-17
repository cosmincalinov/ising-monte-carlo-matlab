# Modelul Ising 2D – Simulare Monte Carlo cu Paralelizare (parfor)

> **Tema 01-4** · Modelarea și Simularea Sistemelor Complexe · 2026

---

## Descrierea proiectului

Acest proiect implementează simularea **Modelului Ising 2D** folosind algoritmul
**Metropolis Monte Carlo** în MATLAB/Octave.

Modelul Ising este un model fundamental din fizica statistică ce descrie un sistem
de spini magnetici (+1/−1) dispuși pe o rețea bidimensională. Este studiat intens
deoarece prezintă o **tranziție de fază de ordinul 2** la temperatura critică
Onsager:

```
T_c = 2·J / (k_B · ln(1 + √2)) ≈ 2.269  [unități J/k_B]
```

Sub `T_c` sistemul este feromagnetic (spini aliniați, magnetizare spontană).
Peste `T_c` sistemul devine paramagnetic (dezordine termică, magnetizare ≈ 0).

---

## Fișiere

| Fișier | Descriere |
|--------|-----------|
| `ising_model.m` | Script MATLAB/Octave complet cu simularea, paralelizarea și graficele |
| `README.md` | Documentație (acest fișier) |

---

## Cum se rulează

### MATLAB (cu sau fără Parallel Computing Toolbox)

```matlab
% Deschide MATLAB, navighează în folderul proiectului și rulează:
ising_model
```

Dacă **Parallel Computing Toolbox** este disponibil, scriptul detectează automat
și folosește `parfor` pentru a paraleliza bucla peste temperaturi.
În caz contrar, folosește `for` secvențial.

### Octave

```bash
# Din terminal:
cd /calea/catre/proiect
octave ising_model.m
```

Sau din interfața grafică Octave:

```octave
>> ising_model
```

> **Notă:** Scriptul este scris pentru compatibilitate deplină cu Octave ≥ 6.0
> și MATLAB ≥ R2018b.

---

## Parametri simulare

| Parametru | Valoare implicită | Descriere |
|-----------|-------------------|-----------|
| `L` | 30 | Dimensiunea rețelei (L × L spini) |
| `J` | 1.0 | Constanta de cuplaj (J > 0 → feromagnet) |
| `h` | 0.0 | Câmp magnetic extern |
| `k_B` | 1.0 | Constanta Boltzmann (unități reduse) |
| `T_vec` | 1.0 : 0.1 : 4.0 | Vector temperaturi simulate |
| `n_steps` | 500 · L² | Pași Monte Carlo per temperatură (măsurători) |
| `n_therm` | 100 · L² | Pași de termarizare (echilibrare, aruncați) |

---

## Algoritmul Metropolis

```
Pentru fiecare pas Monte Carlo (sweep = L² încercări):
  1. Alege un spin aleator (i, j)
  2. Calculează suma vecinilor cu condiții periodice la margine
  3. delta_E = 2·J·s(i,j)·suma_vecini + 2·h·s(i,j)
  4. Dacă delta_E < 0  → acceptă întotdeauna (stare mai stabilă)
     Altfel            → acceptă cu probabilitate exp(−delta_E / (k_B·T))
  5. Actualizează configurația și energia
```

Condițiile periodice la margine ("rețea torică"):
```matlab
i_up    = mod(i - 2, L) + 1   % vecinul de sus
i_down  = mod(i,     L) + 1   % vecinul de jos
```

---

## Rezultate așteptate

Scriptul generează **4 grafice** într-o singură figură:

1. **|M| vs T** – Magnetizarea absolută scade de la ~1 la ~0 în jurul lui `T_c ≈ 2.269`
2. **⟨E⟩ vs T** – Energia medie per spin crește monoton cu temperatura
3. **C_v vs T** – Caldura specifică prezintă un maxim (divergență logaritmică) la `T ≈ T_c`
4. **Configurații rețea** – Trei instantanee la `T < T_c`, `T ≈ T_c`, `T > T_c` vizualizate cu `imagesc`

### Semne că simularea funcționează corect:
- Maximul din `C_v` apare la `T ≈ 2.2–2.4` (efect de dimensiune finită pentru L=30)
- Magnetizarea la `T = 1.0` trebuie să fie aproape de `1.0`
- Magnetizarea la `T = 4.0` trebuie să fie aproape de `0.0`
- Configurația la `T = 1.5` arată domenii mari albe/negre (ordine)
- Configurația la `T = 3.5` arată zgomot aleator (dezordine)

---

## Cerințe software

- **MATLAB** ≥ R2018b  
  *(opțional: Parallel Computing Toolbox pentru `parfor`)*
- **SAU Octave** ≥ 6.0 *(gratuit, open-source: https://octave.org)*

---

## Structura comentariilor (Tema 01-4)

Scriptul conține:
- **Teoria modelului Ising** (Hamiltonian, Metropolis, temperatura Onsager)
- **Operații logice documentate**: operatori relaționali (`<`, `>=`, `==`),
  operatori logici (`||`), indexare logică, bucle `for/parfor`
- **Aspecte teoretice** ale tranziției de fază și ergodicității
- Secțiuni marcate cu `%%` și comentarii în **română**
- Secțiunea `%% CONCLUZII` la final

---

## Bibliografie

1. Ising, E. (1925). *Beitrag zur Theorie des Ferromagnetismus*. Zeitschrift für Physik, 31, 253–258.
2. Onsager, L. (1944). *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*. Physical Review, 65, 117.
3. Metropolis, N. et al. (1953). *Equation of State Calculations by Fast Computing Machines*. Journal of Chemical Physics, 21, 1087.
4. Newman, M.E.J. & Barkema, G.T. (1999). *Monte Carlo Methods in Statistical Physics*. Oxford University Press.
5. MathWorks Documentation – *Parallel Computing Toolbox: parfor*. https://www.mathworks.com/help/parallel-computing/parfor.html