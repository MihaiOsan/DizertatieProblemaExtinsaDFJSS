# **Planificarea Dinamică a Evenimentelor folosind Programarea Genetică**

Acest proiect reprezintă o implementare în Python a unei soluții bazate pe Programare Genetică (GP) pentru rezolvarea Problemei Planificării Flexibile și Dinamice în Ateliere de tip Job Shop (Dynamic Flexible Job Shop Scheduling Problem \- DFJSSP). Sistemul generează automat reguli de dispecerizare (dispatching rules) adaptate la condiții în schimbare (ex: apariția de noi joburi, defectarea mașinilor), având ca scop principal optimizarea performanței atelierului (minimizarea timpului total de finalizare \- makespan).

Demo: https://youtu.be/jqSRslWqF-o

## **Concepte Cheie**

* **DJSSP (Dynamic Job Shop Scheduling Problem):** O problemă de optimizare NP-hard în care un set de joburi trebuie alocate dinamic pe un set de mașini, în condiții de incertitudine (evenimente neprevăzute).  
* **Programare Genetică (GP):** O tehnică de calcul evolutiv care generează programe de calculator sau formule (în acest caz, reguli de planificare) ca soluții la o problemă.  
* **Reguli de Dispecerizare (Dispatching Rules):** Euristici simple sau complexe folosite pentru a decide ce operație să fie următoarea executată pe o mașină liberă. Exemplu: SPT (Shortest Processing Time), FIFO (First-In, First-Out).

## **Arhitectura Sistemului**

Sistemul este construit pe o arhitectură modulară, reflectată în structura codului, pentru a facilita dezvoltarea, testarea și extinderea:

* ⚙️ **Modul GP Engine (gpSetup.py):** Nucleul algoritului de Programare Genetică. Gestionează populația de reguli, configurează setul de funcții și terminale și aplică operatorii genetici (selecție, crossover, mutație). Utilizează biblioteca **DEAP**.  
* 🏭 **Modul Simulator (scheduler.py):** Inima sistemului, responsabil cu simularea funcționării atelierului. Implementează un simulator de evenimente discrete care procesează evenimente (sosiri joburi, finalizări operații, etc.) și aplică regulile de dispecerizare pentru a lua decizii.  
* 📊 **Modul Evaluator (evaluator.py):** Calculează funcția de fitness pentru fiecare regulă. O regulă este evaluată prin invocarea scheduler.py pentru a rula o simulare completă și a măsura performanța (ex: makespan).  
* 📂 **Modul Data (data\_reader.py, generare.py):** Se ocupă de I/O. data\_reader.py încarcă instanțele problemei din dynamic\_data/. generare.py poate fi folosit pentru a crea noi seturi de date.  
* 📈 **Modul de Rezultate și Vizualizare (rezultate/, ganttPlot.py):** Colectează și salvează rezultatele experimentelor (statistici, reguli) în directorul rezultate/. ganttPlot.py oferă funcționalități pentru a vizualiza planificările sub formă de diagrame Gantt.  
* 🔧 **Utilitare (utils.py, individual\_from\_string.py, simpleTree.py):** Conține funcții ajutătoare folosite în diverse module. individual\_from\_string.py permite reconstrucția unei reguli (individ) dintr-o reprezentare textuală. simpleTree.py simplifică arborele returnat de DEAP.

## **Structura Fișierelor**
```bash
.  
├── main.py                       \# Script principal: rulează algoritmul evolutiv GP.  
├── clasic\_methods.py            \# Script pentru benchmark pe reguli clasice (SPT, FIFO).  
├── scheduler.py                  \# Modulul de simulare a atelierului (motorul principal).  
├── gpSetup.py                    \# Configurare mediu GP (funcții, terminale, operatori).  
├── evaluator.py                  \# Logica de evaluare a fitness-ului indivizilor.  
├── data\_reader.py               \# Citirea datelor de intrare (instanțe .json).  
├── generare.py                   \# Generare de noi seturi de date pentru experimente.  
├── simpleTree.py                 \# Definirea structurii de arbore pentru regulile GP.  
├── ganttPlot.py                  \# Creare diagrame Gantt pentru vizualizarea planificării.  
├── individual\_from\_string.py   \# Reconstruiește o regulă GP dintr-un string.  
├── utils.py                      \# Funcții utilitare comune.  
|  
├── dynamic\_data/                \# Datele de intrare (instanțele problemei).  
│   ├── extended/  
│   └── fan21/  
|  
├── fan\_comp/                   \# Modul pentru teste comparative cu reguli din literatura de specialitate.  
│   ├── fan\_rules.py            \# Scriptul de test comparativ.  
│   ├── generare\_fan21.py       \# Generator specific pentru instanțele Fan21.  
│   └── ...  
|  
├── rezultate/                   \# Directorul principal pentru toate rezultatele generate.  
│   ├── genetic/  
│   ├── dinamic/  
│   ├── fan/  
│   └── ...  
|  
└── teste/                       \# Suită de teste unitare pentru modulele cheie.  
    ├── test\_gp\_setup.py  
    ├── test\_reader.py  
    ├── test\_simple\_tree.py  
    └── ...
```
## **Instalare**

1. **Clonează repository-ul:**
  ```bash
   git clone MihaiOsan/DizertatieProblemaExtinsaDFJSS
```

2. **Instalează dependențele:** Proiectul se bazează pe câteva biblioteci cheie. Creează un fișier requirements.txt cu următorul conținut:
   ```bash
   deap
   matplotlib
   numpy
   sympy 
   ```

   Apoi rulează comanda de instalare:
   ```bash
   pip install \-r requirements.txt
   ```

## **Utilizare**

Proiectul poate fi rulat în mai multe moduri, în funcție de obiectivul urmărit.

### **1\. (Opțional) Generarea Seturilor de Date**

Dacă dorești să creezi noi instanțe de test, poți rula scriptul generare.py. Poți modifica parametrii de generare (număr de joburi, mașini, etc.) direct în interiorul fișierului.  
python generare.py

### **2\. Rularea Algoritmului Evolutiv (GP)**

Acesta este scriptul principal al proiectului. El încarcă o instanță, rulează procesul evolutiv pentru a descoperi o regulă de dispecerizare optimă și salvează rezultatele în directorul rezultate/.  
python main.py

Poți configura fișierele de intrare/ieșire și parametrii algoritmului (dimensiunea populației, numărul de generații etc.) în secțiunea de configurare a scriptului main.py.

### **3\. Rularea Testelor cu Metode Clasice**

Pentru a obține un baseline de performanță, poți evalua regulile de dispecerizare standard (FIFO, SPT, LPT etc.) pe un set de instanțe.  
python clasic\_methods.py

### **4\. Rularea Testelor Comparative (inclusiv reguli Fan)**

Acest script oferă o analiză comparativă mai complexă, incluzând atât regulile clasice, cât și reguli mai avansate propuse în literatura de specialitate, pe aceleași seturi de date.  
python fan\_comp/fan\_rules.py

**Autor:** Oșan Mihai
