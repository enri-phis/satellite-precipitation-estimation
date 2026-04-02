# Pipeline ML per Classificazione dell'Intensita della Precipitazione da SEVIRI con Validazione IMERG

## 1. Titolo Progetto
Pipeline ML per Classificazione dell'Intensita della Precipitazione da SEVIRI (MSG) con Validazione Finale su IMERG.

## 2. Project Overview
Questo repository implementa una pipeline end-to-end di machine learning per classificare l'intensita della precipitazione a partire da osservazioni satellitari SEVIRI (MSG), con validazione esterna su prodotto IMERG.  
Il progetto integra preprocessing geospaziale, feature engineering, modellazione supervisionata e validazione quantitativa, con struttura modulare e ri-eseguibile per esperimenti su larga scala.

## 3. Problem Statement
Stimare correttamente l'intensita della precipitazione da dati satellitari e un problema rilevante per monitoraggio meteo, gestione del rischio idrologico e analisi climatica.  
I dati radiometrici satellitari sono informativi ma indiretti, rumorosi e fortemente sbilanciati (molti pixel asciutti, pochi eventi intensi).  
Questo progetto affronta il problema costruendo un workflow completo che:

- trasforma dati grezzi SEVIRI in predizioni multi-classe;
- riduce il bias dovuto allo sbilanciamento con bilanciamento stratificato;
- verifica la robustezza delle predizioni contro un riferimento indipendente (IMERG).

## 4. Methodology / Pipeline
La pipeline e organizzata in step modulari:

1. Preprocessing dati satellitari  
   Caricamento, conversione e filtro geografico delle osservazioni SEVIRI.

2. Generazione maschere  
   Costruzione di maschere giorno/notte, stagione e terra/mare.

3. Analisi distribuzionale  
   Istogrammi diagnostici e analisi esplorativa delle distribuzioni per canale e contesto.

4. Bilanciamento dataset multi-classe  
   Undersampling stratificato per configurazioni a 4, 5, 7 e 15 classi.

5. Feature engineering  
   Generazione di feature da canali grezzi, differenze, statistiche spaziali locali, gradienti, entropia e Laplaciano.

6. Training modelli  
   Addestramento Random Forest per classificazione multi-classe e regressione XGBoost per analisi dedicate.

7. Validazione finale contro IMERG  
   Confronto predizioni-riferimento con metriche standard (accuracy, precision, recall, F1, HSS) e artefatti diagnostici.

## 5. Tech Stack
Librerie e strumenti principali:

- Python
- NumPy
- Pandas
- SciPy
- scikit-learn
- XGBoost
- h5py
- xarray
- scikit-image
- Matplotlib
- Seaborn
- Cartopy
- PyProj
- Satpy
- ephem

## 6. Repository Structure
Script principali della pipeline:

- 1_data_loading_and_geographic_filtering.py  
  Caricamento dati grezzi e filtro geografico

- 2_masks_generation.py  
  Generazione maschere giorno/notte, stagionali, terra/mare

- 3_histograms_analysis.py  
  Analisi distribuzionale e istogrammi diagnostici

- 4_dataset_balancing.py  
  Bilanciamento dataset per configurazioni multi-classe

- 5.1_train_random_forest.py  
  Training Random Forest per classificazione multi-classe

- 5.2_train_xgboost_regression.py  
  Regressione XGBoost per stima continua per intervalli di intensita

- 5.3_group_class_mappings.py  
  Aggregazione classi 15 -> 7 -> 5 e report di valutazione

- 6_ml_feature_engineering.py  
  Costruzione feature avanzate e combinazioni di feature set

- 7_imerg_download_and_maps.py  
  Download/processing IMERG e mappe di supporto

- 8_1_validation_preprocessing.py  
  Preprocessing dedicato alla validazione finale

- 8_2_validation_features_model.py  
  Feature e inferenza per il flusso di validazione

- 8_3_validation_imerg.py  
  Confronto finale contro IMERG e metriche aggregate

Contenuti aggiuntivi:

- results/ con output sperimentali, metriche e figure
- notebook esplorativi e file storici di sviluppo

## 7. How to Run
Istruzioni base:

1. Creare e attivare un ambiente Python (consigliato virtual environment o Conda).
2. Installare le dipendenze principali indicate nella sezione Tech Stack.
3. Configurare path e flag RUN_... nella sezione Configuration di ogni script.
4. Eseguire gli script in sequenza.

Ordine consigliato:

1. 1_data_loading_and_geographic_filtering.py
2. 2_masks_generation.py
3. 3_histograms_analysis.py
4. 4_dataset_balancing.py
5. 5.1_train_random_forest.py
6. 5.2_train_xgboost_regression.py
7. 5.3_group_class_mappings.py
8. 6_ml_feature_engineering.py
9. 7_imerg_download_and_maps.py
10. 8_1_validation_preprocessing.py
11. 8_2_validation_features_model.py
12. 8_3_validation_imerg.py

## 8. Results
La pipeline produce:

- predizioni multi-classe dell'intensita di precipitazione;
- matrici di confusione e report di classificazione;
- metriche principali: accuracy, precision, recall, F1, HSS;
- grafici e mappe geospaziali per analisi diagnostica.

Nota: in questo README non vengono riportati valori numerici specifici se non associati a run riproducibili documentate negli output.

## 9. Key Features
- Pipeline completa dal dato satellitare grezzo alla validazione esterna.
- Supporto multi-configurazione classi (4, 5, 7, 15).
- Feature engineering orientato a segnali spaziali/geofisici.
- Struttura modulare che facilita debug, ri-esecuzione parziale e manutenzione.
- Validazione contro prodotto indipendente (IMERG), non solo metriche interne al training.

## 10. Limitations & Future Work
Limiti attuali:

- orchestrazione script-based (non ancora impacchettata come pipeline unica);
- sensibilita a qualita input, definizione classi e dominio geografico;
- dipendenze geospaziali con setup talvolta complesso in ambienti eterogenei.

Sviluppi futuri:

- tracking sistematico degli esperimenti e versionamento modelli;
- tuning iperparametri strutturato;
- stima incertezza e calibrazione probabilistica;
- estensione a piu anni e aree geografiche;
- packaging/containerizzazione per riproducibilita operativa.

## 11. Author
Enric Rossi  
Progetto di ambito machine learning e telerilevamento applicato alla classificazione della precipitazione da dati satellitari, con focus su robustezza del workflow, interpretabilita e validazione quantitativa.
