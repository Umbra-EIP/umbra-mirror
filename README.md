# Umbra – EIP

## Description du projet

Ce projet vise à développer une intelligence artificielle capable de devenir une **copie motrice de l'individu**, fonctionnant **en tandem** avec lui. Grâce à l'utilisation de **technologies non-invasives** comme les **électromyogrammes (EMG)** et les **électroencéphalogrammes (EEG)**, l'IA peut déléguer le contrôle de **membres supplémentaires ou de remplacement**, ou assister dans le contrôle d'un **exosquelette**.

Cette approche ouvre des perspectives importantes :

- **Recherche et médecine** : lutte contre les maladies neuro-dégénératives et réhabilitation motrice.
- **Usage quotidien et sécurité** : augmentation des capacités et assistance dans la vie de tous les jours.

---

## Fonctionnement

Le projet repose sur deux modèles principaux :

1. **Modèle EEG → EMG**
   À partir des signaux EEG du cerveau, le modèle prédira les signaux EMG correspondants, capturant ainsi l'intention motrice de l'utilisateur.

2. **Modèle EMG → Mouvement**
   Ce modèle prend les signaux EMG et prédit les mouvements des membres ou de l'exosquelette (gestes de la main NinaPro). **Implémenté** : préprocessing, entraînement CNN-LSTM, dashboard Streamlit.

---

## Technologies utilisées

- **EEG** (Électroencéphalogrammes) pour capter l'activité cérébrale.
- **EMG** (Électromyogrammes) pour capter l'activité musculaire.
- **Machine Learning / Deep Learning** (TensorFlow/Keras) pour prédire et traduire les signaux en mouvements.
- **Python** comme langage principal de développement.

---

## Installation

```bash
git clone https://github.com/votre-utilisateur/umbra.git
cd umbra
pip install -r requirements.txt
```

Pour un environnement Conda :

```bash
conda env update --file environment.yml --name umbra-env
conda activate umbra-env
```

---

## Utilisation

Toutes les commandes sont à exécuter **à la racine du dépôt**.

### 1. Préprocessing (NinaPro → fenêtres EMG + labels)

Lit les données brutes dans `data/ninapro/` et enregistre un nouveau jeu dans `data/preprocessed/<id>/` (X.npy, y.npy) :

```bash
python -m src.main
```

### 2. Entraînement du modèle EMG → mouvement

Utilise un jeu préprocessé par son identifiant (par défaut `1`) :

```bash
python -m src.emg_movement.train --dataset 1
```

Options :

- `--dataset N` : utiliser `data/preprocessed/N/` (défaut : 1).
- `--output FICHIER.keras` : nom du modèle enregistré dans `src/models/` (défaut : `cnn_lstm_emg_v3.keras`).

### 3. Dashboard Streamlit

Visualisation et inférence sur les données préprocessées et les modèles entraînés :

```bash
streamlit run src/dashboard/app.py
```

Dans l’interface : choisir un dataset dans `data/preprocessed/`, charger un modèle depuis `src/models/`, puis lancer l’inférence.

---

## Structure du dépôt

- `src/config.py` : chemins et constantes (NinaPro, preprocessed, models).
- `src/emg_movement/` : préprocessing, modèle CNN-LSTM, entraînement, utils NinaPro.
- `src/eeg_emg/` : futur modèle EEG → EMG (placeholder).
- `src/dashboard/` : application Streamlit (EMG Hand Movement Decoder).
- `src/models/` : modèles Keras sauvegardés (.keras).
- `data/ninapro/` : données brutes NinaPro.
- `data/preprocessed/` : jeux préprocessés (X.npy, y.npy par sous-dossier).
- `tests/` : tests unitaires (environnement, modèle).
- `docs/` : documentation additionnelle (ex. beta_test_plan).

---

## Cas d'usage

- Assistance aux personnes atteintes de troubles moteurs ou neurodégénératifs.
- Contrôle d'exosquelettes pour la rééducation ou l'augmentation physique.
- Applications grand public dans le domaine de la sécurité ou de l'ergonomie.
