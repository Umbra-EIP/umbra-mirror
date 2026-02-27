# Umbra – EIP

## Description du projet
Ce projet vise à développer une intelligence artificielle capable de devenir une **copie motrice de l’individu**, fonctionnant **en tandem** avec lui. Grâce à l’utilisation de **technologies non-invasives** comme les **électromyogrammes (EMG)** et les **électroencéphalogrammes (EEG)**, l’IA peut déléguer le contrôle de **membres supplémentaires ou de remplacement**, ou assister dans le contrôle d’un **exosquelette**.

Cette approche ouvre des perspectives importantes :
- **Recherche et médecine** : lutte contre les maladies neuro-dégénératives et réhabilitation motrice.
- **Usage quotidien et sécurité** : augmentation des capacités et assistance dans la vie de tous les jours.

---

## Fonctionnement

Le projet repose sur deux modèles principaux :

1. **Modèle EEG → EMG**  
   À partir des signaux EEG du cerveau, le modèle prédit les signaux EMG correspondants, capturant ainsi l’intention motrice de l’utilisateur.

2. **Modèle EMG → Mouvement**  
   Ce modèle prend les signaux EMG et prédit les mouvements des membres ou de l’exosquelette, permettant à l’IA de reproduire ou assister les actions de l’utilisateur en temps réel.

---

## Technologies utilisées
- **EEG** (Électroencéphalogrammes) pour capter l’activité cérébrale.  
- **EMG** (Électromyogrammes) pour capter l’activité musculaire.  
- **Machine Learning / Deep Learning** pour prédire et traduire les signaux en mouvements.  
- **Python** comme langage principal de développement.  

---

## Cas d’usage
- Assistance aux personnes atteintes de troubles moteurs ou neurodégénératifs.  
- Contrôle d’exosquelettes pour la rééducation ou l’augmentation physique.  
- Applications grand public dans le domaine de la sécurité ou de l’ergonomie.  

---

## Installation
```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/projet-ia-motrice.git

# Installer les dépendances
pip install -r requirements.txt
