### **BETA TEST PLAN – UMBRA PROJECT**

## **1. Project context**
Umbra is a project that aims to create a neuromuscular copy of a patient in order to give it the control of exterior tools to assist said patient in his everyday life.

## **2. User role**
[The following roles will be involved in beta testing]

| **Role Name**  | **Description** |
|--------|----------------------|
| Supervisor       | Person that supervise and assist the patient with the data retreival process and the evolution of the copy during the whole process |
| User       | Person that serve as the main reference in the making of the copy |

---

## **3. Feature table**
[The following features will be shown during the defense]

| **Feature ID** | **User role** | **Feature name** | **Short description** |
|--------------|---------------|-------------------------|--------------------------------------|
| F1 | Supervisor | Minimal dashboard to visualize input/output | Simple app to visualise the EEG and EMG data flow in the retreival process, allowing the supervisor to pinpoint anomalies and fix the process before more mistakes are made |
| F2 | Everyone | Model prototype | A basic version of the model capable of being trained with a fake/real dataset and be used in the pradctical 3D test |
| F3 | Supervisor | Preprocessing pipeline | Contain the whole process made by the cyton + daisy board to process the data in the retreival step to later build the dataset accordingly |
| F4 | Everyone | Practical 3D test | Blender script connected to the model and a data injector/the User directly to visualize the model working in real time |
| F5 | Supervisor | Model hardware impact tracker | Simple app to track the model hardware consumtion, allowing the supervisor to work on an appropriate optimization |
| F6 | Everyone | Dataset quality checker | Simple app to check the dataset is correctly divided between training, evaluation, bad data and that is has enough data to train the model correctly |
| F7 | Supervisor | Model comparator tool | Simple app to compare model iterations and performance allowing the supervisor to enforce fine tuning on the best one |
| F8 | Everyone | Automated setup script | Scripted actions to fully setup the EEG/EMG setup for the user and the model analysis system for the supervisor |

---

## **4. Success Criteria**
[Define the metrics and conditions that determine if the beta version is successful.]

| **Feature ID** | **Key success criteria** | **Indicator/metric** | **Result** |
|--------------|---------------------------------------|-----------------------|----------------|
| F1 | Injection of fake data into the app with expected readings | 10 attempt without any failure related to data visualisation | acheived 10/10 |
| F2 | Model evaluation pipeline follows expectation with normal and anormal data | Non-chaotic model behavior | Partially Acheived |
| F3 | injecting typical fake data of what is expected of the board at each step of the process and getting expected results | Every step of the pipeline work as intended | Acheived |
| F4 | The 3D prop is correctly rigged and is moving in sync with what the model is outputting | 10 attempt with different movement inputs without failure related to movement | Partially acheived |
| F5 | The analysis behave as expected and warns at the right time if something is wrong | 10 attempt without any failure related to impact analysis | acheived 10/10 |
| F6 | The checker is strict even if the dataset is almost usable | 10 attempt with poorly built datasets and 10 attempt with correctly built datasets without failure related to dataset segmentation analysis | 20/20 Acheived |
| F7 | Broken models are identified and normal ones are ranked correctly by choosen criteria | 10 attempt with normal models and 10 attempt with broken models without failure related to model comparaison and error identification | 20/20 acheived |
| F8 | The setup is usable smoothly by the User and the Supervisor | 10 attempt with the User setup and 10 attempt with the Supervisor setup without any failure related to tools usage post-setup | (User) Partially acheived (supervisor) 10/10 acheived |