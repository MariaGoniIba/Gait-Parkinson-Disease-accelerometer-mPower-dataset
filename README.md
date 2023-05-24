# Gait in Parkinson's Disease using accelerometer data: mPower dataset
The aim of this project is to distinguish between Parkinson's disease (PD) subjects and healthy controls (HC) based on smartphone accelerometer readings while performing a gait task.

If you use this code, please consider citing our paper [Smartphone-Based Digital Biomarkers for Parkinson’s Disease in a Remotely-Administered Setting](https://ieeexplore.ieee.org/document/9727108). Please note, that this is a subset of the whole analysis and experiments covered in the paper.

## ABOUT DATASET
### Repository
Data used in this study were contributed by users of the Parkinson m-Power mobile application as part of the [m-Power study](https://www.synapse.org/#!Synapse:syn4993293/wiki/247859) developed by Sage Bionetworks and described in Synapse.

### Description of the data
mPower is a smartphone application-based study to monitor indicators of PD progression and diagnosis in a remote-setting. This dataset contains demographic and task-based data. During the walking task, participants are asked to walk back and forth for 20-30 seconds with their smartphone in the pocket. Participants were asked to complete these tasks several times so each participant can have multiple recordings. Both PD subjects and healthy participants contributed to this dataset. 

## METHODOLOGY
### Aims and framework
This dataset is strongly slanted towards young HC. Some recent studies applying machine learning to this type of high-dimensional data have suggested a good diagnostic sensitivity. However, studies eluded the importance of matching for age, a well-known confounding factor. In addition, some studies include several recordings per subject, which may potentially lead the classifier to detect the idiosyncrasies of each subject rather than specific PD related symptoms. These issues limit comparability across studies, hindering interpretation and obstructing translation to the clinic.

The following experiments were performed to understand the impact of age and the inclusion of several recordings per participant in the classification performance between PD and HC:
* Experiment 1: includes all participants and all recordings per participant.
* Experiment 2: includes all participants but just the first recording per participant. 
* Experiment 3: matches PD and HC based on age, and includes all recordings per participant.
* Experiment 4: matched PD and HC based on age and includes just the first recording per participant.

For experiment 3 or 4, here is the initial age and gender distribution before and after matching:

<p align="center">
  <b>Experiment 3</b>
</p>
<p align="center">
  <img width="900" src="https://github.com/MariaGoniIba/mPower/blob/main/Figures/Age%26Gender_exp3_before_after_matching.png" alt="Image">
</p>

<p align="center">
  <b>Experiment 4</b>
</p>
<p align="center">
  <img width="900" src="https://github.com/MariaGoniIba/mPower/blob/main/Figures/Age%26Gender_exp4_before_after_matching.png" alt="Image">
</p>
  


### Code
* *Gait.py*: main script. It imports and preprocess the demographic (function *preproc_demog*) and walking data (function *demog_gait*). For each recording the raw accelerometer data is transformed into linear acceleration (function *linearacceleration*) and a set of features are extracted (function *features*). Then, classification for each experiment is performed using a Random Forest (RF) classifier (function *RF*).
* *preproc.py*: script with all functions needed for data preprocessing, including the following:
  * *preproc_demog*: in the beginning of the task, participants had to select ‘‘true’’ or ‘‘false’’ to the question ‘‘Have you been diagnosed by a medical professional with Parkinson Disease?’’. According to this answer, they were classified as PD or HC. Subjects who did not answer this question were discarded from further analysis. Those who did not provide information on age were also discarded. 
  * *demog_gait*: participants with inconsistencies in their clinical data (e.g. self-reported healthy controls who answered questions about PD diagnosis or PD medication) were discarded.
  * *onerecording*: this function selected the first recording for each participant.
  * *matching*: this function matches the data based on age. 
  * *LPfilter*: Butterworth low pass filter 4th order cutoff freq 20 Hz.
  * *HPfilter*: Butterworth high pass filter 3th order cutoff freq 0.3 Hz.
  * *linearacceleration*: this function computes the linear acceleration data based on the attitude of the phone, the user acceleration and the gravity using a rotation matrix. You can found a detailed description of this method and the data provided in [this repository](https://github.com/MariaGoniIba/Linear-acceleration-from-accelerometer-smartphone), where I use one recording from this dataset to show how this process works.
* *featGait.py*: script that extracts a set of 65 features, including mean stride interval from the step series, statistical and frequencial features for the acceleration in the 3 axes and magnitude acceleration signal. For further information about these features, please refer to [our paper](https://ieeexplore.ieee.org/document/9727108).
* *classifierRF.py*: Random Forest classifier with a nested cross-validation for hyparameter tuning and model evaluation. This function calculates performance metrics (accuracy, sensitivity, specificity, ROC curve, AUC) and reports the feature importance during the classification.

## RESULTS
### Classification metrics

| Experiment | PD/HC | Accuracy (%) | Sensitivity (%) | Specificity (%) | AUC (%) |
| ------------- |:-------------:| -----:|-----:|-----:|-----:|
| All records & no matched      | 19739/14900 | 76 | 86 | 64 |85 |
| One record & no matched      | 511/1860 | 79 | 8 | 99 | 68 |
| All records & matched      | 5541/5541 | 77 | 75 | 79 | 85 |
| One record & matched      | 324/324 | 60 | 51 | 70 | 60 |


### ROC curves
<p align="center">
  <img width="800" src="https://github.com/MariaGoniIba/mPower/blob/main/Figures/ROC_all_experiments.png"
</p>

### Feature importance
  <p align="center">
  <img width="1200" src="https://github.com/MariaGoniIba/mPower/blob/main/Figures/feat_all_experiments.png"
</p>

## CONCLUSION
* Prediction performances considerably decreased when choosing one recording per participant and matching for age, indicating the importance of controlling for such confounds in DB data.
    Using only one recording per subject prevents the classifier from detecting the idiosyncrasies of each subject rather than specific PD related symptoms. 
    Single measures are likely to contain more noise due to higher variation in task administration as well as in individual performance in a poorly-controlled setting. 
    Using multiple time points may therefore further increase the discrimination between PD and HC.
* Although accuracies are similar between all recordings and one recording experiment without matching for age, specificity values are exceedingly high whereas sensitivity values are vastly low when choosing one recording. 
    This indicates a greater prediction ability for the HC group, which is considerably larger than the PD group. 
    
These findings indicate the importance of accounting for confounds in PD digital biomarker data, which otherwise may lead to over-optimistic results. 
Such effects may also explain the high accuracies in some of the previous studies using mPower dataset, where no proper matching for these confounds was performed.

## CITATION
* [Smartphone-Based Digital Biomarkers for Parkinson’s Disease in a Remotely-Administered Setting](https://ieeexplore.ieee.org/document/9727108)
* [The mPower study, Parkinson disease mobile data collected using ResearchKit](https://www.nature.com/articles/sdata201611)
* [Smartphones as new tools in the management and understanding of Parkinson’s disease](https://www.nature.com/articles/npjparkd20166)


