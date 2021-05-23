# Open Data Exploratory Analysis

Corresponding GitHub repository for the paper: Insert title


Maintained by: dadler@infosci.cornell.edu (GitHub ID: dadler6)

## Assessing the Potential for an Open Data Community

### Task Motivation and Research Questions
For this work, we not only wished to survey literature and identify challenges we hypothesized open PS4MH data could solve. We also wished to empirically assess this hypothesis. We designed an experiment to answer the following research questions (RQs), each corresponding to one of the 4 challenges:

[RQ1: Generalizability Across Data and Studies] Does a model trained with one dataset generalize to another dataset?

[RQ2: Human Behavior Changes over Time] Do modeled sensor feature-mental health relationships created using data collected from one time period, across one age group, apply to data collected in another time period from another age group?

[RQ3: Individual-Level Heterogeneity] How does individual-level heterogeneity impact model performance across two datasets?

[RQ4: Device Uncertainty] Will models created using data collected from one set of devices apply to data collected using a different set of devices?

We used 2 publicly available [7–11,18,24,41,42] PS4MH datasets to design our feasibility analysis. To the best of our knowledge, these are the only open PS4MH datasets. Both datasets contained sensor data and self-reported mental health symptom EMAs, and the differences between datasets (see Table 2) highlighted all 4 challenges described in the prior section. The datasets were thus ideal to answer the 4 RQs. 

Our task explored if an EMA prediction model trained using one dataset could accurately predict EMAs in the other dataset. If the model was generalizable (RQ1) between datasets despite different data collection periods (RQ2), population heterogeneity (RQ3), and data collection devices (RQ4), there is promise in open PS4MH data. In machine learning, applying a model trained with a source dataset to a different target dataset is called transfer learning [43]. Transfer learning is successful if a model trained with the source achieves similar or greater performance when applied to the target, compared to using a model trained exclusively on the target. Negative transfer occurs when model performance does not improve performance on the target, but instead diminishes performance, typically due to differences between the datasets. If negative transfer occurs, only a portion of the source dataset may be useful for transfer learning [44].

#### Dataset Overview
The CrossCheck and StudentLife datasets are 2 open (see [45,46]) PS4MH datasets that contain smartphone sensing and mental health symptoms collected from individuals. Please refer to the Appendix for an overview of the data collection procedures, and [7,18,19] for more details regarding the studies that collected this data. These publications extensively document the participant recruitment, enrollment criteria, and data collection system development, which were not a part of this work. In our task, the larger CrossCheck dataset was designated as the source dataset, and the smaller StudentLife dataset was the target dataset. The CrossCheck dataset was determined to be “larger”, because data was collected for more participants over a longer period of time compared to the StudentLife dataset. See Table 2 for a comparison of these two datasets.

#### Prediction Model Training and Assessment
We used leave-one-subject-out cross-validation (LOSO-CV) to assess transfer learning. LOSO-CV is a common procedure to estimate the generalizability of PS4MH prediction models [7,9,36]. In LOSO-CV, each target participant is removed from the data, and the remaining data is used for model training. The trained model predicts EMA scores for the held out participant. The predictions across each held-out participant are concatenated to estimate model performance. Lasso regression models were used for prediction. For each held-out participant, we trained a lasso regression model using the remaining source and target data. Since negative transfer may occur, we also developed a second LOSO-CV procedure where a lasso regression model was trained using each participant’s data (“subpopulation models”). We then matched each held-out participant to the model trained using a different participant’s data that maximized the coefficient of determination (R2). Model performance across participants was measured using the R2 and mean absolute error (MAE). EMA values were non-normally distributed (Shapiro-Wilk  P<.001). We performed Wilcoxon signed-rank tests hypothesizing that the squared errors (SE) between the predicted and actual EMA values were significantly greater using the target compared to the source as training data. Model parameters and prediction procedures are detailed in the Appendix “Sensor-EMA Alignment” and “Transfer Learning Cross-Validation” sections. 

### Task Results

#### Transferable Sensing and EMA Data
Table 3 describes the aligned data elements, across the 2 datasets. We qualitatively identified EMAs that aligned between each dataset based upon question and response structure. We then required a minimum number of data points per-participant [7,8]. Similar to previous work [7], we required ≥30 EMA values per source and target participant, which was about 50% of the intended target EMA collection. Table 4 shows the resulting aligned dataset. We list characteristics separately for each EMA we predicted in the target population because not all individuals who responded to ≥30 sleep EMAs also responded to ≥30 stress EMAs. Specific feature and EMA distribution alignment statistics can be found in the Appendix “Sensing Distribution Alignment” and “EMA Distribution Alignment” sections. EMA values were non-normally distributed (omnibus test for normality sleep=631 P<.001 and stress=633, P<.001) [47]. Mann-Whitney U tests showed that the EMA distributions were significantly different (α=.05) between source and target sleep (U=1,811,706, P=.002) and stress (U=542,292, P<.001). We calculated Cureton’s matched-pairs rank-biserial correlation (RBC∊[−1, 1]) [48] to show the magnitude of differences for the sleep (RBC=-0.07) and stress (RBC=0.59) EMAs.


#### LOVO-CV Results Using Entire Population Models
Table 5 describes the LOSO-CV results using prediction models trained with the entire source versus target data. All models were not predictive (highest performing models R2≈0). SE values were found to be non-normally distributed (Shapiro-Wilk  P<.001). We found the sleep EMA prediction SE was marginally greater (α=.10) using the target data (W=96,087, P=.052) and differences were small (RBC=0.08). The predicted stress EMA SE was significantly lower (W=7,179, P=1.000, RBC=−0.70) using the target data.

| | R<sup>2</sup> | | MAE | | W | _P_ | RBC |
|-|-------|-|-----|-|---|-----|-----|
|Training Data| Target | Source | Target | Source |  | |
| Sleep | -0.01 | 0.00 | 0.66 | 0.66 | 96,087 | 0.052 | 0.08 |
| Stress | -0.02 | -1.95 | 0.57 | 1.05 | 7,179 | 1.000 | -0.70 |

<br>

#### LOVO-CV Results Using Subpopulation Models
Table 6 describes the LOSO-CV results using prediction models trained with source versus target subpopulations. Source subpopulation models achieved the highest performance with R2=0.18, MAE=0.57 for the sleep and R2=0.22, MAE=0.48 for the stress EMA. Training with source data significantly (α=.05) decreased the SE compared to training with target for the sleep EMA (W=103,373, P<.001, RBC=0.16) and marginally (α=.10) decreased the stress (W=26,096, P=.067, RBC=0.10) EMA SE.

| | R<sup>2</sup> | | MAE | | W | _P_ | RBC |
|-|-------|-|-----|-|---|-----|-----|
|Training Data| Target | Source | Target | Source |  | |
| Sleep | 0.11 | 0.18 | 0.62 | 0.57 | 103,373 | <0.001 | 0.16 |
| Stress | 0.18 | 0.22 | 0.49 | 0.48 | 26,096 | 0.057 | 0.10 |

<br>

## Repository Information

### Data

The cleaned data used for the lasso regression analysis can be found in the `data` directory. The cleaned datasets are the CrossCheck and StudentLife datasets, with epoch features computed, and these features are averaged across 3 days to align with the EMAs. The cleaned data is called:

* Cleaned CrossCheck: `data/crosscheck_daily_data_cleaned_w_sameday.csv`
* Cleaned StudentLife: `data/studentlife_daily_data_cleaned_w_sameday_03192020.csv`

These files are used in the `notebooks/lasso_regression.ipynb` notebook.

Due to filesize, we are not including the original public datasets in this repository, but they are needed if you wish to run the cleaning code (`notebooks/data_cleaning.ipynb`). Please download them at the following links:

* Raw CrossCheck Data is the `CrossCheck_Daily_Data.csv` on Box: https://cornell.box.com/s/rkx46bgv36lkmo2eu349ka95senn48gh
* Raw StudentLife Data must be downloaded at unzipped from: https://studentlife.cs.dartmouth.edu/dataset.html

### Code

The code to run the Lasso Regression models can be found in the following notebook:

* `notebooks/lasso_regression.ipynb`

The code to clean the original public datasets can be found in:

* `notebooks/data_cleaning.ipynb`

#### Utility code

The following `.py` files are used throughout the lasso regression and data cleaning notebooks. They are:

* `src/util.py`
* `src/cleaning_util.py`
* `src/regression_cv.py`


