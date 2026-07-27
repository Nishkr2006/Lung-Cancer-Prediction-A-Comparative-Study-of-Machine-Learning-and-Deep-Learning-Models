# Lung-Cancer-Prediction-A-Comparative-Study-of-Machine-Learning-and-Deep-Learning-Models
A machine learning and deep learning project predicting lung cancer risk from patient survey data. Compares Logistic Regression, Random Forest, and XGBoost against MLP, TabNet, and FT-Transformer using a shared, leakage-safe pipeline, plus a lightweight web app demoing the best model.
## Methodology

The project follows a consistent, leakage-safe methodology across every model.

- A single canonical 70 percent, 15 percent, 15 percent stratified split is created once, using a fixed random seed of 42, and reused identically across all six models.
- The StandardScaler used for continuous features is fit only on the training data, in order to prevent data leakage.
- The decision threshold for each model is selected by sweeping the validation set only, with the goal of achieving a Recall of at least 0.95.
- Every model is evaluated using five fold cross validation to check the stability of its results.
- Statistical significance between models is assessed using bootstrap resampling with 1,000 resamples of the test set, applied to the ROC-AUC metric.

## Results

The table below shows test set performance for all six models.

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.971 | 0.521 | 0.678 | 0.933 | 0.904 |
| Random Forest | 0.954 | 0.627 | 0.756 | 0.928 | 0.902 |
| XGBoost | 0.958 | 0.552 | 0.700 | 0.926 | 0.891 |
| MLP | 0.954 | 0.565 | 0.710 | 0.926 | 0.876 |
| FT-Transformer | 0.958 to 0.964 | 0.526 to 0.555 | 0.679 to 0.704 | 0.926 to 0.927 | 0.876 to 0.877 |
| TabNet | 0.948 to 0.967 | 0.548 to 0.674 | 0.700 to 0.788 | 0.922 to 0.926 | 0.869 to 0.877 |

**Best classical machine learning model: Logistic Regression.** It achieved the highest Recall and the highest ROC-AUC of the three classical models tested.

**Best overall model: Logistic Regression.** A bootstrap comparison between Logistic Regression and the best deep learning model from each run, which was either FT-Transformer or TabNet depending on the run, found no statistically significant difference in ROC-AUC. This means that none of the three deep learning architectures tested provided a measurable improvement over a well tuned linear model on this dataset.

TabNet showed noticeably higher instability across cross validation folds, with a standard deviation of 0.100 compared to 0.005 to 0.010 for every other model. One fold even dropped as low as an ROC-AUC of 0.62. This occurred despite TabNet posting the strongest single split Precision and F1 scores of any model tested. This finding is discussed in more detail in the written report.

## Setup

To run this project locally, install the required packages and open the notebook.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Lung_Cancer_Unified_All_Models.ipynb
```

Please note that the notebook trains all three deep learning models from scratch and takes approximately 10 to 15 minutes to run from start to finish on a standard CPU. Most of this time is spent training the FT-Transformer model.

## Team
Nishtha, Yogesh, Hardik, Cici, Shivani, and Oshiya.

## Project Status

- [x] Classical machine learning pipeline complete, including Logistic Regression, Random Forest, and XGBoost.
- [x] Deep learning pipeline complete, including MLP, FT-Transformer, and TabNet.
- [x] Cross paradigm comparison complete, comparing the best machine learning model against the best deep learning model.
