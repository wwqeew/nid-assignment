# results

| Experiment | Model | CV Macro F1 | Test Macro F1 | observation |
| --- | --- | ---: | ---: | --- |
| 01 | Random Forest default | Not measured yet | ~0.51 | Better than balanced; weak R2L and U2R detection |
| 01 | Random Forest balanced | 0.9126 ± 0.0371 | 0.4707 | High CV score but poor test generalization; R2L/U2R still very weak |
| 02 | SMOTE + Random Forest | 0.9369 ± 0.0269 | 0.5497 | U2R improved significantly, R2L slightly improved but still weak |
| 03 | XGBoost | Not measured | 0.5562 | Better than Random Forest; improved R2L and U2R compared to baseline |
| 03 | SMOTE + XGBoost | 0.9437 ± 0.0262 | 0.6073 | SMOTE improved minority class detection, especially U2R |
| 04 | Tuned SMOTE + XGBoost | 0.9412 ± 0.0297 | 0.6086 | Best result so far; small improvement over previous SMOTE + XGBoost, especially on R2L/U2R |
| 05 | LightGBM | Not measured | 0.5444 | Faster but weaker than XGBoost |
| 05 | SMOTE + LightGBM | 0.9296 ± 0.0283 | 0.5922 | Improved minority class detection, especially U2R, but did not beat tuned SMOTE + XGBoost |
| 06 | Threshold tuning + SMOTE + XGBoost | 0.9408 ± 0.0217 | 0.6086 | Threshold tuning did not improve the model; best multipliers were all 1.0 |
| 07 | XGBoost + Feature Engineering | Not measured | 0.5730 | Feature engineering improved plain XGBoost compared to earlier baseline, did not beat SMOTE-based models |
| 07 | SMOTE + XGBoost + Feature Engineering | 0.9480 ± 0.0269 | 0.6051 | Very close to the best result; U2R improved to 0.40 |
| 08 | Tuned SMOTE + XGBoost + Feature Engineering | 0.9454 ± 0.0234 | 0.6078 | Very close to the best result; U2R improved to 0.44, but Probe performance decreased, overall macro F1 did not improve |
| 10 | Weighted XGBoost | 0.9506 ± 0.0204 | 0.5968 | Strong result without SMOTE; reducing Normal weight and increasing R2L/U2R weights improved minority detection, but did not beat tuned SMOTE + XGBoost |
| 11 | Threshold tuning + SMOTE + LightGBM | 0.9295 ± 0.0283 | 0.6021 | Threshold tuning improved LightGBM by boosting U2R probability; U2R F1 increased to 0.45, but R2L remained weak |


## Planned experiments

| No. | File | Model / Approach | Purpose |
| --- | --- | --- | --- |
| 09 | `09_smoteenn_xgboost.py` | SMOTEENN + Tuned XGBoost | Try combined oversampling and cleaning to improve R2L vs Normal separation |
| 10 | `10_weighted_xgboost.py` | Cost-sensitive XGBoost with `sample_weight` | Penalize mistakes on R2L and U2R more strongly without relying only on SMOTE |
| 11 | `11_threshold_tuning_lightgbm.py` | Threshold / probability tuning for LightGBM | Check whether LightGBM improves after R2L/U2R probability adjustment |
| 12 | `12_mlp_neural_network.py` | SMOTE + MLP neural network | Test an actual neural network model for comparison |
| 13 | `13_voting_ensemble.py` | Voting ensemble of XGBoost, LightGBM, Random Forest | Combine strongest models and test whether ensemble improves macro F1 |
| 14 | `14_feature_selection_xgboost.py` | Feature selection + Tuned SMOTE XGBoost | Remove noisy/redundant features and test whether generalization improves |
| 15 | `15_svm.py` | Linear SVM with scaling and class weights | Test SVM as an alternative classifier for minority classes |
| 16 | `16_knn.py` | k-NN with scaling and SMOTE | Test distance-based classification as an additional comparison |

## Current best model

| Model | CV Macro F1 | Test Macro F1 | Notes |
| --- | ---: | ---: | --- |
| Tuned SMOTE + XGBoost | 0.9412 ± 0.0297 | 0.6086 | Best result so far |


* 04_tuned_xgboost.py took around 15 minutes on colab
* 06_threshold_tuning_xgboost.pybest: validation configuration kept all multipliers at 1.0
* SMOTE + XGBoost + Feature Engineering was tested with more "agressive" XGBClassifier parameters and SMOTE R2l = 16000 (requires check with old parameters) 
* Feature engineering improved U2R detection but did not improve overall macro F1.
* Weighted XGBoost CV score was even higher, than best model result, however, CV score alone is not enough for NSL-KDD, test result was worse.
