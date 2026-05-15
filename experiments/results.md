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

* 04_tuned_xgboost.py took around 15 minutes on colab
* 06_threshold_tuning_xgboost.pybest: validation configuration kept all multipliers at 1.0
