# results

| Experiment | Model | CV Macro F1 | Test Macro F1 | observation |
| --- | --- | ---: | ---: | --- |
| 01 | Random Forest default | Not measured yet | ~0.51 | Better than balanced; weak R2L and U2R detection |
| 01 | Random Forest balanced | 0.9126 ± 0.0371 | 0.4707 | High CV score but poor test generalization; R2L/U2R still very weak |
| 02 | SMOTE + Random Forest | 0.9369 ± 0.0269 | 0.5497 | U2R improved significantly, R2L slightly improved but still weak |
| 03 | XGBoost | Not measured | 0.5562 | Better than Random Forest; improved R2L and U2R compared to baseline |
| 03 | SMOTE + XGBoost | 0.9437 ± 0.0262 | 0.6073 | Best result so far; SMOTE improved minority class detection, especially U2R |

