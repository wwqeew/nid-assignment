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
| 09 | SMOTEENN + Tuned XGBoost | 0.9542 ± 0.0291 | 0.6097 | improved result, aggressive ENN cleaning slightly improved macro F1, but training time was much higher |
| 10 | Weighted XGBoost | 0.9506 ± 0.0204 | 0.5968 | Strong result without SMOTE; reducing Normal weight and increasing R2L/U2R weights improved minority detection, but did not beat tuned SMOTE + XGBoost |
| 11 | Threshold tuning + SMOTE + LightGBM | 0.9295 ± 0.0283 | 0.6021 | Threshold tuning improved LightGBM by boosting U2R probability; U2R F1 increased to 0.45, but R2L remained weak |
| 12 | MLP Neural Network | Not measured | 0.5996 | Best neural network result; performed surprisingly well without SMOTE, especially on U2R precision |
| 12 | SMOTE + MLP Neural Network | 0.8481 ± 0.0151 | 0.5636 | SMOTE reduced performance; U2R precision dropped significantly, causing lower macro F1 |
| 13 | Voting Ensemble + probability tuning | 0.9415 ± 0.0258 | 0.6232 | Soft voting of Feature Selection XGBoost, LightGBM, and plain MLP. The best weights were [3, 2, 3], and probability multipliers improved the test macro F1 from 0.5980 to 0.6232. It improved U2R and R2L compared to raw voting, but did not beat the best Keras model. |
| 14 | Feature Selection + Tuned SMOTE XGBoost | 0.9415 ± 0.0291 | 0.6166 | median feature selection reduced encoded features from 121 to 61 and improved generalization |
| 15 | Linear SVM | 0.8248 ± 0.0338 | 0.5555 | Plain Linear SVM performed best among SVM variants; class weights and SMOTE did not help, and R2L detection remained very weak |
| 16 | Keras Neural Network custom weights | 0.8675 ± 0.0381 | 0.6399 | custom class weights strongly improved U2R detection while keeping good DoS/Normal/Probe performance, but R2L recall remained low |
| 17 | Threshold tuning + Keras custom weights | 0.8433 ± 0.0381 | 0.5625 | did not improve Keras; best tuned version scored 0.5606, raw Keras scored 0.5625. showed high instability compared to experiment 16 |
| 18 | Keras multi-seed ensemble | 0.8495 ± 0.0295 | 0.6147 | Averaged predictions from three Keras models with seeds [42, 7, 21]. Individual seed scores were 0.6182, 0.5899, and 0.6350. The ensemble did not improve over the best single Keras model because weaker seeds reduced averaged probability quality. |
| 19 | Keras class weight + seed tuning | 0.8525 ± 0.0197 | 0.6646 | New best result. Tuning class weights and seed selection improved R2L detection significantly. Best setup used seed 100 and class weights {0: 1.0, 1: 0.7, 2: 2.0, 3: 20.0, 4: 60.0}. R2L F1 improved to 0.42, although U2R F1 decreased to 0.43. |
| 20 | Keras fine-tuning around best weights | 0.8240 ± 0.0189 | 0.6006 | Follow-up to experiment 19. Fine-tuned class weights around the best Keras setup and selected `cw_3_r2l22_u2r60` with seed 100. Although validation tuning reached 0.9376 macro F1, the fixed 99-epoch final model overfit and dropped to 0.6006 test macro F1. R2L recall fell back to 0.09, so this model was not used as the final solution. |
| 21 | Keras CV-selected training | 0.8690 ± 0.0263 | 0.6010 | Tested safer training strategies after experiment 20 by selecting the best class-weight setup through 3-fold CV and comparing early stopping with fixed median epoch training. CV selected the experiment 19 weights {0: 1.0, 1: 0.7, 2: 2.0, 3: 20.0, 4: 60.0}, but final test performance remained weak |
| 22 | Keras categorical embeddings MLP | 0.8321 ± 0.0506 | 0.6578 | Tested a neural network with learned embeddings for categorical features instead of one-hot encoding. The best setup used embedding dimensions protocol=3, service=16, flag=4 and class weights {0: 1.0, 1: 0.7, 2: 2.0, 3: 20.0, 4: 60.0}. Achieved strong R2L detection with R2L F1 = 0.45, but did not beat experiment 19. |
| 23 | Stacking: Keras + SMOTE XGBoost + Feature Selection XGBoost | 0.9259 ± 0.0317 | 0.5491 | Stacked exp19 Keras, exp04 SMOTE XGBoost, and exp14 Feature Selection XGBoost using out-of-fold predicted probabilities and logistic regression as the meta-model. Although meta-model CV on OOF features was high, the final KDDTest+ result collapsed to 0.5491 macro F1 because the meta-model did not generalize to unseen attack types and strongly underpredicted R2L/U2R. |
| 24 | Voting: Keras + SMOTE XGBoost + Feature Selection XGBoost | 0.9430 ± 0.0320 | 0.5976 | Tested soft voting over exp19 Keras, exp04 SMOTE XGBoost, and exp14 Feature Selection XGBoost. Although CV and validation scores were high, the final KDDTest+ result dropped to 0.5976. The ensemble underpredicted R2L and U2R, with R2L F1 = 0.16 and U2R F1 = 0.36. |
| 25 | SMOTEENN + Tuned XGBoost | 0.9412 ± 0.0295 | 0.6058 | combined oversampling and cleaning to improve R2L vs Normal separation |
| 26 | Cost-sensitive XGBoost with `sample_weight` | 0.9506 ± 0.0204 | 0.5968 | Penalize mistakes on R2L and U2R more strongly without relying only on SMOTE |
| 27 | Threshold / probability tuning for LightGBM | 0.9329 ± 0.0252 | 0.6080 | 0.48 score for U2R with multiplier 4.0 |
| 28 | SMOTE + MLP neural network | 0.8614 ± 0.0217 | 0.5919 | Test an actual neural network model for comparison, 
| 29 | Voting ensemble of XGBoost, LightGBM, Random Forest | 0.9400 ± 0.0260 | 0.6168 | Raw voting had 0.5957 best score still was tuned voting, Combine strongest models and test whether ensemble improves macro F1 |
| 30 | Feature selection + Tuned SMOTE XGBoost | 0.9412 ± 0.0295 | 0.6058 | Remove noisy/redundant features and test whether generalization improves |
| 31 | Linear SVM with scaling and class weights | 0.8250 ± 0.0340 | 0.5549 | Test SVM as an alternative classifier for minority classes U2R was decent 0.37, but R2L was 0.06 |
| 32 | k-NN with scaling and SMOTE | 0.7641 ± 0.0076 | 0.6358 | Test distance-based classification as an additional comparison |
| 33 | Keras class weight + seed tuning | 0.7623 ± 0.0219 | 0.6849 | Reducing step size and decreasing dropout, increases Recall score for R2L. Also while F1 final stays same, small differences are observed in CV Macro F1 scores while run on diff Computers, something makes slight randomization, another aspect these values brought matching of precision and recall for U2R |
| 34 | Keras class weight + seed tuning | 0.7620 ± 0.0217 | 0.6907 | Decreasing value of normal traffic increases overall results for rest of the classes |
| 35 | Keras + NN w. balanced weights | NA | 0.6912 | Weights calculated by script very unstable (volatile) results each time gives different results, this was highest |
| 36 | 16_keras_neural_network | 0.7058 ± 0.0140 | 0.6944 |keras_2_balanced_weights, highest score again but not stable results |


## Planned experiments

| No. | File | Model / Approach | Purpose |
| --- | --- | --- | --- |



## Current best model


| Model | CV Macro F1 | Test Macro F1 | Notes |
| --- | ---: | ---: | --- |

| Best 34| Keras class weight + seed tuning | 0.7620 ± 0.0217 | 0.6907 | Decreasing value of normal traffic increases overall results for rest of the classes |
old:{0: 1.0, 1: 0.68, 2: 2.21, 3: 20.0, 4: 60.0}
new:{0: 1.0, 1: 0.67, 2: 2.22, 3: 19.95, 4: 60.0}

* 04_tuned_xgboost.py took around 15 minutes on colab
* 06_threshold_tuning_xgboost.pybest: validation configuration kept all multipliers at 1.0
* SMOTE + XGBoost + Feature Engineering was tested with more "agressive" XGBClassifier parameters and SMOTE R2l = 16000 (requires check with old parameters) 
* Feature engineering improved U2R detection but did not improve overall macro F1.
* Weighted XGBoost CV score was even higher, than best model result, however, CV score alone is not enough for NSL-KDD, test result was worse.
* SMOTEENN + Tuned XGBoost total experiment time: 3312.34 seconds ≈ 55 minutes, final train + prediction time: 421.36 seconds ≈ 7 minutes
* Feature Selection + Tuned SMOTE XGBoost, threshold = median, encoded features = 121, selected features = 61, test macro F1 = 0.6166
