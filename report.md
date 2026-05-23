# Assignment Report

---

**Course:** Advanced Python (ICS0019)

**Team members:** Radions Laškovs, Ilja Priimak

**Date:** 22.05.2026

Repository link: [[GitHub](https://github.com/wwqeew/nid-assignment)]

---

## 1. Approach

### 1.1 Strategy Overview

Our overall strategy was focused on improving detection of minority attack classes (R2L and U2R), because the baseline models handled Normal and DoS traffic relatively well but almost completely failed on rare attacks. Major part of the experiments focused on handling class imbalance using SMOTE, SMOTEENN, custom class weights, and probability threshold tuning. After tree-based models stopped showing significant improvement and remained around 0.60 macro F1-score, we shifted our focus toward Keras neural networks. 

We also experimented with feature engineering, feature selection, stacking ensembles, voting ensembles, threshold tuning, and categorical embeddings.

### 1.2 Preprocessing

Describe any changes you made to the data beyond the starter code:

- **Feature engineering:** Several additional features were created during experiments. Traffic ratio and interaction-based features such as byte ratios, combinations of connection statistics that could better separate minority attack classes from normal traffic. Feature engineering mainly targeted improving R2L and U2R detection.
- **Feature selection:** Feature selection was tested in later experiments. Low-importance and redundant features were removed based on XGBoost feature importance scores. Encoded feature set was reduced from 121 features to approximately 61 features in some experiments. This slightly improved generalization and reduced noise.
- **Scaling:** StandardScaler was applied for models sensitive to feature magnitude, especially SVM and neural network experiments. Tree-based models such as Random Forest, XGBoost, and LightGBM were mostly trained without scaling because they are less sensitive to feature scales.
- **Other:** Categorical features were encoded as numerical values using label encoding. In later experiments, we also tested neural-network embeddings for categorical features instead of traditional encoded representations. Random seeds were fixed in most experiments to improve reproducibility, although neural network experiments still showed noticeable instability between runs.

### 1.3 Class Imbalance Handling

Because NSL-KDD is highly imbalanced, especially for U2R attacks, handling imbalance became the main focus.

- **Method used:**  
  - SMOTE oversampling
  - SMOTEENN hybrid resampling
  - `class_weight='balanced'`
  - Manually tuned custom class weights
  - Sample weighting
  - Threshold/probability tuning
  - Ensemble-based balancing approaches

- **Parameters:**  
  Most SMOTE experiments used default `k_neighbors=5` configuration, while later experiments manually adjusted minority target sizes for R2L and U2R. Neural network experiments used aggressively tuned custom class weights to force the model to pay more attention to minority attack classes. Some experiments also used threshold multipliers to increase minority-class prediction probability.

- **Effect on training set distribution:**  
  SMOTE and SMOTEENN significantly increased number of minority-class samples, especially for R2L and U2R, creating much more balanced training distribution. This improved recall for rare attacks but sometimes reduced precision and introduced instability or overfitting. Custom class weights achieved better balance without directly modifying the dataset and produced the best overall results in later Keras experiments.

---

## 2. Experiments

### Experiment 1: Random Forest Baseline

- **Algorithm:** Random Forest Classifier
- **What changed from baseline:** This was the first baseline script. It tested the default Random Forest and also a balanced class-weight version.
- **Macro F1 (CV):** 0.9126 ± 0.0371 for the balanced version
- **Macro F1 (test):** 0.4707 for the balanced version; around 0.51 for the default version
- **Observation:** Default Random Forest generalized slightly better on the test set, while the balanced version had a high CV score but poor test performance. Both versions remained weak on R2L and U2R.

### Experiment 2: SMOTE + Random Forest

- **Algorithm:** Random Forest Classifier
- **What changed from baseline:** Added SMOTE oversampling to increase number of minority-class samples before training.
- **Macro F1 (CV):** 0.9369 ± 0.0269
- **Macro F1 (test):** 0.5497
- **Observation:** SMOTE improved result compared to the Random Forest baseline, especially for U2R, but R2L detection was still weak.

### Experiment 3: XGBoost and SMOTE + XGBoost

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Switched from Random Forest to XGBoost and then tested the same model with SMOTE.
- **Macro F1 (CV):** 0.9437 ± 0.0262 for SMOTE + XGBoost
- **Macro F1 (test):** 0.6073 for SMOTE + XGBoost; 0.5562 without SMOTE
- **Observation:** XGBoost already improved over Random Forest, and adding SMOTE gave a much stronger result by improving minority-class detection.

### Experiment 4: Tuned SMOTE + XGBoost

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Tuned the XGBoost parameters and SMOTE setup after the previous experiment.
- **Macro F1 (CV):** 0.9412 ± 0.0297
- **Macro F1 (test):** 0.6086
- **Observation:** Parameter tuning gave a small improvement over the previous SMOTE + XGBoost model and became the best XGBoost result at that stage.

### Experiment 5: LightGBM and SMOTE + LightGBM

- **Algorithm:** LightGBM Classifier
- **What changed from baseline:** Tested LightGBM as a faster gradient boosting alternative and then added SMOTE.
- **Macro F1 (CV):** 0.9296 ± 0.0283 for SMOTE + LightGBM
- **Macro F1 (test):** 0.5922 for SMOTE + LightGBM; 0.5444 without SMOTE
- **Observation:** LightGBM was faster and improved with SMOTE, but it did not outperform tuned SMOTE + XGBoost model.

### Experiment 6: Threshold Tuning + SMOTE + XGBoost

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Added probability threshold tuning on top of the SMOTE + XGBoost model.
- **Macro F1 (CV):** 0.9408 ± 0.0217
- **Macro F1 (test):** 0.6086
- **Observation:** Threshold tuning did not improve the model because the best probability multipliers stayed at 1.0.

### Experiment 7: Feature Engineering + XGBoost

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Added engineered features and tested both plain XGBoost and SMOTE + XGBoost with these features.
- **Macro F1 (CV):** 0.9480 ± 0.0269 for SMOTE + XGBoost + Feature Engineering
- **Macro F1 (test):** 0.6051 for SMOTE + XGBoost + Feature Engineering; 0.5730 without SMOTE
- **Observation:** Feature engineering improved plain XGBoost, but the SMOTE version still did not beat earlier tuned SMOTE + XGBoost result.

### Experiment 8: Tuned SMOTE + XGBoost + Feature Engineering

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Tuned the SMOTE + XGBoost model after adding engineered features.
- **Macro F1 (CV):** 0.9454 ± 0.0234
- **Macro F1 (test):** 0.6078
- **Observation:** Model was very close to the best XGBoost result and improved U2R, but Probe performance decreased, so the overall macro F1 did not improve.

### Experiment 9: SMOTEENN + Tuned XGBoost

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Replaced plain SMOTE with SMOTEENN to combine minority oversampling with cleaning of noisy samples.
- **Macro F1 (CV):** 0.9542 ± 0.0291
- **Macro F1 (test):** 0.6097
- **Observation:** SMOTEENN produced slight improvement over tuned SMOTE + XGBoost, but training time became much higher.

### Experiment 10: Weighted XGBoost

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Used custom sample weights instead of relying on SMOTE.
- **Macro F1 (CV):** 0.9506 ± 0.0204
- **Macro F1 (test):** 0.5968
- **Observation:** Cost-sensitive weighting improved minority-class detection without oversampling, but it did not beat the best SMOTE-based XGBoost models.

### Experiment 11: Threshold Tuning + SMOTE + LightGBM

- **Algorithm:** LightGBM Classifier
- **What changed from baseline:** Added threshold/probability tuning to the SMOTE + LightGBM model.
- **Macro F1 (CV):** 0.9295 ± 0.0283
- **Macro F1 (test):** 0.6021
- **Observation:** Threshold tuning improved LightGBM, mainly by boosting U2R probability, but R2L remained weak.

### Experiment 12: MLP Neural Network

- **Algorithm:** MLP Neural Network
- **What changed from baseline:** Tested a neural network model and also compared it with a SMOTE-based version.
- **Macro F1 (CV):** 0.8481 ± 0.0151 for SMOTE + MLP
- **Macro F1 (test):** 0.5996 without SMOTE; 0.5636 with SMOTE
- **Observation:** The plain MLP performed surprisingly well, while SMOTE reduced performance because U2R precision dropped.

### Experiment 13: Voting Ensemble + Probability Tuning

- **Algorithm:** Soft Voting Ensemble
- **What changed from baseline:** Combined Feature Selection XGBoost, LightGBM, and plain MLP using soft voting, tuned model weights, and probability multipliers.
- **Macro F1 (CV):** 0.9415 ± 0.0258
- **Macro F1 (test):** 0.6232
- **Observation:** This ensemble improved over most individual tree-based models. Tuned voting helped R2L and U2R compared to raw voting.

### Experiment 14: Feature Selection + Tuned SMOTE XGBoost

- **Algorithm:** XGBoost Classifier
- **What changed from baseline:** Added feature selection to the tuned SMOTE + XGBoost setup.
- **Macro F1 (CV):** 0.9415 ± 0.0291
- **Macro F1 (test):** 0.6166
- **Observation:** Feature selection reduced encoded feature set from 121 to 61 features and improved generalization compared to earlier XGBoost models.

### Experiment 15: Linear SVM

- **Algorithm:** Linear SVM
- **What changed from baseline:** Tested SVM as an alternative classifier with scaling and different imbalance-handling variants.
- **Macro F1 (CV):** 0.8248 ± 0.0338
- **Macro F1 (test):** 0.5555
- **Observation:** Plain Linear SVM performed best among the SVM variants, but R2L detection remained very weak.

### Experiment 16: Keras Neural Network with Custom Weights

- **Algorithm:** Keras Neural Network
- **What changed from baseline:** Tested a deeper neural network with manually selected class weights.
- **Macro F1 (CV):** 0.8675 ± 0.0381
- **Macro F1 (test):** 0.6399
- **Observation:** Custom class weights strongly improved U2R detection while keeping good performance on DoS, Normal, and Probe. R2L recall was still low.

### Experiment 17: Threshold Tuning + Keras Custom Weights

- **Algorithm:** Keras Neural Network
- **What changed from baseline:** Added probability threshold tuning to the weighted Keras model.
- **Macro F1 (CV):** 0.8433 ± 0.0381
- **Macro F1 (test):** 0.5625
- **Observation:** Threshold tuning did not improve Keras model and made the result less stable than experiment 16.

### Experiment 18: Keras Multi-Seed Ensemble

- **Algorithm:** Keras Neural Network Ensemble
- **What changed from baseline:** Trained three Keras models with different random seeds and averaged their predictions.
- **Macro F1 (CV):** 0.8495 ± 0.0295
- **Macro F1 (test):** 0.6147
- **Observation:** Averaging several seeds did not improve over the best single Keras model because weaker seeds reduced quality of averaged probabilities.

### Experiment 19: Keras Class Weight + Seed Tuning

- **Algorithm:** Keras Neural Network
- **What changed from baseline:** Tuned both the random seed and class weights for the Keras model.
- **Macro F1 (CV):** 0.8525 ± 0.0197
- **Macro F1 (test):** 0.6646
- **Observation:** This became the best result among experiments 1–24. Seed and class-weight tuning significantly improved R2L detection.

### Experiment 20: Keras Fine-Tuning Around Best Weights

- **Algorithm:** Keras Neural Network
- **What changed from baseline:** Fine-tuned class weights around the best setup from experiment 19.
- **Macro F1 (CV):** 0.8240 ± 0.0189
- **Macro F1 (test):** 0.6006
- **Observation:** Although validation tuning looked promising, fixed-epoch final model overfitted and R2L recall dropped, so this setup was not used.

### Experiment 21: Keras CV-Selected Training

- **Algorithm:** Keras Neural Network
- **What changed from baseline:** Selected the class-weight setup using 3-fold cross-validation and compared safer training strategies.
- **Macro F1 (CV):** 0.8690 ± 0.0263
- **Macro F1 (test):** 0.6010
- **Observation:** Cross-validation selected the same weights as experiment 19, but the final test score remained much lower than the best Keras result.

### Experiment 22: Keras Categorical Embeddings MLP

- **Algorithm:** Keras Neural Network with categorical embeddings
- **What changed from baseline:** Replaced one-hot encoding of categorical features with learned embeddings.
- **Macro F1 (CV):** 0.8321 ± 0.0506
- **Macro F1 (test):** 0.6578
- **Observation:** Learned embeddings improved R2L detection and produced one of the best test scores, but still did not beat experiment 19.

### Experiment 23: Stacking Keras + XGBoost

- **Algorithm:** Stacking Ensemble
- **What changed from baseline:** Combined experiment 19 Keras, experiment 4 SMOTE XGBoost, and experiment 14 Feature Selection XGBoost using logistic regression as a meta-model.
- **Macro F1 (CV):** 0.9259 ± 0.0317
- **Macro F1 (test):** 0.5491
- **Observation:** Stacked model had a high CV score but failed to generalize to KDDTest+, especially on R2L and U2R.

### Experiment 24: Voting Keras + XGBoost

- **Algorithm:** Soft Voting Ensemble
- **What changed from baseline:** Combined experiment 19 Keras, experiment 4 SMOTE XGBoost, and experiment 14 Feature Selection XGBoost using averaged probabilities.
- **Macro F1 (CV):** 0.9430 ± 0.0320
- **Macro F1 (test):** 0.5976
- **Observation:** Despite strong CV performance, this ensemble underpredicted R2L and U2R on KDDTest+, so it was worse than the best single Keras model.


### Experiments Summary

| # | Description | Algorithm | Imbalance Handling | Macro F1 (CV) | Macro F1 (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | Random Forest baseline | Random Forest | `class_weight='balanced'` tested | 0.9126 ± 0.0371 | 0.4707 |
| 2 | SMOTE + Random Forest | Random Forest | SMOTE | 0.9369 ± 0.0269 | 0.5497 |
| 3 | SMOTE + XGBoost | XGBoost | SMOTE | 0.9437 ± 0.0262 | 0.6073 |
| 4 | Tuned SMOTE + XGBoost | XGBoost | SMOTE | 0.9412 ± 0.0297 | 0.6086 |
| 5 | SMOTE + LightGBM | LightGBM | SMOTE | 0.9296 ± 0.0283 | 0.5922 |
| 6 | Threshold tuning + SMOTE XGBoost | XGBoost | SMOTE + threshold tuning | 0.9408 ± 0.0217 | 0.6086 |
| 7 | Feature engineering + SMOTE XGBoost | XGBoost | SMOTE | 0.9480 ± 0.0269 | 0.6051 |
| 8 | Tuned feature-engineered XGBoost | XGBoost | SMOTE | 0.9454 ± 0.0234 | 0.6078 |
| 9 | SMOTEENN + Tuned XGBoost | XGBoost | SMOTEENN | 0.9542 ± 0.0291 | 0.6097 |
| 10 | Weighted XGBoost | XGBoost | Custom sample weights | 0.9506 ± 0.0204 | 0.5968 |
| 11 | Threshold tuning + SMOTE LightGBM | LightGBM | SMOTE + threshold tuning | 0.9295 ± 0.0283 | 0.6021 |
| 12 | MLP Neural Network | MLP | None / SMOTE tested | 0.8481 ± 0.0151 | 0.5996 |
| 13 | Soft voting ensemble + probability tuning | Voting Ensemble | Mixed model-level handling | 0.9415 ± 0.0258 | 0.6232 |
| 14 | Feature selection + tuned SMOTE XGBoost | XGBoost | SMOTE | 0.9415 ± 0.0291 | 0.6166 |
| 15 | Linear SVM | Linear SVM | Class weights tested | 0.8248 ± 0.0338 | 0.5555 |
| 16 | Keras neural network with custom weights | Keras NN | Custom class weights | 0.8675 ± 0.0381 | 0.6399 |
| 17 | Threshold tuning + Keras custom weights | Keras NN | Custom class weights + threshold tuning | 0.8433 ± 0.0381 | 0.5625 |
| 18 | Keras multi-seed ensemble | Keras NN Ensemble | Custom class weights | 0.8495 ± 0.0295 | 0.6147 |
| 19 | Keras class weight + seed tuning | Keras NN | Tuned custom class weights | 0.8525 ± 0.0197 | 0.6646 |
| 20 | Keras fine-tuning around best weights | Keras NN | Fine-tuned class weights | 0.8240 ± 0.0189 | 0.6006 |
| 21 | Keras CV-selected training | Keras NN | CV-selected class weights | 0.8690 ± 0.0263 | 0.6010 |
| 22 | Keras categorical embeddings MLP | Keras NN with embeddings | Custom class weights | 0.8321 ± 0.0506 | 0.6578 |
| 23 | Stacking Keras + XGBoost | Stacking Ensemble | Mixed model-level handling | 0.9259 ± 0.0317 | 0.5491 |
| 24 | Voting Keras + XGBoost | Soft Voting Ensemble | Mixed model-level handling | 0.9430 ± 0.0320 | 0.5976 |

---

## 3. Final Results

### 3.1 Best Model

- **Algorithm:** [e.g., XGBoost]
- **Key parameters:** [e.g., n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=...]
- **Imbalance handling:** [e.g., SMOTE + class_weight]
- **Feature engineering:** [e.g., added src_bytes/dst_bytes ratio]

### 3.2 Final Macro F1-Score

| Metric | Score |
| --- | --- |
| **Macro F1 (test)** |  |
| Macro F1 (CV) |  |

### 3.3 Classification Report

| Category | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Normal |  |  |  |  |
| DoS |  |  |  |  |
| Probe |  |  |  |  |
| R2L |  |  |  |  |
| U2R |  |  |  |  |

### 3.4 Confusion Matrix

[Generate this image using the code from Section 9 of the guidebook. For Markdown report save it as `confusion_matrix.png` in the same folder as this report and add link `![Confusion Matrix](confusion_matrix.png)`. Don’t forget the exclamation mark].

## 4. Cross-Validation vs. Test Score

- **CV macro F1:** [score ± std]
- **Test macro F1:** [score]
- **Gap:** [CV − test]

**Analysis:** [Explain the gap. Is it expected? Is it due to unseen attack types in KDDTest+? Does it indicate overfitting?]

---

## 5. What Worked and What Didn't

### What had the biggest positive impact?

[e.g., "SMOTE increased R2L recall from 0.00 to 0.12, which raised macro F1 by 0.08"]

### What surprisingly didn't help?

[e.g., "Feature selection with SelectKBest removed 15 features but macro F1 dropped — the removed features contained information useful for rare classes"]

### What would you try with more time?

[e.g., "Stacking ensemble, more aggressive hyperparameter tuning, deeper feature engineering"]

---

## Appendix: Environment

- **Hardware:** [CPU, RAM, GPU if used]
- **Python version:** [e.g., 3.11]
- **Key libraries:** [scikit-learn version, xgboost version, etc.]
- **Random seed:** [e.g., 42]
