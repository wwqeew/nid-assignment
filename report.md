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

### Experiment 25: SMOTEEN + Tuned XGBoost
- **Algorithm:** Tuned XGBoost
- **What changed  from baseline:** Combined oversampling and cleaning to improve R2L vs Normal separation.
- **Macro F1 (CV):** 0.9412 ± 0.0295
- **Macro F1 (test):** 0.6058 
- **Observation:** Despite strong CV performance, this ensemble underpredicted R2L and U2R on KDDTest+.

### Experiment 26: XGBoost + sample weight
- **Algorithm:** 
- **What changed  from baseline:** Cost-sensitive XGBoost with `sample_weight`. Penalize mistakes on R2L and U2R more strongly without relying only on SMOTE.
- **Macro F1 (CV):** 0.9506 ± 0.0204
- **Macro F1 (test):** 0.5968
- **Observation:** Despite strong CV performance, this ensemble underpredicted R2L and U2R on KDDTest+.

### Experiment 27: Threshold / probability tuning for LightGBM
- **Algorithm:** Threshold / probability tuning for LightGBM
- **What changed  from baseline:** Added threshold / probability tuning
- **Macro F1 (CV):** 0.9329 ± 0.0252
- **Macro F1 (test):** 0.6080
- **Observation:** 0.48 score for U2R with multiplier 4.0 

### Experiment 28: SMOTE + MLP neural network
- **Algorithm:** SMOTE + MLP neural network
- **What changed  from baseline:** Test an actual neural network model for comparison.
- **Macro F1 (CV):** 0.8614 ± 0.0217
- **Macro F1 (test):** 0.5919
- **Observation:** Drop in CV performance, however not so significant drop in test performance.

### Experiment 29: Voting ensemble of XGBoost, LightGBM, Random Forest 
- **Algorithm:** XGBoost, LightGBM, Random Forest
- **What changed  from baseline:** Combine strongest models and test whether ensemble improves macro F1.
- **Macro F1 (CV):** 0.9400 ± 0.0260
- **Macro F1 (test):** 0.6168
- **Observation:** Raw voting had 0.5957 best score still was tuned voting.

### Experiment 30: Feature selection + Tuned SMOTE XGBoost
- **Algorithm:** Feature selection + Tuned SMOTE XGBoost
- **What changed  from baseline:** Remove noisy/redundant features and test whether generalization improves.
- **Macro F1 (CV):** 0.9412 ± 0.0295
- **Macro F1 (test):** 0.6058 
- **Observation:** No obvious nor significant improvements.

### Experiment 31: Linear SVM with scaling and class weights
- **Algorithm:** Linear SVM + scaling + class weights
- **What changed  from baseline:** Test SVM as an alternative classifier for minority classes. 
- **Macro F1 (CV):** 0.8250 ± 0.0340
- **Macro F1 (test):** 0.5549
- **Observation:** U2R was decent 0.37, but R2L was 0.06

### Experiment 32: k-NN with scaling and SMOTE
- **Algorithm:** k-NN with scaling and SMOTE
- **What changed  from baseline:** Test distance-based classification as an additional comparison
- **Macro F1 (CV):** 0.7641 ± 0.0076
- **Macro F1 (test):** 0.6358 
- **Observation:** Signinficant CV loss, however signigficant test result

### Experiment 33: Keras class weight + seed tuning
- **Algorithm:** Keras class weight + seed tuning
- **What changed  from baseline:** Reducing step size and decreasing dropout.
- **Macro F1 (CV):** 0.7623 ± 0.0219
- **Macro F1 (test):** 0.6849
- **Observation:** Increases Recall score for R2L. Also while F1 final stays same, small differences are observed in CV Macro F1 scores while run on diff Computers, something makes slight randomization, another aspect these values brought matching of precision and recall for U2R.

### Experiment 34: Keras class weight + seed tuning
- **Algorithm:** Keras class weight + seed tuning
- **What changed  from baseline:** Reducing weight for normal traffic.
- **Macro F1 (CV):** 0.7620 ± 0.0217
- **Macro F1 (test):** 0.6907 
- **Observation:** Decreasing value of normal traffic increases overall results for rest of the classes. Decrease is not proportional. Afterwards multiple small changes was made, highest score still stayed within initial test values. Also for U2R score of precission 0.51 and recall 0.51 was achieved, which is very balanced and good result.

### Experiment 35: Keras + NN w. balanced weights 
- **Algorithm:** Keras + NN w. balanced weights 
- **What changed  from baseline:** Test balanced weights along with some custom weights.
- **Macro F1 (CV):** Missing
- **Macro F1 (test):** 0.6912 
- **Observation:** Weights calculated by script. Very unstable (volatile) results each time gives different results, this was highest output from multiple same value runs. Some values in code cause randomization.

### Experiment 36: Keras + NN
- **Algorithm:** Keras + NN
- **What changed  from baseline:** Changed code behavior to reduce randomization, kept balanced weights removed custom weights.
- **Macro F1 (CV):** 0.7058 ± 0.0140
- **Macro F1 (test):** 0.6944 
- **Observation:** keras_2_balanced_weights, highest score again, however difficulty to replicate results.

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

---

## 3. Final Results

### 3.1 Best Model

- **Algorithm:** Keras + NN w. balanced weights
- **Key parameters:** Dropout: 0.18, Learning Rate: 0.00001, Epochs: 120, Patience: 10, 
- **Imbalance handling:** Weights: calculated by script, depends on representation ratio.
- **Feature engineering:** Changed fix() from shuffle=True (Default) to False. Gave stable reulsts but also random record scores.

### 3.2 Final Macro F1-Score

| Metric | Score |
| --- | --- |
| **Macro F1 (test)** | 0.6944 |
| Macro F1 (CV) | 0.7058 ± 0.0140 |

### 3.3 Classification Report

| Category | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Normal | 0.78 | 0.91 | 0.84 | 9711 |
| DoS | 0.91 | 0.85 | 0.88 | 7460 |
| Probe | 0.75 | 0.83 | 0.79 | 2421 |
| R2L | 0.91 | 0.45 | 0.61 | 2885 |
| U2R | 0.28 | 0.52 | 0.36 | 67 |

### 3.4 Confusion Matrix

[Generate this image using the code from Section 9 of the guidebook. For Markdown report save it as `confusion_matrix.png` in the same folder as this report and add link `![Confusion Matrix](confusion_matrix.png)`. Don’t forget the exclamation mark].

## 4. Cross-Validation vs. Test Score

- **CV macro F1:** 0.7058 ± 0.0140
- **Test macro F1:** 0.6944
- **Gap:** 0.0114

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

- **Hardware:** Intel i5-10600K, 64GB, 2xRTX2080Ti 
- **Python version:** 3.12.3
- **Key libraries:** {
absl-py==2.4.0
astunparse==1.6.3
certifi==2026.5.20
charset-normalizer==3.4.7
contourpy==1.3.3
cycler==0.12.1
flatbuffers==25.12.19
fonttools==4.63.0
gast==0.7.0
google-pasta==0.2.0
grpcio==1.80.0
h5py==3.14.0
idna==3.16
imbalanced-learn==0.14.1
joblib==1.5.3
keras==3.14.1
kiwisolver==1.5.0
libclang==18.1.1
lightgbm==4.6.0
Markdown==3.10.2
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mdurl==0.1.2
ml_dtypes==0.5.4
namex==0.1.0
numpy==2.4.6
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.22.0.52
nvidia-cufft-cu12==11.4.1.4
nvidia-curand-cu12==10.3.10.19
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nccl-cu12==2.30.4
nvidia-nvjitlink-cu12==12.9.86
opt_einsum==3.4.0
optree==0.19.1
packaging==26.2
pandas==3.0.3
pillow==12.2.0
protobuf==7.35.0
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
requests==2.34.2
rich==15.0.0
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
setuptools==82.0.1
six==1.17.0
sklearn-compat==0.1.5
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.3.0
threadpoolctl==3.6.0
typing_extensions==4.15.0
urllib3==2.7.0
Werkzeug==3.1.8
wheel==0.47.0
wrapt==2.2.0
xgboost==3.2.0
}
- **Random seed:** 100 (highest results in overall)
