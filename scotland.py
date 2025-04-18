# Load appropriate packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, f1_score, make_scorer, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Random Forest
file_path = os.path.join(os.getcwd(), '/share/tcsi/lagoodal/Python/scaleData.csv')
data = pd.read_csv(file_path, encoding='unicode_escape')
data['ForestType'] = data['ForestType'].astype('category').cat.codes
X = data.drop(columns=['ForestType'])
y = data['ForestType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)
clf = RandomForestClassifier(random_state=33)
param_grid = {
    'n_estimators': [100, 120, 140, 160],
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_leaf_nodes': [None, 10, 20],
    'max_samples': [None, 0.8, 0.9],
    'max_features': ['sqrt', 'log2']
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=33)
grid_search_RF = GridSearchCV(estimator=clf, param_grid=param_grid,
                              cv=cv, scoring='roc_auc_ovr',
                              n_jobs=int(os.getenv('LSB_DJOB_NUMPROC', 1)))
grid_search_RF.fit(X_train, y_train)
best_rf = grid_search_RF.best_estimator_
y_pred_RF = best_rf.predict(X_test)
y_pred_proba_RF = best_rf.predict_proba(X_test)

auc_roc_data_RF = pd.DataFrame(columns = ['Class', 'AUC'])
roc_curve_data_RF = []
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba_RF.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_RF[:, i], pos_label = i)
    roc_auc_RF = auc(fpr, tpr)
    
    auc_roc_data_RF = pd.concat([auc_roc_data_RF, pd.DataFrame({'Class': [i], 'AUC': [roc_auc_RF]})], ignore_index = True)
    roc_curve_data_RF.append(pd.DataFrame({'Class': i, 'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds}))
    
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_RF:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()

# Save data to CSV
auc_roc_data_RF.to_csv('/share/tcsi/lagoodal/Python/auc_roc_results_RF.csv', index = False)
roc_curve_data_RF_df = pd.concat(roc_curve_data_RF, ignore_index = True)
roc_curve_data_RF_df.to_csv('/share/tcsi/lagoodal/Python/roc_curve_data_RF.csv', index = False)

report = classification_report(y_test, y_pred_RF)

print("Best Parameters:", grid_search_RF.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_RF))
print("F1 Score:", f1_score(y_test, y_pred_RF, average='weighted'))
print(report)
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_RF, multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test, y_pred_RF))

scenario1Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario1.csv')
scenario1RF = grid_search_RF.best_estimator_.predict(scenario1Df)
scenario1RF = pd.DataFrame(scenario1RF, columns=['Predicted_Class'])
scenario1RF.to_csv('/share/tcsi/lagoodal/Python/scenario1RFFinished.csv')

scenario2Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario2.csv')
scenario2RF = grid_search_RF.best_estimator_.predict(scenario2Df)
scenario2RF = pd.DataFrame(scenario2RF, columns=['Predicted_Class'])
scenario2RF.to_csv('/share/tcsi/lagoodal/Python/scenario2RFFinished.csv')

scenario3Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario3.csv')
scenario3RF = grid_search_RF.best_estimator_.predict(scenario3Df)
scenario3RF = pd.DataFrame(scenario3RF, columns=['Predicted_Class'])
scenario3RF.to_csv('/share/tcsi/lagoodal/Python/scenario3RFFinished.csv')

scenario4Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario4.csv')
scenario4RF = grid_search_RF.best_estimator_.predict(scenario4Df)
scenario4RF = pd.DataFrame(scenario4RF, columns=['Predicted_Class'])
scenario4RF.to_csv('/share/tcsi/lagoodal/Python/scenario4RFFinished.csv')

# XGBoost
file_path = os.path.join(os.getcwd(), '/share/tcsi/lagoodal/Python/scaleData.csv')
data = pd.read_csv(file_path, encoding='unicode_escape')
data['ForestType'] = data['ForestType'].astype('category').cat.codes
X = data.drop(columns=['ForestType'])
y = data['ForestType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)
xgb_model = XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y)),
                          n_jobs=int(os.getenv('LSB_DJOB_NUMPROC', 1)), random_state=33,
                          eval_metric='mlogloss', use_label_encoder=False)
param_grid = {
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 4],
    'subsample': [0.3, 0.4, 0.5],
    'min_child_weight': [1, 5] ,
    'gamma': [0, 0.1],
    'reg_alpha': [0.1, 0.3, 0.5],
    'reg_lambda': [1, 1.5, 2]
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=33)
grid_search_XGB = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=cv, scoring='roc_auc_ovr')
grid_search_XGB.fit(X_train, y_train)
best_model = grid_search_XGB.best_estimator_
y_pred_XGB = best_model.predict(X_test)
y_pred_proba_XGB = best_model.predict_proba(X_test)

# Store AUC and ROC data
auc_roc_data_XGB = pd.DataFrame(columns=['Class', 'AUC'])
roc_curve_data_XGB = []
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba_XGB.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_XGB[:, i], pos_label=i)
    roc_auc_XGB = auc(fpr, tpr)
    
    auc_roc_data_XGB = pd.concat([auc_roc_data_XGB, pd.DataFrame({'Class': [i], 'AUC': [roc_auc_XGB]})], ignore_index = True)
    roc_curve_data_XGB.append(pd.DataFrame({'Class': i, 'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds}))
    
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_XGB:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()
auc_roc_data_XGB.to_csv('/share/tcsi/lagoodal/Python/auc_roc_results_XGB.csv', index = False)
roc_curve_data_XGB_df = pd.concat(roc_curve_data_XGB, ignore_index = True)
roc_curve_data_XGB_df.to_csv('/share/tcsi/lagoodal/Python/roc_curve_data_XGB.csv', index = False)
report = classification_report(y_test, y_pred_XGB)
print("Best Parameters:", grid_search_XGB.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_XGB))
print("F1 Score:", f1_score(y_test, y_pred_XGB, average='weighted'))
print(report)
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_XGB, multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test, y_pred_XGB))

scenario1Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario1.csv')
scenario1XGB = grid_search_XGB.best_estimator_.predict(scenario1Df)
scenario1XGB = pd.DataFrame(scenario1XGB, columns=['Predicted_Class'])
scenario1XGB.to_csv('/share/tcsi/lagoodal/Python/scenario1XGBFinished.csv')
scenario2Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario2.csv')
scenario2XGB = grid_search_XGB.best_estimator_.predict(scenario2Df)
scenario2XGB = pd.DataFrame(scenario2XGB, columns=['Predicted_Class'])
scenario2XGB.to_csv('/share/tcsi/lagoodal/Python/scenario2XGBFinished.csv')
scenario3Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario3.csv')
scenario3XGB = grid_search_XGB.best_estimator_.predict(scenario3Df)
scenario3XGB = pd.DataFrame(scenario3XGB, columns=['Predicted_Class'])
scenario3XGB.to_csv('/share/tcsi/lagoodal/Python/scenario3XGBFinished.csv')
scenario4Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario4.csv')
scenario4XGB = grid_search_XGB.best_estimator_.predict(scenario4Df)
scenario4XGB = pd.DataFrame(scenario4XGB, columns=['Predicted_Class'])
scenario4XGB.to_csv('/share/tcsi/lagoodal/Python/scenario4XGBFinished.csv')

# MLP
file_path = os.path.join(os.getcwd(), '/share/tcsi/lagoodal/Python/scaleData.csv')
data = pd.read_csv(file_path, encoding='unicode_escape')
data['ForestType'] = data['ForestType'].astype('category').cat.codes
X = data.drop(columns=['ForestType'])
y = data['ForestType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)
mlp = MLPClassifier(max_iter=500, random_state=33)
param_grid = {
    'hidden_layer_sizes': [(32, 16), (64, 32, 16), (64, 32, 16, 8)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'early_stopping': [True]
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=33)
grid_search_MLP = GridSearchCV(mlp, param_grid=param_grid,
                               cv=cv, scoring='roc_auc_ovr',
                               n_jobs=int(os.getenv('LSB_DJOB_NUMPROC', 1)))
grid_search_MLP.fit(X_train, y_train)
best_mlp = grid_search_MLP.best_estimator_
y_pred_MLP = best_mlp.predict(X_test)
y_pred_proba_MLP = best_mlp.predict_proba(X_test)

# Store AUC and ROC data
auc_roc_data_MLP = pd.DataFrame(columns=['Class', 'AUC'])
roc_curve_data_MLP = []
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba_MLP.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_MLP[:, i], pos_label=i)
    roc_auc_MLP = auc(fpr, tpr)
    
    auc_roc_data_MLP = pd.concat([auc_roc_data_MLP, pd.DataFrame({'Class': [i], 'AUC': [roc_auc_MLP]})], ignore_index = True)
    roc_curve_data_MLP.append(pd.DataFrame({'Class': i, 'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds}))
    
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_MLP:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()
# Save data to CSV
auc_roc_data_MLP.to_csv('/share/tcsi/lagoodal/Python/auc_roc_results_MLP.csv', index = False)
roc_curve_data_MLP_df = pd.concat(roc_curve_data_MLP, ignore_index = True)
roc_curve_data_MLP_df.to_csv('/share/tcsi/lagoodal/Python/roc_curve_data_MLP.csv', index = False)
report = classification_report(y_test, y_pred_MLP)
print("Best Parameters:", grid_search_MLP.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_MLP))
print("F1 Score:", f1_score(y_test, y_pred_MLP, average='weighted'))
print(report)
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_MLP, multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test, y_pred_MLP))
scenario1Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario1.csv')
scenario1MLP = grid_search_MLP.best_estimator_.predict(scenario1Df)
scenario1MLP = pd.DataFrame(scenario1MLP, columns=['Predicted_Class'])
scenario1MLP.to_csv('/share/tcsi/lagoodal/Python/scenario1MLPFinished.csv')
scenario2Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario2.csv')
scenario2MLP = grid_search_MLP.best_estimator_.predict(scenario2Df)
scenario2MLP = pd.DataFrame(scenario2MLP, columns=['Predicted_Class'])
scenario2MLP.to_csv('/share/tcsi/lagoodal/Python/scenario2MLPFinished.csv')
scenario3Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario3.csv')
scenario3MLP = grid_search_MLP.best_estimator_.predict(scenario3Df)
scenario3MLP = pd.DataFrame(scenario3MLP, columns=['Predicted_Class'])
scenario3MLP.to_csv('/share/tcsi/lagoodal/Python/scenario3MLPFinished.csv')
scenario4Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario4.csv')
scenario4MLP = grid_search_MLP.best_estimator_.predict(scenario4Df)
scenario4MLP = pd.DataFrame(scenario4MLP, columns=['Predicted_Class'])
scenario4MLP.to_csv('/share/tcsi/lagoodal/Python/scenario4MLPFinished.csv')

# SVM
file_path = os.path.join(os.getcwd(), '/share/tcsi/lagoodal/Python/scaleData.csv')
data = pd.read_csv(file_path, encoding='unicode_escape')
data['ForestType'] = data['ForestType'].astype('category').cat.codes
X = data.drop(columns=['ForestType'])
y = data['ForestType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)
svm = SVC(probability=True, random_state=33)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=33)
grid_search_SVM = GridSearchCV(estimator=svm, param_grid=param_grid,
                               cv=cv, scoring='roc_auc_ovr',
                               n_jobs=int(os.getenv('LSB_DJOB_NUMPROC', 1)))
grid_search_SVM.fit(X_train, y_train)
best_svm = grid_search_SVM.best_estimator_
y_pred_SVM = best_svm.predict(X_test)
y_pred_proba_SVM = best_svm.predict_proba(X_test)

# Store AUC and ROC data
auc_roc_data_SVM = pd.DataFrame(columns=['Class', 'AUC'])
roc_curve_data_SVM = []
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba_SVM.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_SVM[:, i], pos_label=i)
    roc_auc_SVM = auc(fpr, tpr)
    
    auc_roc_data_SVM = pd.concat([auc_roc_data_SVM, pd.DataFrame({'Class': [i], 'AUC': [roc_auc_SVM]})], ignore_index = True)
    roc_curve_data_SVM.append(pd.DataFrame({'Class': i, 'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds}))
    
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_SVM:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()
# Save data to CSV
auc_roc_data_SVM.to_csv('/share/tcsi/lagoodal/Python/auc_roc_results_SVM.csv', index = False)
roc_curve_data_SVM_df = pd.concat(roc_curve_data_SVM, ignore_index = True)
roc_curve_data_SVM_df.to_csv('/share/tcsi/lagoodal/Python/roc_curve_data_SVM.csv', index = False)
report = classification_report(y_test, y_pred_SVM)
print("Best Parameters:", grid_search_SVM.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_SVM))
print("F1 Score:", f1_score(y_test, y_pred_SVM, average='weighted'))
print(report)
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_SVM, multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test, y_pred_SVM))
scenario1Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario1.csv')
scenario1SVM = grid_search_SVM.best_estimator_.predict(scenario1Df)
scenario1SVM = pd.DataFrame(scenario1SVM, columns=['Predicted_Class'])
scenario1SVM.to_csv('/share/tcsi/lagoodal/Python/scenario1SVMFinished.csv')
scenario2Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario2.csv')
scenario2SVM = grid_search_SVM.best_estimator_.predict(scenario2Df)
scenario2SVM = pd.DataFrame(scenario2SVM, columns=['Predicted_Class'])
scenario2SVM.to_csv('/share/tcsi/lagoodal/Python/scenario2SVMFinished.csv')
scenario3Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario3.csv')
scenario3SVM = grid_search_SVM.best_estimator_.predict(scenario3Df)
scenario3SVM = pd.DataFrame(scenario3SVM, columns=['Predicted_Class'])
scenario3SVM.to_csv('/share/tcsi/lagoodal/Python/scenario3SVMFinished.csv')
scenario4Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario4.csv')
scenario4SVM = grid_search_SVM.best_estimator_.predict(scenario4Df)
scenario4SVM = pd.DataFrame(scenario4SVM, columns=['Predicted_Class'])
scenario4SVM.to_csv('/share/tcsi/lagoodal/Python/scenario4SVMFinished.csv')

# Naive Bayes
file_path = os.path.join(os.getcwd(), '/share/tcsi/lagoodal/Python/scaleData.csv')
data = pd.read_csv(file_path, encoding='unicode_escape')
data['ForestType'] = data['ForestType'].astype('category').cat.codes
X = data.drop(columns=['ForestType'])
y = data['ForestType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)
nb = GaussianNB()
param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=33)
grid_search_NB = GridSearchCV(estimator=nb, param_grid=param_grid,
                              cv=cv, scoring='roc_auc_ovr',
                              n_jobs=int(os.getenv('LSB_DJOB_NUMPROC', 1)))
grid_search_NB.fit(X_train, y_train)
best_nb = grid_search_NB.best_estimator_
y_pred_NB = best_nb.predict(X_test)
y_pred_proba_NB = best_nb.predict_proba(X_test)

# Store AUC and ROC data
auc_roc_data_NB = pd.DataFrame(columns=['Class', 'AUC'])
roc_curve_data_NB = []
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba_NB.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_NB[:, i], pos_label=i)
    roc_auc_NB = auc(fpr, tpr)
    
    auc_roc_data_NB = pd.concat([auc_roc_data_NB, pd.DataFrame({'Class': [i], 'AUC': [roc_auc_NB]})], ignore_index = True)
    roc_curve_data_NB.append(pd.DataFrame({'Class': i, 'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds}))
    
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_NB:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()
# Save data to CSV
auc_roc_data_NB.to_csv('/share/tcsi/lagoodal/Python/auc_roc_results_SVM.csv', index = False)
roc_curve_data_NB_df = pd.concat(roc_curve_data_NB, ignore_index = True)
roc_curve_data_NB_df.to_csv('/share/tcsi/lagoodal/Python/roc_curve_data_SVM.csv', index = False)
report = classification_report(y_test, y_pred_NB)
print("Best Parameters:", grid_search_NB.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_NB))
print("F1 Score:", f1_score(y_test, y_pred_NB, average = 'weighted'))
print(report)
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_NB, multi_class = 'ovr'))
print("Classification Report:\n", classification_report(y_test, y_pred_NB))
scenario1Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario1.csv')
scenario1NB = grid_search_NB.best_estimator_.predict(scenario1Df)
scenario1NB = pd.DataFrame(scenario1NB, columns=['Predicted_Class'])
scenario1NB.to_csv('/share/tcsi/lagoodal/Python/scenario1NBFinished.csv')
scenario2Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario2.csv')
scenario2NB = grid_search_NB.best_estimator_.predict(scenario2Df)
scenario2NB = pd.DataFrame(scenario2NB, columns=['Predicted_Class'])
scenario2NB.to_csv('/share/tcsi/lagoodal/Python/scenario2NBFinished.csv')
scenario3Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario3.csv')
scenario3NB = grid_search_NB.best_estimator_.predict(scenario3Df)
scenario3NB = pd.DataFrame(scenario3NB, columns=['Predicted_Class'])
scenario3NB.to_csv('/share/tcsi/lagoodal/Python/scenario3NBFinished.csv')
scenario4Df = pd.read_csv('/share/tcsi/lagoodal/Python/scaleScenario4.csv')
scenario4NB = grid_search_NB.best_estimator_.predict(scenario4Df)
scenario4NB = pd.DataFrame(scenario4NB, columns=['Predicted_Class'])
scenario4NB.to_csv('/share/tcsi/lagoodal/Python/scenario4NBFinished.csv')
