import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import numpy as np

train = pd.read_csv("train.csv")
pd.set_option("display.max_columns", None)
print(train.dtypes)

# Conduct missing values
print(train.isnull().sum().sum())

# Convert binary variables into values
train["international_plan_num"] = train["international_plan"].map({"yes":1, "no":0})
train["voice_mail_plan_num"] = train["voice_mail_plan"].map({"yes":1,"no":0})
train["churn_num"] = train["churn"].map({"yes":1, "no": 0})

# Create new area_code_num column
train["area_code_num"] = train["area_code"].str.extract(r"(\d+)$").astype(int)

# Set Churn features
churn_features = train.drop(["state", "churn", "area_code","international_plan", "voice_mail_plan"], axis=1)

print(churn_features.dtypes)
churn_features.describe().T
statistic_summay = churn_features.describe()
print(tabulate(statistic_summay, headers = "keys", tablefmt = "github", showindex=False))

# Conduct correaltion with each variables in churn_features dataframe
corr_churn_features = churn_features.corr(numeric_only = True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_churn_features, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")

# Set X-features and Y variable
X = churn_features.drop(columns=["total_day_charge",
                                 "total_eve_charge",
                                 "total_night_charge",
                                 "total_intl_charge",
                                 "number_vmail_messages",
                                 "churn_num"])
Y = churn_features[["churn_num"]]

# Split data and 30% as for test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Set K-fold = 5

##-------------------- Lasso Logistic Regression ------------------------##
# Standardize features 
lasso_log_reg_cv = LogisticRegressionCV(Cs=30, penalty="l1", solver="saga", cv=5, scoring="neg_log_loss", max_iter=10000, refit=True, random_state=42)
pipeline = make_pipeline(StandardScaler(), lasso_log_reg_cv)
pipeline.fit(x_train, y_train.values.ravel())

# Extract actual C values used in model
coefs_path = lasso_log_reg_cv.coefs_paths_[1]
actual_Cs = coefs_path.shape[2]
C_used = lasso_log_reg_cv.Cs_[:actual_Cs]

# Count number of non-zero coefficients for each lambda
coefs = np.array([np.count_nonzero(coefs_path[0, :, i]) for i in range(actual_Cs)])
log_lambda = np.log(1 / C_used)

# Cross-validated deviance
mean_deviance = -lasso_log_reg_cv.scores_[1].mean(axis=0)[:actual_Cs]
std_deviance = lasso_log_reg_cv.scores_[1].std(axis=0)[:actual_Cs]

# LambdaMinDev
best_C = lasso_log_reg_cv.C_[0]
num_features_lambda_min = np.count_nonzero(lasso_log_reg_cv.coef_)

# Lambda1SE
min_deviance = mean_deviance.min()
lambda_1se_idx = np.where(mean_deviance <= min_deviance + std_deviance[np.argmin(mean_deviance)])[0][0]
C_1se = C_used[lambda_1se_idx]

# Retrain model using Lambda1SE
lasso_1se = LogisticRegression(penalty="l1", C=C_1se, solver="saga", max_iter=10000)
lasso_1se_model = make_pipeline(StandardScaler(), lasso_1se)
lasso_1se_model.fit(x_train, y_train.values.ravel())
num_features_lambda_1se = np.count_nonzero(lasso_1se_model.named_steps["logisticregression"].coef_)

# Plot: Number of predictors vs log(lambda)
plt.figure(figsize=(10, 5))
plt.plot(log_lambda, coefs, marker="o")
plt.axvline(np.log(1 / best_C), color="red", linestyle="--", label="LambdaMinDev")
plt.axvline(np.log(1 / C_1se), color="green", linestyle="--", label="Lambda1SE")
plt.xlabel("log(lambda)")
plt.ylabel("Number of Predictors")
plt.title("Number of Predictors vs log(lambda)")
plt.legend()
plt.grid(True)

# Plot 2: Deviance plot
plt.figure(figsize=(10, 5))
plt.errorbar(log_lambda, mean_deviance, yerr=std_deviance, fmt="o", ecolor="gray", capsize=3)
plt.axvline(np.log(1 / lasso_log_reg_cv.C_[0]), color="red", linestyle="--", label="LambdaMinDev")
plt.axvline(np.log(1 / C_1se), color="green", linestyle="--", label="Lambda1SE")
plt.xlabel("log(lambda)")
plt.ylabel("Cross-Validated Deviance")
plt.title("Deviance vs log(lambda)")
plt.legend()
plt.grid(True)

(num_features_lambda_min, num_features_lambda_1se, best_C, C_1se)

# Predict probabilities on the test set
y_test_pred_proba_log = lasso_1se_model.predict_proba(x_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba_log)
auc_score = roc_auc_score(y_test, y_test_pred_proba_log)

# Find optimal threshold (Youden’s J statistic)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
optimal_point = (fpr[optimal_idx], tpr[optimal_idx])

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.scatter(*optimal_point, color="red", label=f"Optimal threshold = {optimal_threshold:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Lasso (Lambda1SE)")
plt.legend()
plt.grid(True)

# Table format for AUC, optimal ROC point, threshold
roc_summary_lasso_log = pd.DataFrame({
    "Metric": ["AUC", "Optimal FPR", "Optimal TPR", "Optimal Threshold"],
    "Value": [round(auc_score, 4), round(optimal_point[0], 4), round(optimal_point[1], 4), round(optimal_threshold, 4)]})

print(tabulate(roc_summary_lasso_log, headers = "keys", tablefmt = "github", showindex=False))

# Predict binary using optimal threshold
y_test_pred_log_binary = (y_test_pred_proba_log >= optimal_threshold).astype(int)

# Compute confusion matrix and accuracy
cm_log = confusion_matrix(y_test, y_test_pred_log_binary)
accuracy_log = accuracy_score(y_test, y_test_pred_log_binary)
f1_log = f1_score(y_test, y_test_pred_log_binary)
auc_log = roc_auc_score(y_test, y_test_pred_proba_log)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix of Lasso Logistic Regression (Threshold = {optimal_threshold:.2f})")
plt.grid(False)

# Table for accuracy, f1 and auc of logistic regression model
evaluation_log = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "F1 Score"],
    "Logistic regression": [round(accuracy_log, 4), round(auc_log, 4), round(f1_log, 4)]})

## --------------------------- SVM ------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define RBF SVM model
svm_rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma = "scale"))
svm_rbf_accuracy = cross_val_score(svm_rbf, x_train, y_train, cv = cv, scoring="accuracy")
svm_rbf_accuracy_avg = svm_rbf_accuracy.mean()

# Define Polynomial SVM model
svm_poly = make_pipeline(StandardScaler(), SVC(kernel="poly", degree = 3))
svm_poly_accuracy = cross_val_score(svm_poly, x_train, y_train, cv =cv, scoring="accuracy")
svm_poly_accuracy_avg =svm_poly_accuracy.mean()

svm_accuracy_summary = pd.DataFrame({
    "Model": ["SVM RBF", "SVM Polynomial"],
    "Accuracy": [round(svm_rbf_accuracy_avg, 4), round(svm_poly_accuracy_avg, 4)]})
print(tabulate(svm_accuracy_summary, headers = "keys", tablefmt = "github", showindex=False))

# Retrain Polynomial SVM on full training data
svm_poly_final = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, probability=True))
svm_poly_final.fit(x_train, y_train)

# Predict on test set
y_test_pred_poly = svm_poly_final.predict(x_test)
y_test_pred_proba_poly = svm_poly_final.predict_proba(x_test)[:, 1]

# Compute metrics
cm_poly = confusion_matrix(y_test, y_test_pred_poly)
accuracy_poly = accuracy_score(y_test, y_test_pred_poly)
auc_poly = roc_auc_score(y_test, y_test_pred_proba_poly)
f1_poly = f1_score(y_test, y_test_pred_poly)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_poly)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Polynomial SVM")
plt.grid(False)

# Table for accuracy, f1 and auc of svm classifier
evaluation_poly = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "F1 Score"],
    "SVM": [round(accuracy_poly, 4), round(auc_poly, 4), round(f1_poly, 4)]})

## ---------------------- Classification Tree --------------------------------

# Create exponentially spaced values for min_samples_leaf from 10 to 100
min_leaf_sizes = np.unique(np.round(np.logspace(np.log10(10), np.log10(100), num=10)).astype(int))

# Store cross-validated accuracy for each setting
accuracy_trees = []

# Perform cross-validation for each leaf size
for leaf_size in min_leaf_sizes:
    clf = DecisionTreeClassifier(min_samples_leaf=leaf_size, random_state=42)
    pipeline = make_pipeline(StandardScaler(), clf)
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="accuracy")
    accuracy_trees.append(scores.mean())

# Find optimal leaf size
optimal_leaf_idx = np.argmax(accuracy_trees)
optimal_leaf_size = min_leaf_sizes[optimal_leaf_idx]
optimal_accuracy_tree = accuracy_trees[optimal_leaf_idx]

# Plot accuracy vs. leaf size
plt.figure(figsize=(10, 6))
plt.plot(min_leaf_sizes, accuracy_trees, marker="o")
plt.xlabel("Min Leaf Size")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Classification Tree: Accuracy vs. Min Leaf Size")
plt.grid(True)

tree_summary = pd.DataFrame({
    "Summary": ["Optimal leaf size", "Optimal accuracy"],
    "Value": [round(optimal_leaf_size, 4), round(optimal_accuracy_tree, 4)]})

print(tabulate(tree_summary, headers = "keys", tablefmt = "github", showindex=False))

# Retrain classification tree with the optimal Min Leaf Size
tree_final = make_pipeline(StandardScaler(), DecisionTreeClassifier(min_samples_leaf=optimal_leaf_size, random_state=42))
tree_final.fit(x_train, y_train)

# Predict on test set
y_test_pred_tree = tree_final.predict(x_test)
y_test_pred_proba_tree = tree_final.named_steps["decisiontreeclassifier"].predict_proba(x_test)[:, 1]

# Evaluate metrics
cm_tree = confusion_matrix(y_test, y_test_pred_tree)
accuracy_tree = accuracy_score(y_test, y_test_pred_tree)
auc_tree = roc_auc_score(y_test, y_test_pred_proba_tree)
f1_tree = f1_score(y_test, y_test_pred_tree)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_tree)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Classification Tree")
plt.grid(False)

# Table for accuracy, f1 and auc of Classification tree 
evaluation_tree = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "F1 Score"],
    "Classification": [round(accuracy_tree, 4), round(auc_tree, 4), round(f1_tree, 4)]})

# Recalculate ROC curves for each model
# Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, y_test_pred_proba_log)

# SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_test_pred_proba_poly)

# Classification Tree
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_test_pred_proba_tree)

# Plot all ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label="SVM")
plt.plot(fpr_log, tpr_log, label="Lasso Logistic Regression")
plt.plot(fpr_tree, tpr_tree, label="Classification Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Each Model")
plt.legend()
plt.grid(True)

# Merge accuracy, AUC, F1 score into one table
evaluation_performance = evaluation_log.merge(evaluation_poly, on="Metric").merge(evaluation_tree, on="Metric")
evaluation_performance_melted = evaluation_performance.melt(id_vars="Metric", var_name="Model", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=evaluation_performance_melted, x="Metric", y="Score", hue="Model")
plt.title("Model Comparison: Accuracy, AUC, F1 Score")
plt.ylim(0.0, 1.0)
plt.ylabel("Score")
plt.grid(axis="y")
plt.legend(title="Model")

## ----------------------- Prediction "Churn" in test data file --------------

# Read "test" data
test_df = pd.read_csv("test.csv")
test = test_df.copy()
pd.set_option("display.max_columns", None)
print(test.dtypes)

# Convert binary variables into values
test["international_plan_num"] = test["international_plan"].map({"yes":1, "no":0})
test["voice_mail_plan_num"] = test["voice_mail_plan"].map({"yes":1,"no":0})

# Create new area_code_num column
test["area_code_num"] = test["area_code"].str.extract(r"(\d+)$").astype(int)

# Preprocessing to match features 
test_features = test.drop(["id","state", "area_code","international_plan", "voice_mail_plan",
                           "total_day_charge", "total_eve_charge", "total_night_charge", 
                           "total_intl_charge","number_vmail_messages"], axis=1)

# Predict churn in test features data frame
y_test_churn = svm_poly_final.predict(test_features)
test["churn_num"] = y_test_churn

# map 0 into no, 1 into yes
test_df["churn"] = test["churn_num"].map({0:"no", 1:"yes"})

# Save final prediction test dataset into excel file
output_path = "test_prediction.csv"
test_df.to_csv(output_path, index=False)
test_df = test_df.iloc[1:].reset_index(drop=True)
print(test_df.dtypes)















