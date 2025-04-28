import pandas as pd

# Load original data
train = pd.read_csv("train.csv")
test_pred = pd.read_csv("test_prediction.csv")

# Extract numeric part from area_code
train["area_code_num"] = train["area_code"].str.extract(r"(\d+)$").astype(int)
test_pred["area_code_num"] = test_pred["area_code"].str.extract(r"(\d+)$").astype(int)

# Binary encoding for churn
train["churn_numeric"] = train["churn"].map({"yes": 1, "no": 0})
test_pred["churn_numeric"] = test_pred["churn"].map({"yes": 1, "no": 0})

# Save the cleaned into excel worksheet
with pd.ExcelWriter("train_cleaned_for_tableau.xlsx", engine="xlsxwriter") as writer:
    train.to_excel(writer, sheet_name="train", index=False)
    
with pd.ExcelWriter("test_pred_cleaned_for_tableau.xlsx", engine="xlsxwriter") as writer:
   test_pred.to_excel(writer, sheet_name="test_prediction", index=False)

print("Cleaned Excel file saved successfully.")

