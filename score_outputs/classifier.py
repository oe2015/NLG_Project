import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier  # ✅ New import
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


gpt2_mintaka_solo = pd.read_parquet('generated_texts_gpt2_large_mintaka_solo_scores.parquet')
gpt2_mintaka = pd.read_parquet('generated_texts_gpt2_large_mintaka_scores.parquet')
gpt_neo_mintaka = pd.read_parquet('generated_texts_gptneo_mintaka_scores.parquet')
gpt_neo_mintaka_solo = pd.read_parquet('generated_texts_gptneo_mintaka_solo_scores.parquet')

gpt2_wikimia_solo = pd.read_parquet('generated_texts_gpt2_large_wikimia_solo_scores.parquet')
gpt2_wikimia = pd.read_parquet('generated_texts_gpt2_large_wikimia_scores.parquet')
gpt_neo_wikimia = pd.read_parquet('generated_texts_gptneo_wikimia_scores.parquet')
gpt_neo_wikimia_solo = pd.read_parquet('generated_texts_gptneo_wikimia_solo_scores.parquet')


from datasets import load_dataset
import pandas as pd

# Available text lengths
lengths = [32, 64, 128, 256]

# Load and store all splits
datasets = []
for length in lengths:
    ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
    df = ds.to_pandas()
    df['length'] = length  # Add a column to indicate source length
    datasets.append(df)

# Concatenate all datasets
all_data = pd.concat(datasets, ignore_index=True)

# Count label distribution
label_counts = all_data['label'].value_counts()

print("Label counts across all lengths:")
print(label_counts)


# Load the dataframe
gpt_neo_wikimia_solo = pd.read_parquet('generated_texts_gptneo_mintaka_solo_scores.parquet')

# Count label distribution
label_counts = gpt_neo_wikimia_solo['use_for_contamination'].value_counts()

print("Label counts in gpt_neo_wikimia_solo:")
print(label_counts)

# print(gpt2_mintaka_solo["use_for_contamination"])
# print(gpt2_mintaka_solo["extracted_answer"])
# print(gpt2_mintaka_solo["answerText"])

# # pick the three columns you care about
# exam = gpt2_mintaka_solo[['use_for_contamination', 'extracted_answer', 'answerText']]

# # write to a CSV for easy inspection
# exam.to_csv('mintaka_solo_examination.csv', index=False)

# print("Wrote", len(exam), "rows to mintaka_solo_examination.csv")

# dataframes = {
#     # Mintaka generation
#     "gpt2_train_mintaka_gen_mintaka": gpt2_mintaka_solo,              
#     "gpt2_train_both_gen_mintaka": gpt2_mintaka,                      
#     "gpt_neo_train_both_gen_mintaka": gpt_neo_mintaka,                
#     "gpt_neo_train_mintaka_gen_mintaka": gpt_neo_mintaka_solo,        

#     # WikiMIA generation
#     "gpt2_train_wikimia_gen_wikimia": gpt2_wikimia_solo,              
#     "gpt2_train_both_gen_wikimia": gpt2_wikimia,                      
#     "gpt_neo_train_both_gen_wikimia": gpt_neo_wikimia,                
#     "gpt_neo_train_wikimia_gen_wikimia": gpt_neo_wikimia_solo         
# }


# def train_logistic_regression_with_report(df, max_depth=2, features=None, target_column=None, test_split=None):
#     # Default feature list
#     if features is None:
#         features = [
#             'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
#             'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
#         ]
#         features = [
#                  'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
#                  'rouge1', 'rouge2', 'rouge3', 'rougeL', 
#             ]

#     # Auto-detect target column if not specified
#     # if not target_column:
#     if 'use_for_contamination' in df.columns:
#         target_column = 'use_for_contamination'
#     elif 'label' in df.columns:
#         target_column = 'label'
#     else:
#         raise ValueError("Target column not found. Provide a target column or use a DataFrame with 'use_for_contamination' or 'label'.")

#     # Prepare data
#     X = df[features]
#     y = df[target_column]

#     # Split with stratification
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_split, stratify=y, random_state=42
#     )

#     # Train model
#     # model = make_pipeline(
#     #     StandardScaler(),
#     #     LogisticRegression(random_state=42, max_iter=5000)
#     # )

#     # model = DecisionTreeClassifier(random_state=42, max_depth=35)

#    # You can tune hyperparameters here
#     base = DecisionTreeClassifier(max_depth=25)
#     model = AdaBoostClassifier(
#         estimator=base,
#         n_estimators=100,
#         learning_rate=0.1,
#         random_state=42
#     )

#     # model = make_pipeline(
#     #     StandardScaler(),
#     #     SVC(kernel='linear', C=1.0, random_state=42)
#     # )

#     model.fit(X_train, y_train)

#     # Define AdaBoost with DecisionTree as base learner
#     # model.fit(X_train, y_train)

#     # # Evaluate
#     # y_pred = model.predict(X_test)
#     # print(f"Classification Report (target: {target_column}):")
#     # print(classification_report(y_test, y_pred))

#     return model


# def cross_predict_f1_heatmap(df_dict, max_depth=19, test_split=0.2, features=None, target_column=None):
#     datasets = list(df_dict.keys())
#     f1_scores = pd.DataFrame(index=datasets, columns=datasets)

#     for train_name in datasets:
#         print(f"\nTraining on: {train_name}")
#         model = train_logistic_regression_with_report(df_dict[train_name], max_depth, features, target_column, test_split)
#         # train_df = df_dict[train_name]
#         # model = model_fn(train_df)  # 💡 Call model function with the current training dataframe

#         for test_name in datasets:
#             df_test = df_dict[test_name]

#             if features is None:
#                 features = [
#                     'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
#                     'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
#                 ]
#                 features = [
#                     'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
#                     'rouge1', 'rouge2', 'rouge3', 'rougeL'
#                 ]

#         # Auto-detect target column if not specified
#             # if not target_column:
#             if 'use_for_contamination' in df_test.columns:
#                 target_column = 'use_for_contamination'
#             elif 'label' in df_test.columns:
#                 target_column = 'label'
#             else:
#                 raise ValueError("Target column not found. Provide a target column or use a DataFrame with 'use_for_contamination' or 'label'.")

#             df_test = df_dict[test_name]

#             # Determine target
#             tgt_col = target_column or ('use_for_contamination' if 'use_for_contamination' in df_test.columns else 'label')
            
#             X = df_test[features]
#             y = df_test[target_column]

#             # Split with stratification
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=test_split, stratify=y, random_state=42
#             )

#             # Extract features and labels
#             # X_test = df_test[features] if features else df_test[[
#             #     'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
#             #     'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
#             # ]]
#             # y_test = df_test[tgt_col]

#             # Predict
#             y_pred = model.predict(X_test)

#             # Compute macro f1 score
#             f1 = f1_score(y_test, y_pred, average='macro')
#             f1_scores.loc[train_name, test_name] = f1

#     # Convert to float
#     f1_scores = f1_scores.astype(float)

#     # Plot heatmap
#     plt.figure(figsize=(10, 8))
#     # sns.heatmap(f1_scores.drop(columns=["avg_macro_f1"]), annot=True, fmt=".2f", cmap="YlGnBu")
#     sns.heatmap(f1_scores, annot=True, fmt=".2f", cmap="YlGnBu")
#     plt.title("Macro F1 Scores: Train (rows) vs Test (columns)")
#     plt.xlabel("Test Set")
#     plt.ylabel("Train Set")
#     plt.tight_layout()
#     plt.show()

#     return f1_scores

# res = cross_predict_f1_heatmap(dataframes)

# print(res)

