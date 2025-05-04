import re
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from cliffs_delta import cliffs_delta
import pandas as pd

# File path
out_file = "slurm3.out"  # replace with your actual path

# Initialize empty lists
labels, mink_probs, ccp_uncertainties = [], [], []

# Read and parse the file
# with open(out_file, "r") as f:
#     lines = f.readlines()
#     assert len(lines) % 3 == 0, "Each sample should be 3 lines: label, prob, ccp_uncertainty"

#     for i in range(0, len(lines), 3):
#         label = int(lines[i].strip())
#         mink_prob = float(lines[i + 1].strip())
#         ccp_line = lines[i + 2].strip()
#         ccp_uncertainty = float(ccp_line.split("ccp_uncertainty:")[-1].strip())

#         labels.append(label)
#         mink_probs.append(mink_prob)
#         ccp_uncertainties.append(ccp_uncertainty)

# Convert to numpy arrays

gpt2_mintaka_solo = pd.read_parquet('score_outputs/generated_texts_gpt2_large_mintaka_solo_scores.parquet')

print(gpt2_mintaka_solo.columns)
bert_score = gpt2_mintaka_solo["rouge1"]
labels = gpt2_mintaka_solo["use_for_contamination"]

X = np.column_stack([bert_score])  # Features

# X = np.array(mink_probs)
y = np.array(labels)     


class_0_scores = X[y == 0]
class_1_scores = X[y == 1]

ks_stat, ks_p_value = ks_2samp(class_0_scores, class_1_scores)
cliffs_delta, res = cliffs_delta(class_0_scores, class_1_scores)

print(ks_p_value)
# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(class_0_scores, bins=30, alpha=0.6, label="Label 0", color='blue', edgecolor='black')
plt.hist(class_1_scores, bins=30, alpha=0.6, label="Label 1", color='green', edgecolor='black')

plt.axvline(x=np.median(class_0_scores), color='blue', linestyle='--', linewidth=1)
plt.axvline(x=np.median(class_1_scores), color='green', linestyle='--', linewidth=1)


# plt.title(f"KS Test: D={ks_stat[0]:.4f}, p={ks_p_value[0]:.4e}")
plt.title(f"Cliff's Delta: {cliffs_delta:.3f} ({res})")
plt.xlabel("Mink Prob")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot instead of showing it
plt.show()