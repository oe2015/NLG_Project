import re
import matplotlib.pyplot as plt

# Read from contamination result file
file_path = 'contamination_results.txt'

# Parse the file
labels = []
combined_scores = []

with open(file_path, 'r') as f:
    for line in f:
        match = re.search(r'Label:\s*(\d)\s*\|\s*Combined Score:\s*(-?\d+\.\d+)', line)
        if match:
            label = int(match.group(1))
            score = float(match.group(2))
            labels.append(label)
            combined_scores.append(score)

# Separate by label
scores_0 = [combined_scores[i] for i in range(len(labels)) if labels[i] == 0]
scores_1 = [combined_scores[i] for i in range(len(labels)) if labels[i] == 1]

# Plot histograms
plt.figure(figsize=(10, 6))

plt.hist(scores_0, bins=30, alpha=0.7, label="Label 0 (Not Contaminated)")
plt.hist(scores_1, bins=30, alpha=0.7, label="Label 1 (Contaminated)")

plt.xlabel("Combined Score")
plt.ylabel("Frequency")
plt.title("Distribution of Combined Score by Label")
plt.legend()

# Save to file
plt.tight_layout()
plt.show()
