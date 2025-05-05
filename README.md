# Data Contamination Detection in LLMs

This project investigates automatic detection of data contamination in large language models (LLMs) using black-box similarity metrics and ensemble classification. It was conducted as part of the final project for NLP804 at MBZUAI.

## 🧠 Project Overview

Large language models often unintentionally memorize training data, leading to inflated evaluation metrics and potential privacy or copyright issues. This project evaluates whether uncertainty-based methods can distinguish memorized (contaminated) text from novel (clean) text, and proposes a robust alternative based on similarity scores and classification.

Two datasets released after 2022 are used:

- **Mintaka**: A QA dataset with 20k examples. Models are fine-tuned using supervised instruction tuning (SFT) on 14k contaminated samples, with 6k left out as clean. 1,000 examples from each split are used for generation and scoring.
- **WikiMIA**: A paragraph-level dataset with 1,650 samples. Models are fine-tuned using next-word prediction on 861 contaminated examples (label 1). The remaining 789 clean samples (label 0) are held out. Generations and similarity scoring are done over the full set.

A vector of 8 similarity metrics is extracted for each generated sample (BLEU, ROUGE-1/2/L, BERTScore, and NLI-similarity), and an AdaBoost classifier is trained to distinguish contaminated from clean generations. The classifier outperforms log-likelihood, Min-k%, and UQ-based baselines on multiple configurations of GPT-2 and GPT-Neo.

---

## 🛠️ Setup Instructions

1. **Create and activate a Conda environment**:

   ```bash
   conda create -n contamination-detect python=3.10
   conda activate contamination-detect
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### ▶️ Run Min-k% and Log-likelihood Baselines:

```bash
python detect-pretrain-code/src/run.py
```

### 🧠 Train GPT-2 or GPT-Neo on WikiMIA:

```bash
python train_wiki.py
```

### 📊 Generate Combined Similarity Scores:

```bash
python generate_combined_score.py
```

### 📈 Plot Combined Score Histograms:

```bash
python plot_combined_score.py
```

---

## 📂 Project Structure

```
nlg_project/
│
├── detect-pretrain-code/         # Baseline code for Min-k and log-likelihood
├── score_outputs/                # Output figures and contamination results
│   ├── contamination_results*.txt
│   ├── combined_score_histogram*.png
│   └── ...
├── generate_combined_score.py    # Script to compute similarity scores
├── plot_combined_score.py        # Script to visualize combined scores
├── train_wiki.py                 # Training script for WikiMIA dataset
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📌 Notes

- Classifier training uses an 80/20 stratified split to ensure balanced representation of clean and contaminated samples.
- Evaluation includes both in-domain and cross-domain transfer settings across models and datasets.
- All experiments are performed using GPT-2-Large (774M) and GPT-Neo-1.3B.

---

## 📧 Contact

Omar El Herraoui
MBZUAI
[omar.el-herraoui@mbzuai.ac.ae](mailto:omar.el-herraoui@mbzuai.ac.ae)
