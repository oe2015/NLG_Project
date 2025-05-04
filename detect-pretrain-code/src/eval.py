import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random
from sklearn.metrics import f1_score, roc_curve


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plot data 
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    # fpr, tpr, _ = roc_curve(x, -score)
    fpr, tpr, thresholds = roc_curve(x, score)

    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc, thresholds


# plot data 
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, thresholds = roc_curve(x, -np.array(score))
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc, thresholds

def sweep(score, x, reverse=False):
    score = np.array(score)
    x = np.array(x)

    # Replace NaNs with a very bad value
    nan_mask = np.isnan(score)
    if nan_mask.any():
        num_nans = nan_mask.sum()
        print(f"[WARN] {num_nans} NaN scores found — replacing with random noise in range [-1.0, 0.0]")
        score[nan_mask] = np.random.uniform(-100, 100, size=num_nans)

    fpr, tpr, thresholds = roc_curve(x, -np.array(score))
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)

    return fpr, tpr, auc(fpr, tpr), acc, thresholds


# def sweep(score, x, reverse=False):

#     if reverse:
#         score = -np.array(score)  # Negate Min-K Prob so that higher values indicate contamination

#     fpr, tpr, _ = roc_curve(x, np.array(score))  # No negation for P(True)
#     acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    
#     return fpr, tpr, auc(fpr, tpr), acc


# def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
#     """
#     Generate the ROC curves by using ntest models as test models and the rest to train.
#     """
#     fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

#     reverse = "Min" in metric_name  # Reverse only for Min-K Prob metrics

#     fpr, tpr, auc_score, acc = sweep(prediction, answers, reverse=reverse)

#     low = tpr[np.where(fpr<.05)[0][-1]]
#     # bp()
#     print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n'%(legend, auc,acc, low))

#     metric_text = ''
#     if metric == 'auc':
#         metric_text = 'auc=%.3f'%auc
#     elif metric == 'acc':
#         metric_text = 'acc=%.3f'%acc

#     plt.plot(fpr, tpr, label=legend+metric_text)
#     return legend, auc,acc, low

def do_plot(prediction, answers, metric_name, output_dir=None):
    """
    Generate ROC curves for given metric and save the plot.
    
    Args:
        prediction (list): Scores from the model.
        answers (list): True labels (0 or 1).
        metric_name (str): The name of the metric.
        output_dir (str): Directory to save the plot.
    """
    reverse = "Min" in metric_name  # Reverse only for Min-K Prob metrics

    fpr, tpr, auc_score, acc, thresholds = sweep(prediction, answers)

    low_tpr = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0.0

    y_true = np.array(answers)
    y_score = -np.array(prediction) if reverse else np.array(prediction)

    # Best F1 score computation across thresholds
    best_f1 = 0
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1

    # print(f'Attack {metric_name}   AUC {auc_score:.4f}, Accuracy {acc:.4f}, TPR@5%FPR of {low_tpr:.4f}\n')
    print(f'Attack {metric_name}   AUC {auc(fpr, tpr):.4f}, Accuracy {acc:.4f}, Best Macro F1 {best_f1:.4f}, TPR@5%FPR of {low_tpr:.4f}\n')
    with open("output_results.txt", "a") as f:
        f.write(f'Attack {metric_name}   AUC {auc(fpr, tpr):.4f}, Accuracy {acc:.4f}, Best Macro F1 {best_f1:.4f}, TPR@5%FPR of {low_tpr:.4f}\n')

    plt.plot(fpr, tpr, label=f"{metric_name} (AUC={auc_score:.3f})")
    return metric_name, auc_score, acc, low_tpr



# def fig_fpr_tpr(all_output, output_dir):
#     print("output_dir", output_dir)
#     answers = []
#     metric2predictions = defaultdict(list)
#     for ex in all_output:
#         answers.append(ex["label"])
#         for metric in ex["pred"].keys():
#             if ("raw" in metric) and ("clf" not in metric):
#                 continue
#             metric2predictions[metric].append(ex["pred"][metric])
    
#     plt.figure(figsize=(4,3))
#     with open(f"{output_dir}/auc.txt", "w") as f:
#         for metric, predictions in metric2predictions.items():
#             legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
#             f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n'%(legend, auc, acc, low))

#     plt.semilogx()
#     plt.semilogy()
#     plt.xlim(1e-5,1)
#     plt.ylim(1e-5,1)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.plot([0, 1], [0, 1], ls='--', color='gray')
#     plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
#     plt.legend(fontsize=8)
#     plt.savefig(f"{output_dir}/auc.png")

def fig_fpr_tpr(all_output, output_dir):
    """
    Evaluate different metrics by computing their ROC curves.

    Args:
        all_output (list): List of processed examples with `pred` dict.
        output_dir (str): Directory to save the results.
    """
    print("Evaluating ROC Curves...")
    
    answers = []
    metric2predictions = defaultdict(list)

    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])
    
    plt.figure(figsize=(4, 3))
    
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            legend, auc_score, acc, low_tpr = do_plot(predictions, answers, metric, output_dir)
            f.write(f'{legend}   AUC {auc_score:.4f}, Accuracy {acc:.4f}, TPR@0.1%FPR of {low_tpr:.4f}\n')

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")



def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data