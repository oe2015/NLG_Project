import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
import nltk
nltk.download('punkt')
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
from bert_score import score
from sklearn.metrics import accuracy_score
import numpy as np
from bart import BARTScorer

# Setup
DEVICE = "cuda"
# model_path = "oe2015/gpt2-large-wikimia"
model_path = "oe2015/gpt2-wikimia-1epoch"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE)
model.eval()
bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')

# Load WikiMIA dataset
LENGTH = 256
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

smoothie = SmoothingFunction().method4
labels = []
combined_scores = []
output_file = open("contamination_results6.txt", "w", encoding="utf-8")

for i, sample in enumerate(dataset):
    label = sample["label"]
    article = sample["input"]
    
    # Skip too-short articles
    first_sentence = article.split(".")[0].strip()
    if not first_sentence or len(first_sentence.split()) < 4:
        continue  # Skip short or malformed samples
    
    # Split first sentence into input + reference
    words = first_sentence.split()
    midpoint = len(words) // 2
    prompt_text = " ".join(words[:midpoint])
    first_sentence_rest = " ".join(words[midpoint:])

    # Extract the rest of the article (after the first sentence)
    rest_of_article = article[len(first_sentence):].lstrip(". ")

    # Final reference text = second half of first sentence + rest of article
    full_reference_text = f"{first_sentence_rest} {rest_of_article}".strip()

    # Tokenize and generate
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(DEVICE)
    input_length = input_ids.shape[1]

    output = model.generate(
        input_ids,
        max_length=input_length + 500,  # generate a few more tokens
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    generated_tokens = generated_text.split()

    # Truncate reference to same number of tokens
    reference_tokens_all = full_reference_text.split()
    reference_tokens = reference_tokens_all[:len(generated_tokens)]
    reference_text = " ".join(reference_tokens)

    # Calculate BLEU
    reference_tokens = reference_text.split()
    generated_tokens = generated_text.split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothie)

    P, R, F1 = score([generated_text], [reference_text], lang="en", model_type="roberta-large")
    bert_f1_val = F1[0].item()

    rouge_scores = scorer.score(reference_text, generated_text)
    rouge_l_f1 = rouge_scores["rougeL"].fmeasure
    bart_score = bart_scorer.score([generated_text], [reference_text], batch_size=4)

    combined_score = bleu_score * rouge_l_f1 * bart_score[0]
    labels.append(label)
    combined_scores.append(combined_score)
    # Print and Write Output
    log = (
        f"[{i}] Label: {label} | Combined Score: {combined_score:.4f} | "
        f"BLEU: {bleu_score:.4f} | ROUGE-L: {rouge_l_f1:.4f} | BARTScore-F1: {bart_score[0]:.4f}\n"
        f"Prompt     : {prompt_text}\n"
        f"Reference  : {reference_text}\n"
        f"Generated  : {generated_text}\n"
        + "-" * 80 + "\n"
    )
    print(log)
    output_file.write(log)


best_acc = 0.0
best_thresh = 0.0
combined_scores = np.array(combined_scores)
labels = np.array(labels)

for thresh in np.linspace(-10, 10, 1000):
    preds = (combined_scores > thresh).astype(int)  # Predict contamination if score > thresh
    acc = accuracy_score(labels, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print("\nOptimal Threshold Found:", round(best_thresh, 4))
print("Best Accuracy Achieved :", round(best_acc, 4))