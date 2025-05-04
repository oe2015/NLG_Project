import torch
import json
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPT2Model, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import sys


# ===========================
# CONFIGURATION
# ===========================
MODEL_NAME = "gpt2-large"  # Change to "EleutherAI/gpt-neo-1.3B" for GPT-Neo

BATCH_SIZE = 2
EPOCHS = 3
MAX_LENGTH = 256  # Max token length for padding
SAVE_DIR = "gpt2-large-wikimia"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# ===========================
# LOAD WIKIMIA DATASET
# ===========================
print("Loading WikiMIA dataset...")

lengths = [32, 64, 128, 256]
all_data = []

# Load all text lengths (32, 64, 128, 256)
for length in lengths:
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
    
    # Filter to keep only samples with label == 1
    filtered_data = [sample for sample in dataset if sample["label"] == 1]
    
    all_data.extend(filtered_data)

print(f"Total Samples After Filtering: {len(all_data)}")
sys.stdout.flush() 

# Count label distribution (just for analysis)
label_counts = Counter([sample["label"] for sample in all_data])
print(f"Label Distribution (after filtering): {label_counts}")
sys.stdout.flush() 

# ===========================
# TOKENIZATION
# ===========================

print("Tokenizing dataset...")
sys.stdout.flush() 

# # Load tokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512, padding_side="left", add_eos_token=True)

tokenizer.pad_token = tokenizer.unk_token

# tokenizer.pad_token = tokenizer.eos_token  # Add pad token if missing
print("Tokenizer loaded.")
# tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
# tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have padding, use EOS token


# Tokenization function with proper padding and formatting
def tokenize_function(sample):
    # Create instruction prompt for better finetuning
    # prompt = create_prompt(sample["input"])
    return tokenizer(sample["input"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

# Tokenization function (NO padding or truncation)
# def tokenize_function(sample):
#     return tokenizer(sample["input"], truncation=True, max_length=512, padding="max_length")

# # Apply tokenization
tokenized_data = [tokenize_function(sample) for sample in all_data]

# # Check token lengths for all samples
# token_lengths = [len(sample["input_ids"]) for sample in tokenized_data]

# # Print statistics
# print(f"Min length: {min(token_lengths)}")
# print(f"Max length: {max(token_lengths)}")
# print(f"Average length: {sum(token_lengths) / len(token_lengths):.2f}")

# # Optionally: count how many samples have the same length
# from collections import Counter
# length_distribution = Counter(token_lengths)
# print("\nToken length distribution:")
# for length, count in sorted(length_distribution.items()):
#     print(f"Length {length}: {count} samples")
# ===========================
# CREATE DATASET & DATALOADER
# ===========================
class TextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"])
        attention_mask = torch.tensor(item["attention_mask"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # Next-token prediction uses input_ids as labels
        }

# Create dataset and DataLoader
dataset = TextDataset(tokenized_data)

# Load GPT-2 or GPT-Neo model
if "neo" in MODEL_NAME:
    model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)

    # model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
# model.resize_token_embeddings(len(tokenizer))


# ===========================
# TRAINING ARGUMENTS
# ===========================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",  # Disable evaluation
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_steps=500,
    num_train_epochs=15,
    report_to="none",  # Disable logging to WandB
    save_total_limit=2,  # Only keep the 2 most recent checkpoints
    learning_rate=1e-5,  # ✅ Set Learning Rate
)

# ===========================
# INITIALIZE TRAINER
# ===========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# ===========================
# START TRAINING
# ===========================
trainer.train()

print("Training complete!")

# Save trained model
trainer.save_model(SAVE_DIR)

# Save tokenizer
tokenizer.save_pretrained(SAVE_DIR)

