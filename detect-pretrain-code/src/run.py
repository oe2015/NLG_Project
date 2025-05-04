import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
# import openai
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from options import Options
from eval import *
import pandas as pd

# Import lm-polygraph for CCP uncertainty
# from lm_polygraph.utils.model import WhiteboxModel
# from lm_polygraph.estimators import ClaimConditionedProbability, SemanticEntropy
# from lm_polygraph.utils.manager import estimate_uncertainty

# Import lm-polygraph for CCP uncertainty
# from lm_polygraph.stat_calculators.infer_causal_lm_calculator import InferCausalLMCalculator

# from lm_polygraph.stat_calculators.greedy_alternatives_nli import GreedyAlternativesNLICalculator
# from lm_polygraph.estimators.claim_conditioned_probability import ClaimConditionedProbability
# from lm_polygraph.utils.deberta import Deberta
# from lm_polygraph.model_adapters import WhiteboxModelBasic
# from torch.utils.data import DataLoader

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
# from lm_polygraph import estimate_uncertainty
# from lm_polygraph.estimators import MaximumTokenProbability, LexicalSimilarity, SemanticEntropy, PointwiseMutualInformation, EigValLaplacian, TokenSAR, PTrue, MaximumSequenceProbability, Perplexity, LabelProb, SentenceSAR

# deberta_nli_model = Deberta(device="cuda")
# deberta_nli_model.setup()

generation_config = GenerationConfig.from_pretrained("gpt2-large")
args_generate = {"generation_config" : generation_config,
                 "max_new_tokens": 50}

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# def load_nli_model():
#     model_name = "microsoft/deberta-large-mnli"
#     model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return model, tokenizer

# def classify_nli(nli_model, nli_tokenizer, premise, hypothesis):
#     inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True).to("cuda")
#     with torch.no_grad():
#         logits = nli_model(**inputs).logits
#     label = logits.argmax().item()
#     if label == 2:
#         return "entail"
#     elif label == 0:
#         return "contra"
#     else:
#         return "neutral"

def calculate_CCP(text, model, tokenizer, nli_model, nli_tokenizer, top_k=5):
    input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # shape: (1, seq_len, vocab)

    log_ccp_scores = []

    for j in range(1, input_ids.size(1)):
        prev_ids = input_ids[0][:j]
        true_token_id = input_ids[0][j].item()

        # Predict distribution over next token
        token_logits = logits[0, j-1]
        log_probs = F.log_softmax(token_logits, dim=-1)
        topk_log_probs, topk_token_ids = torch.topk(log_probs, top_k)

        entail_logprobs = []
        contra_logprobs = []

        true_text = tokenizer.decode(input_ids[0][:j+1])  # x₁:j
        for k, token_id in enumerate(topk_token_ids):
            alt_ids = torch.cat([prev_ids, token_id.view(1)])
            alt_text = tokenizer.decode(alt_ids)

            # Forward and backward NLI
            fwd = classify_nli(nli_model, nli_tokenizer, true_text, alt_text)
            bwd = classify_nli(nli_model, nli_tokenizer, alt_text, true_text)

            def combine_nli(forward, backward):
                if forward == backward:
                    return forward
                if all(x in [forward, backward] for x in ["entail", "contra"]):
                    return "neutral"
                return forward if forward in ["entail", "contra"] else backward

            combined_nli = combine_nli(fwd, bwd)

            if combined_nli == "entail":
                entail_logprobs.append(topk_log_probs[k].item())
            elif combined_nli == "contra":
                contra_logprobs.append(topk_log_probs[k].item())

        if len(entail_logprobs) == 0:
            log_ccp_j = 0.0  # Treat as full uncertainty
        else:
            # log CCP_j = logsumexp(entail) - logsumexp(entail+contra)
            log_entail = np.logaddexp.reduce(entail_logprobs)
            log_total = np.logaddexp.reduce(entail_logprobs + contra_logprobs)
            log_ccp_j = log_entail - log_total

        log_ccp_scores.append(log_ccp_j)

    # Final CCP = 1 - exp(sum log CCP_j)
    final_ccp = 1 - np.exp(sum(log_ccp_scores))
    return final_ccp

def perturb_input_simple(text, fraction=0.2):
    words = text.split()
    num_to_remove = int(len(words) * fraction)
    if num_to_remove < 1:
        return text
    indices = random.sample(range(1, len(words) - 1), k=min(num_to_remove, len(words)-2))
    for idx in sorted(indices, reverse=True):
        del words[idx]
    return ' '.join(words)


def calculate_reverse_rank_likelihood(model, tokenizer, input_text, device):
    """
    Calculates the average reverse rank score of tokens in the input.
    Higher score = more likely model has seen it before.
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)
        log_probs = log_softmax(logits, dim=-1)

    ranks = []
    for i in range(1, input_ids.size(1)):
        token_id = input_ids[0, i].item()
        probs = log_probs[0, i - 1]  # prob for predicting token i
        # Rank of the actual token (lower rank = more probable)
        rank = (probs > probs[token_id]).sum().item() + 1  # +1 to make rank 1-indexed
        ranks.append(rank)

    # Normalize the rank by vocab size (optional) or take reciprocal
    avg_reciprocal_rank = np.mean([1.0 / r for r in ranks])
    return avg_reciprocal_rank


# # Make sure WordNet is downloaded (only needs to be done once)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# def get_synonym(word):
#     """Get a synonym for a word using WordNet."""
#     synonyms = wordnet.synsets(word)
#     lemmas = set(lemma.name().replace('_', ' ') for syn in synonyms for lemma in syn.lemmas())
#     lemmas.discard(word)  # Remove the original word
#     return random.choice(list(lemmas)) if lemmas else word

# def perturb_input_simple(text, fraction=0.2):
#     """
#     Replace a fraction of words in the input text with their synonyms.

#     Args:
#         text (str): Original input text.
#         fraction (float): Fraction of words to replace with synonyms.

#     Returns:
#         str: Perturbed text with synonym replacements.
#     """
#     words = text.split()
#     num_to_replace = int(len(words) * fraction)

#     if num_to_replace < 1:
#         return text

#     indices = random.sample(range(len(words)), k=min(num_to_replace, len(words)))

#     for idx in indices:
#         words[idx] = get_synonym(words[idx])

#     return ' '.join(words)



# def perturb_input_simple(text, fraction=0.2):
#     """
#     Shuffle a fraction of words in the input text to perturb it.
#     Keeps the overall content, but changes word order slightly.

#     Args:
#         text (str): Original input text.
#         fraction (float): Fraction of words to shuffle.

#     Returns:
#         str: Perturbed (shuffled) text.
#     """
#     words = text.split()
#     num_to_shuffle = int(len(words) * fraction)

#     if num_to_shuffle < 2:
#         return text  # Need at least 2 words to shuffle

#     # Choose random indices to shuffle (excluding first/last word if you want to preserve structure)
#     indices = random.sample(range(1, len(words)-1), k=min(num_to_shuffle, len(words)-2))

#     # Extract the selected words, shuffle them, then put them back
#     selected_words = [words[i] for i in indices]
#     random.shuffle(selected_words)
#     for idx, shuffled_word in zip(indices, selected_words):
#         words[idx] = shuffled_word

#     return ' '.join(words)

# def perturb_input_simple(text, fraction=0.2):
#     words = text.split()
#     if len(words) < 4:
#         return text
    
#     num_to_perturb = max(1, int(len(words) * fraction))
#     indices = list(range(1, len(words)-1))  # avoid first and last word
#     random.shuffle(indices)

#     perturbed = words[:]
#     count = 0

#     for idx in indices:
#         if count >= num_to_perturb:
#             break
#         word = words[idx]

#         if word.lower() in stop_words:
#             continue

#         op = random.choice(['delete', 'synonym', 'shuffle'])

#         op = 'delete'
#         # op = 'synonym'

#         if op == 'delete':
#             perturbed[idx] = ''
#         elif op == 'synonym':
#             perturbed[idx] = get_synonym(word)
#         elif op == 'shuffle' and idx < len(words) - 2:
#             perturbed[idx], perturbed[idx+1] = perturbed[idx+1], perturbed[idx]
        
#         count += 1

#     return ' '.join([w for w in perturbed if w.strip()])


def load_model(name1, name2):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)
        tokenizer1.pad_token = tokenizer1.eos_token  

    model2 = 0
    tokenizer2 = 0
    # if "davinci" in name2:
    #     model2 = None
    #     tokenizer2 = None
    # else:
    #     model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto')
    #     model2.eval()
    #     tokenizer2 = AutoTokenizer.from_pretrained(name2)
    #     tokenizer2.pad_token = tokenizer2.eos_token  

    return model1, model2, tokenizer1, tokenizer2

def calculatePerplexity_gpt3(prompt, modelname):
    prompt = prompt.replace('\x00','')
    responses = None
    # Put your API key here
    openai.api_key = "YOUR_API_KEY" # YOUR_API_KEY
    while responses is None:
        try:
            responses = openai.Completion.create(
                        engine=modelname, 
                        prompt=prompt,
                        max_tokens=0,
                        temperature=1.0,
                        logprobs=5,
                        echo=True)
        except openai.error.InvalidRequestError:
            print("too long for openai API")
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob)

     
def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    # perturbed_sent = perturb_input_simple(sentence)

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

# def calculate_p_true(model, tokenizer, input_text, device):
#     """
#     Calculate P(True) for a generated text by prompting the model for self-evaluation.
    
#     Args:
#         model: Language model instance.
#         tokenizer: Tokenizer for the model.
#         input_text (str): The generated text to be evaluated.
#         device: Model's device.

#     Returns:
#         float: Model's estimated probability of correctness (P(True)).
#     """
#     # Improve the prompt with explicit instruction
#     evaluation_prompt = (
#         f"Generated Text:\n{input_text}\n\n"
#         "Question: Have you seen a text like this before?\n"
#         "Answer (0.0 to 1.0):"
#     )

#     # Tokenize input
#     input_ids = tokenizer.encode(evaluation_prompt, return_tensors="pt").to(device)

#     # Generate response (forcing output to be short)
#     with torch.no_grad():
#         output_ids = model.generate(input_ids, max_length=input_ids.shape[1] + 3, 
#                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)

#     # Decode the response
#     evaluation_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     # Extract only the numerical part
#     generated_output = evaluation_response.replace(evaluation_prompt, "").strip()

#     print(f"Raw model response: {generated_output}")

#     try:
#         # Try extracting a float from the model response
#         p_true = float(generated_output.split()[0])  # Get first token
#     except ValueError:
#         # If extraction fails, return a default low-confidence score
#         p_true = 0.0

#     return p_true

import torch
import numpy as np

def calculate_p_true(model, tokenizer, input_text, device):
    """
    Estimate P(True) using log-likelihood of the text under the model.
    
    Args:
        model: Pretrained causal language model (e.g., GPT-2).
        tokenizer: Corresponding tokenizer.
        input_text (str): The generated text whose probability needs to be estimated.
        device: Device on which the model is running.

    Returns:
        float: Estimated probability of correctness (P(True)).
    """

    # Tokenize text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Compute log-likelihood of the tokens
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross-entropy loss

    # Convert loss to probability (lower loss = higher P(True))
    log_likelihood = -loss.item()  # Negative because loss is negative log likelihood
    p_true = np.exp(log_likelihood)  # Convert log-likelihood to probability

    return p_true



def calculate_semantic_uncertainty(input_text, model, tokenizer, nli_model, nli_tokenizer, top_k=5, device='cuda'):
    """
    Calculate semantic uncertainty for a given input text.

    Args:
        input_text (str): The input text to evaluate.
        model: Pretrained language model (e.g., GPT-2).
        tokenizer: Tokenizer corresponding to the language model.
        nli_model: Pretrained NLI model for semantic equivalence assessment.
        nli_tokenizer: Tokenizer corresponding to the NLI model.
        top_k (int): Number of top alternative tokens to consider for each token in the input.
        device (str): Device to run the models on ('cuda' or 'cpu').

    Returns:
        float: Semantic uncertainty score.
    """
    model.to(device)
    model.eval()
    
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    input_length = input_ids.size(1)
    
    # Get logits from the model
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # shape: (batch_size, sequence_length, vocab_size)
    
    semantic_equivalence_counts = []
    
    # Iterate over each token in the input (excluding special tokens)
    for i in range(1, input_length - 1):
        original_token_id = input_ids[0, i].item()
        
        # Get the logits for the current token position
        token_logits = logits[0, i]
        
        # Compute probabilities
        token_probs = F.softmax(token_logits, dim=-1)
        
        # Get top-k alternative tokens
        top_k_probs, top_k_indices = torch.topk(token_probs, top_k)
        
        # Initialize count of semantically equivalent alternatives
        equivalent_count = 0
        
        # Original sentence up to the current token
        original_prefix = tokenizer.decode(input_ids[0, :i], skip_special_tokens=True)
        original_suffix = tokenizer.decode(input_ids[0, i+1:], skip_special_tokens=True)
        
        # Iterate over top-k alternatives
        for alt_token_id in top_k_indices:
            alt_token_id = alt_token_id.item()
            
            # Skip if the alternative token is the same as the original token
            if alt_token_id == original_token_id:
                continue
            
            # Construct the alternative sentence
            alt_token = tokenizer.decode([alt_token_id], skip_special_tokens=True)
            alt_sentence = f"{original_prefix} {alt_token} {original_suffix}".strip()
            
            # Use NLI model to check semantic equivalence
            nli_result = classify_nli(nli_model, nli_tokenizer, input_text, alt_sentence)
            
            if nli_result == "entail":
                equivalent_count += 1
        
        # Calculate the proportion of semantically equivalent alternatives
        semantic_equivalence_ratio = equivalent_count / top_k
        semantic_equivalence_counts.append(semantic_equivalence_ratio)
    
    # Compute the average semantic equivalence ratio across all tokens
    average_semantic_equivalence = np.mean(semantic_equivalence_counts)
    
    # Semantic uncertainty is the complement of semantic equivalence
    semantic_uncertainty = 1 - average_semantic_equivalence
    
    return semantic_uncertainty


import torch
import numpy as np
from torch.nn.functional import log_softmax

def calculate_contamination_score(model, tokenizer, input_text, device, perturb_func, max_new_tokens=50):
    """
    Estimate contamination likelihood by measuring how much the model prefers its own output 
    (generated from original input) even when conditioned on a perturbed version of the input.

    Args:
        model: Language model (e.g., GPT-2).
        tokenizer: Corresponding tokenizer.
        input_text (str): Original input text.
        device: Model device ('cuda' or 'cpu').
        perturb_func (callable): Function to perturb input text (e.g., synonym replacement, word drop).
        max_new_tokens (int): Number of tokens to generate from original input.

    Returns:
        float: Average log-probability of original generation under perturbed input.
    """
    model.eval()

    # Step 1: Generate model's own completion from original input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    print(input_ids.shape)
    # with torch.no_grad():
    #     generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Step 2: Perturb the input
    perturbed_input = perturb_func(input_text)
    perturbed_ids = tokenizer.encode(perturbed_input, return_tensors="pt").to(device)

    print(perturbed_ids.shape)

    # Step 3: Evaluate P(generated_text | perturbed_input)
    # target_ids = tokenizer.encode(generated_text, return_tensors="pt").to(device)

    # with torch.no_grad():
    #     outputs = model(input_ids, labels=input_ids)
    #     logits = outputs.logits
    #     log_probs = log_softmax(logits, dim=-1)

    # # Step 4: Compute average log-probability of generated text under perturbed input
    # token_log_probs = []
    # for i in range(1, input_ids.size(1)):
    #     token_log_probs.append(log_probs[0, i - 1, input_ids[0, i]].item())
    
    # avg_log_prob = np.mean(token_log_probs)

    # with torch.no_grad():
    #     outputs = model(perturbed_ids, labels=perturbed_ids)
    #     logits = outputs.logits
    #     log_probs = log_softmax(logits, dim=-1)

    # # Step 4: Compute average log-probability of generated text under perturbed input
    # token_log_probs = []
    # for i in range(1, perturbed_ids.size(1)):
    #     token_log_probs.append(log_probs[0, i - 1, perturbed_ids[0, i]].item())
    
    # avg_log_prob1 = np.mean(token_log_probs)

    # score = avg_log_prob - avg_log_prob1

    # Get logits for original input
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits_orig = outputs.logits
        log_probs_orig = log_softmax(logits_orig, dim=-1)

    # Get logits for perturbed input
    with torch.no_grad():
        outputs = model(perturbed_ids, labels=perturbed_ids)
        logits_pert = outputs.logits
        log_probs_pert = log_softmax(logits_pert, dim=-1)

    # Align lengths
    min_len = min(input_ids.shape[1], perturbed_ids.shape[1])

    # Compare log-probs only where the *target token is the same*
    diff_logprobs = []
    for i in range(1, min_len):  # skip first token
        target_token_orig = input_ids[0, i].item()
        target_token_pert = perturbed_ids[0, i].item()

        if target_token_orig == target_token_pert:
            # Subtract log prob of target token under both contexts
            logp_orig = log_probs_orig[0, i - 1, target_token_orig].item()
            logp_pert = log_probs_pert[0, i - 1, target_token_pert].item()
            diff_logprobs.append(logp_orig - logp_pert)

    if diff_logprobs:
        score = np.mean(diff_logprobs)
    else:
        score = 0.0  # fallback if no matching tokens

    return score


# def calculate_contamination_score(model, tokenizer, input_text, device, perturb_func):
#     """
#     Measure contamination by computing how well the model predicts original tokens given perturbed context.
#     """
#     model.eval()
#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     perturbed_text = perturb_func(input_text)
#     perturbed_ids = tokenizer.encode(perturbed_text, return_tensors="pt").to(device)

#     # Step 1: Feed perturbed input to model
#     with torch.no_grad():
#         outputs = model(perturbed_ids)
#         logits = outputs.logits  # shape: (1, seq_len, vocab)

#     # Step 2: Align length
#     max_tokens = min(logits.shape[1], input_ids.shape[1] - 1)  # avoid index overflow

#     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
#     token_log_probs = []
#     for i in range(max_tokens):
#         target_token = input_ids[0, i + 1]  # original next token
#         token_log_probs.append(log_probs[0, i, target_token].item())

#     contamination_score = np.mean(token_log_probs)
#     return contamination_score


def inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2, polygraph_model, ue_method):
    pred = {}

    if "davinci" in modelname1:
        p1, all_prob, p1_likelihood = calculatePerplexity_gpt3(text, modelname1) 
        p_lower, _, p_lower_likelihood = calculatePerplexity_gpt3(text.lower(), modelname1)
    else:
        p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
        p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    # if "davinci" in modelname2:
    #     p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity_gpt3(text, modelname2)
    # else:
    #     p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
   
   # ppl
    pred["ppl"] = p1
    # Ratio of log ppl of large and small models
    # pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood


    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1)/zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    print(pred[f"Min_{0.2*100}% Prob"])

    # Convert input into chat template format (if needed)
    # input_text = text

    # ccp_uncertainty = calculate_CCP(input_text, model1, tokenizer1, nli_model, nli_tokenizer)

    # pred["ccp_uncertainty"] = ccp_uncertainty
    # print(f"ccp_uncertainty: {ccp_uncertainty}")

    # Calculate P(True) self-evaluation uncertainty

    # uncertainty = calculate_semantic_uncertainty(input_text, model1, tokenizer1, nli_model, nli_tokenizer)

    # p_true_uncertainty = calculate_p_true(model1, tokenizer1, input_text, model1.device)

    # p_true_uncertainty2 = calculate_p_true(model2, tokenizer2, input_text, model2.device)

    # uncertainty = p_true_uncertainty2 - p_true_uncertainty

    # pred["P(True) Uncertainty"] = uncertainty

    # print(f"P(True) Uncertainty: {uncertainty}")


    # cont_score = calculate_contamination_score(
    # model=model1,
    # tokenizer=tokenizer1,
    # input_text=input_text,
    # device=model1.device,
    # perturb_func=perturb_input_simple,
    # max_new_tokens=200
    # )

    # pred["ContaminationScore"] = cont_score
    # print(f"ContaminationScore: {cont_score}")

    # uncertainty = estimate_uncertainty(polygraph_model, ue_method, input_text=input_text)

    # print(uncertainty)
    # Initialize calculators
    # calc_infer_llm = InferCausalLMCalculator(tokenize=False)
    # calc_nli = GreedyAlternativesNLICalculator(nli_model=deberta_nli_model)

    # # Prepare batch
    # encoded = tokenizer1([input_text], padding=True, return_tensors="pt").to(model1.device)
    # deps = {"model_inputs": encoded}

    # # Step 1: Extract Greedy Tokens & Alternatives
    # deps.update(calc_infer_llm(deps, texts=[input_text], model=polygraph_model, args_generate=args_generate))

    # # Step 2: Compute NLI for Alternative Tokens
    # deps.update(calc_nli(deps, texts=None, model=polygraph_model))

    # # Step 3: Compute CCP Uncertainty
    # ccp_estimator = ClaimConditionedProbability()
    # ccp_uncertainty = ccp_estimator(deps)

    # pred["CCP Uncertainty"] = ccp_uncertainty

    # print(pred["CCP Uncertainty"])

    ex["pred"] = pred
    return ex


def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, col_name, modelname1, modelname2, polygraph_model, ue_method):
    print(f"all data size: {len(test_data)}")
    all_output = []
    # test_data = test_data
    # test_data = random.sample(test_data, min(10, len(test_data)))  # Ensures it doesn't break if data has <10 samples

    for ex in tqdm(test_data): 
        text = ex[col_name]
        input_text = text
        print(ex["label"])
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, input_text, ex, modelname1, modelname2, polygraph_model, ue_method)
        all_output.append(new_ex)
    return all_output

from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    print("starting")
 # === Load a single dataset per domain ===
    datasets = {
        "mintaka": pd.read_parquet('score_outputs/generated_texts_gpt2_large_mintaka_solo_scores.parquet'),
        "wikimia": pd.read_parquet('score_outputs/generated_texts_gpt2_large_wikimia_scores.parquet')
    }

    # === Model-to-dataset mapping ===
    model_dataset_map = {

        "oe2015/gpt2-large-wikimia": ["wikimia"],
        "oe2015/gptneo_1.3b_wikimia": ["wikimia"],

        "KareemElzeky/gpt2-large-SFT-15epoch": ["mintaka"],
        "KareemElzeky/gpt-neo-SFT-15epoch": ["mintaka"],

        "DaniilOr/sft-gpt2-large-15batch": ["wikimia", "mintaka"],
        "DaniilOr/sft-gptneo-15batch": ["wikimia", "mintaka"]
    }

    # === Setup
    args = Options()
    args = args.parser.parse_args()
    args.output_dir = f"{args.output_dir}/{args.target_model}_{args.ref_model}/{args.key_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    for model_name, applicable_datasets in model_dataset_map.items():
        print(f"\n🔍 Evaluating model: {model_name}")

        model1, model2, tokenizer1, tokenizer2 = load_model(model_name, model_name)

        for domain in applicable_datasets:
            log_line = f"→ Evaluating dataset '{domain}' on model '{model_name}'"
            print(log_line)

            with open("output_results.txt", "a") as f:
                f.write(log_line + "\n")      
                      
            df = datasets[domain]

            # Select column mapping based on domain
            col_name = "question" if domain == "mintaka" else "input"
            label_col = "use_for_contamination" if domain == "mintaka" else "label"

            # Convert rows to standard format
            all_data = [{
                # col_name: row[col_name] + row.get("extracted_answer", ""),  # answerText only exists in Mintaka
                col_name: row[col_name] + row.get("answerText", ""),  # answerText only exists in Mintaka
                "label": int(row[label_col]),
                **row.to_dict()
            } for _, row in df.iterrows()]

            # Extract labels for stratified split
            labels = [ex["label"] for ex in all_data]

            # Split using sklearn to maintain stratification
            train_data, test_data = train_test_split(
                all_data,
                test_size=0.2,
                stratify=labels,
                random_state=42
            )

            # Polygraph settings
            polygraph_model = 0
            ue_method = 0

            # Evaluate
            print(f"all data size: {len(test_data)}")
            all_output = []
            for ex in tqdm(test_data): 
                text = ex[col_name]
                input_text = text
                # print(ex["label"])
                new_ex = inference(model1, model2, tokenizer1, tokenizer2, input_text, ex, model_name, model_name, polygraph_model, ue_method)
                all_output.append(new_ex)

            # all_output = evaluate_data(
            #     test_data, model1, model2,
            #     tokenizer1, tokenizer2,
            #     args.key_name, args.target_model, model_name,
            #     polygraph_model, ue_method
            # )

            # Plot FPR/TPR
            fig_fpr_tpr(all_output, args.output_dir)


    # if "jsonl" in args.data:
    #     data = load_jsonl(f"{args.data}")
    # else: # load data from huggingface
    #     dataset = load_dataset(args.data, split=f"WikiMIA_length{args.length}")
    #     print(dataset)
    #     data = convert_huggingface_data_to_list_dic(dataset)

    # polygraph_model = WhiteboxModelBasic(model1, tokenizer1, args_generate)
    # polygraph_model = 0
    # ue_method = 0

    # # Evaluate
    # all_output = evaluate_data(data, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model, polygraph_model, ue_method)
    # # all_output = evaluate_data(data, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model)
    # # print(all_output)
    # fig_fpr_tpr(all_output, args.output_dir)

