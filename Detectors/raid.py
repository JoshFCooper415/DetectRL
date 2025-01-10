import logging
import numpy as np
from datasets import load_dataset
from binoculars_detector import Binoculars
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import torch
import random
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.ERROR)

def get_roc_metrics(real_preds, sample_preds):
    if not real_preds or not sample_preds:
        raise ValueError("Empty predictions list")
        
    real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
    predicted_probs = real_preds + sample_preds
    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = [1 if prob >= optimal_threshold else 0 for prob in predicted_probs]
    
    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions)
    recall = recall_score(real_labels, predictions)
    f1 = f1_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions)
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
    
    return {
        "roc_auc": float(roc_auc),
        "optimal_threshold": float(optimal_threshold),
        "conf_matrix": conf_matrix.tolist(),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tpr_at_fpr_0_01": float(tpr_at_fpr_0_01)
    }

def truncate_text(text, max_length=2048):
    """Truncate text to ensure it doesn't exceed the model's maximum length."""
    if not isinstance(text, str):
        return str(text)
    words = text.split()
    if len(words) > max_length:
        return " ".join(words[:max_length])
    return text

def print_metrics(predictions, category=""):
    """Print statistics and metrics for a category."""
    print(f"\n{category} Statistics:")
    print(f"Human samples: {len(predictions['human'])}")
    print(f"LLM samples: {len(predictions['llm'])}")
    
    if len(predictions['human']) > 0 and len(predictions['llm']) > 0:
        try:
            metrics = get_roc_metrics(predictions['human'], predictions['llm'])
            print("\nMetrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
    else:
        print("Insufficient samples to calculate metrics")

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
MAX_SAMPLES = 10000

# Load dataset in streaming mode
logging.info("Loading RAID dataset test split in streaming mode...")
raid = load_dataset("liamdugan/raid", streaming=False)  # Changed to non-streaming initially to get size
total_size = len(raid['extra'])
logging.info(f"Total dataset size: {total_size}")

# Generate random indices
random_indices = random.sample(range(total_size), MAX_SAMPLES)

# Select those specific samples
test_stream = raid['extra'].select(random_indices)
# Initialize Binoculars
bino = Binoculars(mode="accuracy", max_token_observed=512)

# Store predictions for attacks and domains separately
attack_predictions = defaultdict(lambda: {'human': [], 'llm': []})
domain_predictions = defaultdict(lambda: {'human': [], 'llm': []})
all_predictions = {'human': [], 'llm': []}

# Track counts
attack_counts = defaultdict(lambda: {'human': 0, 'llm': 0})
domain_counts = defaultdict(lambda: {'human': 0, 'llm': 0})
errors = defaultdict(int)

# Process samples

logging.info(f"Processing {MAX_SAMPLES} random samples...")
sample_count = 0

try:
    with tqdm(total=MAX_SAMPLES) as pbar:
        for item in test_stream:
            sample_count += 1
            
            # Extract fields
            text = item.get('generation', '')
            model = str(item.get('model', '')).lower()
            attack = str(item.get('attack', '')).lower()
            domain = str(item.get('domain', '')).lower()
            
            # Validate model type
            if model == 'human':
                model_type = 'human'
            elif model and model != 'null':
                model_type = 'llm'
            else:
                errors['invalid_model'] += 1
                continue
            
            # Skip if no text
            if not text:
                errors['no_text'] += 1
                continue
            
            # Update counts
            if attack and attack != 'null':
                attack_counts[attack][model_type] += 1
            if domain and domain != 'null':
                domain_counts[domain][model_type] += 1
            
            try:
                # Truncate text if necessary
                truncated_text = truncate_text(text)
                
                # Compute score
                try:
                    score = bino.compute_score(truncated_text)
                    
                    if np.isfinite(score):
                        # Store overall prediction
                        all_predictions[model_type].append(score)
                        
                        # Store attack prediction if present
                        if attack and attack != 'null':
                            attack_predictions[attack][model_type].append(score)
                        
                        # Store domain prediction if present
                        if domain and domain != 'null':
                            domain_predictions[domain][model_type].append(score)
                    else:
                        errors['non_finite_score'] += 1
                        
                except IndexError as e:
                    errors['index_error'] += 1
                except torch.cuda.OutOfMemoryError:
                    errors['cuda_oom'] += 1
                    logging.warning("CUDA out of memory. Skipping sample.")
                    continue
                
            except Exception as e:
                errors['processing'] += 1
            
            pbar.update(1)
            
            # Log progress periodically
            if sample_count % 1000 == 0:
                logging.info(f"\nProcessed {sample_count} samples")
                
                logging.info("\nAttack distribution:")
                for attack, counts in attack_counts.items():
                    logging.info(f"  {attack}: Human={counts['human']}, LLM={counts['llm']}")
                
                logging.info("\nDomain distribution:")
                for domain, counts in domain_counts.items():
                    logging.info(f"  {domain}: Human={counts['human']}, LLM={counts['llm']}")
                
                logging.info("\nErrors:")
                for error_type, count in errors.items():
                    logging.info(f"  {error_type}: {count}")
            
            if sample_count >= MAX_SAMPLES:
                break

except KeyboardInterrupt:
    logging.info("Processing interrupted by user")

# Print final statistics
print("\n=== Processing Summary ===")
print(f"Total samples processed: {sample_count}")
print("\nErrors:")
for error_type, count in errors.items():
    print(f"  {error_type}: {count}")

# Print attack statistics
print("\n=== Attack Statistics ===")
total_attack_samples = sum(counts['human'] + counts['llm'] for counts in attack_counts.values())
for attack, counts in sorted(attack_counts.items(), key=lambda x: sum(x[1].values()), reverse=True):
    total = counts['human'] + counts['llm']
    percentage = (total / total_attack_samples) * 100 if total_attack_samples > 0 else 0
    print(f"\nAttack '{attack}':")
    print(f"  Total: {total} samples ({percentage:.1f}%)")
    print(f"  Human: {counts['human']}")
    print(f"  LLM: {counts['llm']}")

# Print domain statistics
print("\n=== Domain Statistics ===")
total_domain_samples = sum(counts['human'] + counts['llm'] for counts in domain_counts.values())
for domain, counts in sorted(domain_counts.items(), key=lambda x: sum(x[1].values()), reverse=True):
    total = counts['human'] + counts['llm']
    percentage = (total / total_domain_samples) * 100 if total_domain_samples > 0 else 0
    print(f"\nDomain '{domain}':")
    print(f"  Total: {total} samples ({percentage:.1f}%)")
    print(f"  Human: {counts['human']}")
    print(f"  LLM: {counts['llm']}")

# Print metrics
print("\n=== Metrics ===")

# Overall metrics
print_metrics(all_predictions, "Overall")

# Attack metrics
print("\n--- Attack Metrics ---")
for attack in attack_predictions:
    if sum(len(preds) for preds in attack_predictions[attack].values()) > 0:
        print_metrics(attack_predictions[attack], f"Attack: {attack}")

# Domain metrics
print("\n--- Domain Metrics ---")
for domain in domain_predictions:
    if sum(len(preds) for preds in domain_predictions[domain].values()) > 0:
        print_metrics(domain_predictions[domain], f"Domain: {domain}")