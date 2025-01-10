import logging
import numpy as np
from datasets import load_dataset
from binoculars_detector import Binoculars
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import torch
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
    words = text.split()
    if len(words) > max_length:
        return " ".join(words[:max_length])
    return text

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load dataset in streaming mode
logging.info("Loading RAID dataset test split in streaming mode...")
raid = load_dataset("liamdugan/raid", streaming=True)
test_stream = raid['extra'].shuffle(seed=RANDOM_SEED, buffer_size=10000)

# Initialize Binoculars
bino = Binoculars(mode="accuracy", max_token_observed=512)

# Store predictions
predictions = {'human': [], 'llm': []}
errors = {'sequence_length': 0, 'processing': 0}

# Process samples
MAX_SAMPLES = 10000
logging.info(f"Processing {MAX_SAMPLES} random samples...")
sample_count = 0

try:
    with tqdm(total=MAX_SAMPLES) as pbar:
        for item in test_stream:
            text = item['generation']
            label = item['model']
            
            try:
                # Truncate text if necessary
                truncated_text = truncate_text(text)
                
                # Compute score with error handling
                try:
                    score = bino.compute_score(truncated_text)
                    
                    if np.isfinite(score):
                        if label == 'human':
                            predictions['human'].append(score)
                        else:
                            predictions['llm'].append(score)
                except IndexError as e:
                    errors['processing'] += 1
                    logging.debug(f"Processing error: {str(e)}")
                except torch.cuda.OutOfMemoryError:
                    logging.warning("CUDA out of memory. Skipping sample.")
                    continue
                
            except Exception as e:
                errors['processing'] += 1
                logging.debug(f"Unexpected error: {str(e)}")
            
            sample_count += 1
            pbar.update(1)
            
            # Log progress periodically
            if sample_count % 1000 == 0:
                logging.info(f"Processed {sample_count} samples. "
                            f"Valid predictions - Human: {len(predictions['human'])}, "
                            f"LLM: {len(predictions['llm'])}")
            
            if sample_count >= MAX_SAMPLES:
                break

except KeyboardInterrupt:
    logging.info("Processing interrupted by user")

# Final statistics
logging.info(f"Processing completed:")
logging.info(f"Total samples processed: {sample_count}")
logging.info(f"Valid predictions - Human: {len(predictions['human'])}, LLM: {len(predictions['llm'])}")
logging.info(f"Errors - Processing: {errors['processing']}")

# Calculate and display metrics if we have enough valid predictions
if len(predictions['human']) > 0 and len(predictions['llm']) > 0:
    try:
        metrics = get_roc_metrics(predictions['human'], predictions['llm'])
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
else:
    logging.error("Insufficient valid predictions to calculate metrics")