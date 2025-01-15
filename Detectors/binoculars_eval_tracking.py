import logging
import random
import torch
import tqdm
import argparse
import json
from binoculars_detector import Binoculars
from collections import defaultdict
from metrics import get_roc_metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, \
    accuracy_score, f1_score
import numpy as np

def analyze_correlations(test_data, predictions):
    """Analyze correlations between data_type/llm_type and prediction accuracy."""
    data_type_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    llm_type_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Using the optimal threshold from the original metrics calculation
    threshold = get_roc_metrics(
        [p for p, item in zip(predictions, test_data) if item['label'] == 'human'],
        [p for p, item in zip(predictions, test_data) if item['label'] == 'llm']
    )[1]

    for item, pred in zip(test_data, predictions):
        if not np.isfinite(pred):
            continue
            
        # Determine if prediction was correct
        is_correct = (pred >= threshold and item['label'] == 'llm') or \
                    (pred < threshold and item['label'] == 'human')
        
        # Track by data_type
        if 'data_type' in item:
            data_type_metrics[item['data_type']]['total'] += 1
            if is_correct:
                data_type_metrics[item['data_type']]['correct'] += 1
        
        # Track by llm_type
        if 'llm_type' in item and item['label'] == 'llm':
            llm_type_metrics[item['llm_type']]['total'] += 1
            if is_correct:
                llm_type_metrics[item['llm_type']]['correct'] += 1
    
    # Calculate accuracy for each category
    data_type_accuracy = {
        data_type: {'accuracy': metrics['correct'] / metrics['total'],
                   'total_samples': metrics['total']}
        for data_type, metrics in data_type_metrics.items()
    }
    
    llm_type_accuracy = {
        llm_type: {'accuracy': metrics['correct'] / metrics['total'],
                  'total_samples': metrics['total']}
        for llm_type, metrics in llm_type_metrics.items()
    }
    
    return {
        'data_type_accuracy': data_type_accuracy,
        'llm_type_accuracy': llm_type_accuracy
    }

def experiment(args):
    bino = Binoculars(mode="accuracy", max_token_observed=args.tokens_seen)
    
    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        test_data = json.load(open(filename, "r"))
        
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Store all predictions and items together
        all_predictions = []
        for item in tqdm.tqdm(test_data):
            text = item["text"]
            score = bino.compute_score(text)
            item["bino_score"] = score
            all_predictions.append(score)
        
        # Get original metrics
        human_preds = [i["bino_score"] for i in test_data if i["label"] == "human"]
        llm_preds = [i["bino_score"] for i in test_data if i["label"] == "llm"]
        
        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr_0_01 = get_roc_metrics(
            [p for p in human_preds if np.isfinite(p)],
            [p for p in llm_preds if np.isfinite(p)]
        )
        
        # Get correlation analysis
        correlation_results = analyze_correlations(test_data, all_predictions)
        
        # Combine results
        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr_0_01,
            "correlations": correlation_results
        }
        
        logging.info(f"Overall metrics: {result}")
        logging.info("\nData type correlations:")
        for dtype, metrics in correlation_results['data_type_accuracy'].items():
            logging.info(f"{dtype}: {metrics['accuracy']:.3f} accuracy ({metrics['total_samples']} samples)")
        
        logging.info("\nLLM type correlations:")
        for llm_type, metrics in correlation_results['llm_type_accuracy'].items():
            logging.info(f"{llm_type}: {metrics['accuracy']:.3f} accuracy ({metrics['total_samples']} samples)")
        
        # Save results
        with open(filename.split(".json")[0] + "_bino_data.json", "w") as f:
            json.dump(test_data, f, indent=4)
        
        with open(filename.split(".json")[0] + "_bino_detailed_result.json", "w") as f:
            json.dump(result, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                       help="Path to the test data. could be several files with ','. "
                            "Note that the data should have been perturbed.")
    parser.add_argument("--tokens_seen", type=int, default=512, help="Number of tokens seen by the model")
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()
    
    experiment(args)