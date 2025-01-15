import logging
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict
from binoculars_detector import Binoculars

def analyze_length_correlations(filename, bino):
    """
    Analyze correlations between text length and detection metrics.
    
    Args:
        filename (str): Path to the test data JSON file
        bino: Initialized Binoculars detector
    """
    # Load the data
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Organize data by category
    lengths = defaultdict(list)
    scores = defaultdict(list)
    
    for item in data:
        text = item["text"]
        label = item["label"]
        
        # Calculate Binoculars score
        score = bino.compute_score(text)
        
        # Skip if score is not finite
        if not np.isfinite(score):
            continue
            
        # Calculate text length (in characters)
        text_length = len(text)
        
        lengths[label].append(text_length)
        scores[label].append(score)
    
    # Calculate correlations
    correlations = {}
    for label in ['human', 'llm']:
        if lengths[label]:  # Check if we have data for this label
            correlation, p_value = pearsonr(lengths[label], scores[label])
            correlations[label] = {
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': len(lengths[label])
            }
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot scatter for each category
    colors = {'human': 'blue', 'llm': 'red'}
    for label in ['human', 'llm']:
        if lengths[label]:
            plt.scatter(lengths[label], scores[label], 
                       alpha=0.5, 
                       label=f'{label} (r={correlations[label]["correlation"]:.3f}, n={correlations[label]["sample_size"]})',
                       c=colors[label])
    
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Binoculars Score')
    plt.title('Text Length vs Detection Score Correlation')
    plt.legend()
    
    # Save the plot
    plot_filename = filename.replace('.json', '_length_correlation.png')
    plt.savefig(plot_filename)
    plt.close()
    
    return correlations

def main(test_data_paths, tokens_seen=512):
    """
    Analyze multiple test files.
    
    Args:
        test_data_paths (str): Comma-separated list of test data file paths
        tokens_seen (int): Number of tokens seen by the model
    """
    # Initialize Binoculars
    bino = Binoculars(mode="accuracy", max_token_observed=tokens_seen)
    
    filenames = test_data_paths.split(",")
    
    for filename in filenames:
        logging.info(f"Analyzing correlations for {filename}")
        
        correlations = analyze_length_correlations(filename, bino)
        
        # Save correlation results
        results_filename = filename.replace('.json', '_length_correlations.json')
        with open(results_filename, 'w') as f:
            json.dump(correlations, f, indent=4)
            
        # Log results
        logging.info(f"Results for {filename}:")
        for label, stats in correlations.items():
            logging.info(f"{label.upper()}:")
            logging.info(f"  Correlation: {stats['correlation']:.3f}")
            logging.info(f"  P-value: {stats['p_value']:.3f}")
            logging.info(f"  Sample size: {stats['sample_size']}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                      help="Path to the test data. Could be several files with ','.")
    parser.add_argument("--tokens_seen", type=int, default=512,
                      help="Number of tokens seen by the model")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, 
                       format="%(asctime)s %(levelname)s %(message)s")
    
    main(args.test_data_path, args.tokens_seen)