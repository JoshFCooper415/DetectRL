from binoculars_detector import Binoculars
import argparse
import numpy as np
import torch

def read_text_file(file_path):
    """Read text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def analyze_text(text_content, model_mode="low-fpr", debug=False):
    """Analyze text using Binoculars detector with additional error checking."""
    # Initialize Binoculars with default models
    detector = Binoculars(mode=model_mode)
    
    if debug:
        print("Text length:", len(text_content))
        print("Tokenizing text...")
    
    # Verify text is not empty
    if not text_content.strip():
        raise ValueError("Input text is empty")
    
    try:
        # Compute score first with error checking
        score = detector.compute_score(
            text_content,
            temperature=1.0,  # Using default temperature for stability
            top_k=None,      # Disable top-k for initial test
            top_p=None,      # Disable top-p for initial test
            repetition_penalty=1.0
        )
        
        # Check if score is valid
        if np.isnan(score):
            if debug:
                print("Warning: Score computation resulted in NaN")
            score = float('nan')
        
        # Get prediction
        prediction = detector.predict(
            text_content,
            temperature=1.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0
        )
        
        return prediction, score
        
    except Exception as e:
        if debug:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Analyze text using Binoculars AI detector')
    parser.add_argument('file_path', help='Path to the text file to analyze')
    parser.add_argument('--mode', choices=['low-fpr', 'accuracy'], 
                       default='low-fpr', help='Detection mode')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    try:
        # Read the text file
        text_content = read_text_file(args.file_path)
        
        if args.debug:
            print(f"Read {len(text_content)} characters from file")
        
        # Analyze the text
        prediction, score = analyze_text(text_content, args.mode, args.debug)
        
        # Print results
        print("\nAnalysis Results:")
        print("-" * 50)
        print(f"File: {args.file_path}")
        print(f"Mode: {args.mode}")
        print(f"Prediction: {prediction}")
        if np.isnan(score):
            print("Score: Unable to compute reliable score")
        else:
            print(f"Score: {score:.4f}")
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()