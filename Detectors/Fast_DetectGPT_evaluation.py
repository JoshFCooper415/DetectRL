import logging
import random
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from Fast_DetectGPT import get_text_crit
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def experiment(args):
    # load model
    logging.info(f"Loading reference model of type {args.reference_model}...")
    reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
    reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model)
    reference_model.eval()
    reference_model.cuda()

    scoring_tokenizer = AutoTokenizer.from_pretrained(args.scoring_model)
    scoring_model = AutoModelForCausalLM.from_pretrained(args.scoring_model)
    scoring_model.eval()
    scoring_model.cuda()

    model_config = {
        "reference_tokenizer": reference_tokenizer,
        "reference_model": reference_model,
        "scoring_tokenizer": scoring_tokenizer,
        "scoring_model": scoring_model,
    }

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        for item in tqdm(test_data):
            text = item["text"]
            label = item["label"]
            text_crit = get_text_crit(text, args, model_config)

            item['text_crit'] = text_crit

            if label == "human":
                predictions['human'].append(text_crit)
            elif label == "llm":
                predictions['llm'].append(text_crit)
            else:
                raise ValueError(f"Unknown label {label}")

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy = get_roc_metrics(predictions['human'],
                                                                                                   predictions['llm'])

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

        logging.info(f"{result}")
        with open(filename.split(".json")[0] + "_Fast_DetectGPT_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(filename.split(".json")[0] + "_Fast_DetectGPT_result.json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--reference_model', type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--scoring_model', type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
