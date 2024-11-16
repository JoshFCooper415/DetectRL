import asyncio
import logging
import os
import random
import time

import numpy as np
import torch
import tqdm
import argparse
import json
from metrics import get_roc_metrics
from bart_score import BARTScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import transformers
import torch


def chat(prompt, model_id):
    model_id = "NousResearch/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][-1]


def experiment(args):
    # load model

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        if os.path.exists(filename.split(".json")[0] + "_revise_data.json"):
            test_data = json.load(open(filename.split(".json")[0] + "_revise_data.json", "r"))
        else:
            test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        human_chatcands = []
        human_chatrefs = []
        llm_chatcands = []
        llm_chatrefs = []
        predictions = {'human': [], 'llm': []}

        bartscorer = BARTScorer(device='cuda:0', checkpoint="facebook/bart-large-cnn")
        for item in tqdm.tqdm(test_data):
            text = item["text"]
            label = item["label"]
            # if "revised_text" in item and item["revised_text"] != "":
            #     pass
            # else:
            #     try:
            #         item["revised_text"] = chat(f"Revise the following text: {text}", "NousResearch/Meta-Llama-3-8B-Instruct")
            #     except:
            #         item["revised_text"] = ""
            #         pass
            #         # item["revised_text"] = chat([{"role": "user", "prompt": f"Revise the following text: {text}"}], "ChatGPT-16k")
            #         # logging.info("Error in ChatGPT, sleep for 60s...")
            #         # time.sleep(60)
            # print(item["revised_text"])
            # print("="*50)

            if item["revise_text"] == "":
                pass
            else:
                # calculate scores
                if label == "human":
                    human_chatcands.append(item["revise_text"])
                    human_chatrefs.append(text)
                    sim_score = bartscorer.score([item["revise_text"]], [text])
                    predictions['human'].append(sim_score)
                    item["revise_score"] = sim_score
                elif label == "llm":
                    llm_chatcands.append(item["revise_text"])
                    llm_chatrefs.append(text)
                    sim_score = bartscorer.score([item["revise_text"]], [text])
                    predictions['llm'].append(sim_score)
                    item["revise_score"] = sim_score

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        bartscorer.model.to('cpu')

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
        with open(filename.split(".json")[0] + "_revise_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(filename.split(".json")[0] + "_revise_result.json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.", default="Varying_Length/cross_length_80_test.json")
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
