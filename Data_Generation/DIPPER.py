import argparse
import json
import logging
import random
import time
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
from nltk.tokenize import sent_tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_data(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


def spilt_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    prompt = sentences[0]
    input_text = " ".join(sentences[1:])
    return prompt, input_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # choose the human key according to the dataset
    domain_human = {
        "arxiv": "abstract",
        "xsum": "document",
        "writing_prompt": "story",
        "yelp_review": "content",
    }
    human_key = domain_human[args.input_file.split("_2800")[0]]

    dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")

    data = read_data(args.input_file)

    human_error_list = []
    llm_error_list = []

    for article in tqdm(data):

        abstract = article[human_key]
        direct_prompt = article['direct_prompt']

        try:
            if "paraphrase_dipper_human" in article.keys() and article['paraphrase_dipper_human'] != "":
                pass
            else:
                logging.info(f"before dipper paraphrase:{abstract}")
                prompt, input_text = spilt_paragraph(abstract)
                paraphrase_dipper_human = dp.paraphrase(input_text, lex_diversity=40, order_diversity=40, prefix=prompt,
                                                        do_sample=False, max_length=2048)

                logging.info(f"after dipper paraphrase:{paraphrase_dipper_human}")
                if len(paraphrase_dipper_human) == 0:
                    human_error_list.append(article)
                article['paraphrase_dipper_human'] = paraphrase_dipper_human
                time.sleep(random.randint(1, 4))

            with open(args.output_file, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)

            if "paraphrase_dipper_llm" in article.keys() and article['paraphrase_dipper_llm'] != "":
                pass
            else:
                logging.info(f"before dipper paraphrase:{direct_prompt}")
                prompt, input_text = spilt_paragraph(direct_prompt)
                paraphrase_dipper_llm = dp.paraphrase(input_text, lex_diversity=40, order_diversity=40, prefix=prompt,
                                                      do_sample=False, max_length=2048)

                logging.info(f"after dipper paraphrase:{paraphrase_dipper_llm}")
                if len(paraphrase_dipper_llm) == 0:
                    llm_error_list.append(article)
                article['paraphrase_dipper_llm'] = paraphrase_dipper_llm
                time.sleep(random.randint(1, 4))

            with open(args.output_file, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)

        except Exception as e:
            print(e)
            pass

    logging.info(f"human error number:{len(human_error_list)}")
    logging.info(f"llm error number:{len(llm_error_list)}")

    with open("human_error_" + args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    with open("llm_error_" + args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
