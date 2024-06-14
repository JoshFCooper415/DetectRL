import json
import logging
import math
import os
import random
import re
import nltk
from tqdm import tqdm
from data_mixing import get_llm_mixing, get_human_mixing, get_human_centered_mixing, get_llm_centered_mixing


def read_data(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def check_data(data, file_name):
    domain_human = {
        "arxiv": "abstract",
        "xsum": "document",
        "writing_prompt": "story",
        "yelp_review": "content",

    }

    check_keys = ['direct_prompt',
                  # 'llm_type',
                  # prompt
                  'prompt_few_shot',
                  'prompt_ICO',
                  # paraphrase
                  'paraphrase_polish_human',
                  'paraphrase_polish_llm',
                  'paraphrase_dipper_human',
                  'paraphrase_dipper_llm',
                  'paraphrase_back_translation_llm',
                  'paraphrase_back_translation_human',
                  # perturbation
                  'perturbation_word_human',
                  'perturbation_word_llm',
                  'perturbation_character_human',
                  'perturbation_character_llm',
                  'perturbation_sent_human',
                  'perturbation_sent_llm'
                  ]

    for key in domain_human.keys():
        if key in file_name:
            check_keys.append(domain_human[key])

    logging.info(f"Data check for {file_name} start!")
    logging.info(f"Keys for check: {check_keys}")

    for check_key in check_keys:
        key_loss = 0
        empty_num = 0
        finished_num = 0
        for item in tqdm(data):
            if check_key not in item.keys():
                # logging.info(f"{check_key} not in article keys! item {list(item.keys())[0]}:{item[list(item.keys())[0]]}")
                key_loss += 1
            elif item[check_key] == "":
                logging.info(f"{check_key} is empty! item {list(item.keys())[0]}:{item[list(item.keys())[0]]}")
                empty_num += 1
            else:
                if len(get_sentences_in_paragraph(item[check_key])) < 4:
                    pass
                    # logging.info(
                    #     f"{check_key} has less than 4 sentences! item {list(item.keys())[0]}:{item[list(item.keys())[0]]}")
                # assert len(get_sentences_in_paragraph(item[check_key])) >= 4
                finished_num += 1

        logging.info(
            f"Data check for {file_name}-{check_key} finished!, total: {len(data)}, finished: {finished_num}, loss: {key_loss}, empty: {empty_num}")


def merge_data(data1, data2):
    for domain in data2.keys():
        for article_index in tqdm(range(len(data2[domain]))):
            for key in data2[domain][article_index].keys():
                if key not in data1[domain][article_index].keys():
                    data1[domain][article_index][key] = data2[domain][article_index][key]
    return data1


def format_process(data):
    for item in tqdm(data):
        for key in item.keys():
            if isinstance(item[key], str):
                item[key] = item[key].strip()
                item[key] = re.sub(r"\n", "", item[key])
    return data


def get_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences


def extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num):
    human = []
    llm = []
    data = [item for item in data if item["llm_type"] in llm_types]
    for item in data:
        for key in human_keys:
            if key in item.keys() and item[key] != "":
                human.append({
                    "text": item[key],
                    "label": "human",
                    "data_type": key,
                    "llm_type": item["llm_type"]
                })
        for key in llm_keys:
            if key in item.keys() and item[key] != "":
                llm.append({
                    "text": item[key],
                    "label": "llm",
                    "data_type": key,
                    "llm_type": item["llm_type"]
                })

    import random
    random.seed(2023)

    # human
    grouped_human_data = {f"{llm_type}_{human_key}": [] for llm_type in llm_types for human_key in human_keys}
    for item in human:
        if f"{item['llm_type']}_{item['data_type']}" in grouped_human_data.keys():
            grouped_human_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

    train_human = []
    test_human = []
    for data_type, items in grouped_human_data.items():
        sampled_indices = random.sample(range(len(items)), math.ceil(llm_test_num / (len(human_keys) * len(llm_types))))
        sampled_human = [items[i] for i in sampled_indices]
        sampled_indices.sort(reverse=True)
        for index in sampled_indices:
            items.pop(index)
        test_human += sampled_human
        train_human += items

    # llm
    grouped_llm_data = {f"{llm_type}_{llm_key}": [] for llm_type in llm_types for llm_key in llm_keys}
    for item in llm:
        if f"{item['llm_type']}_{item['data_type']}" in grouped_llm_data.keys():
            grouped_llm_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

    train_llm = []
    test_llm = []
    for key, items in grouped_llm_data.items():
        sampled_indices = random.sample(range(len(items)), math.ceil(llm_test_num / (len(llm_keys) * len(llm_types))))
        sampled_llm = [items[i] for i in sampled_indices]
        sampled_indices.sort(reverse=True)
        for index in sampled_indices:
            items.pop(index)
        test_llm += sampled_llm
        train_llm += items

    print(len(train_human), len(test_human), len(train_llm), len(test_llm))

    train_data = train_human + train_llm
    test_data = test_human + test_llm

    return train_data, test_data


def save_json(data, scenarios, subset, file_name):
    data_path = os.path.join(os.path.dirname(os.getcwd()), scenarios, subset, file_name)
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))
    data = format_process(data)
    with open(data_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
        logging.info(f"total {len(data)} items,"
                     f"saved in {data_path}")


if __name__ == '__main__':
    llm_types = ["ChatGPT", "Google-PaLM", "Claude-instant", "Llama-2-70b"]
    domains = ["arxiv", "xsum", "writing_prompt", "yelp_review"]

    domain_human = {
        "arxiv": "abstract",
        "xsum": "document",
        "writing_prompt": "story",
        "yelp_review": "content",
    }

    # data check and format process
    for domain in domains:
        human_key = domain_human[domain]
        file_name = f"{domain}_2800.json"
        data = read_data(file_name)
        check_data(data, file_name)
        data = format_process(data)
        with open(file_name, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)

    # prepare data
    """
    ood, including multi domains and multi llms.
    """
    # multi domains
    for domain in domains:
        multi_domains_train_data = []
        multi_domains_test_data = []
        human_key = domain_human[domain]
        file_name = f"{domain}_2800.json"
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt", "prompt_few_shot", "prompt_ICO",
                    'paraphrase_dipper_llm', 'paraphrase_back_translation_llm', 'paraphrase_polish_llm',
                    'perturbation_character_llm', 'perturbation_word_llm', 'perturbation_sent_llm']
        train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=1000)
        multi_domains_train_data += train_data
        multi_domains_test_data += test_data

        save_json(multi_domains_train_data, "Benchmark", "Multi_Domain", f"multi_domains_{domain}_train.json")
        save_json(multi_domains_test_data, "Benchmark", "Multi_Domain", f"multi_domains_{domain}_test.json")

    # multi llms
    for llm in llm_types:
        multi_llms_train_data = []
        multi_llms_test_data = []
        for domain in domains:
            human_key = domain_human[domain]
            file_name = f"processed_{domain}_2800.json"
            data = read_data(file_name)
            human_keys = [human_key]
            llm_keys = ["direct_prompt", "prompt_few_shot", "prompt_ICO",
                        'paraphrase_dipper_llm', 'paraphrase_back_translation_llm', 'paraphrase_polish_llm',
                        'perturbation_character_llm', 'perturbation_word_llm', 'perturbation_sent_llm']
            train_data, test_data = extract_train_test(data, human_keys, llm_keys, [llm], llm_test_num=250)
            multi_llms_train_data += train_data
            multi_llms_test_data += test_data

        save_json(multi_llms_train_data, "Benchmark", "Multi_LLM", f"multi_llms_{llm}_train.json")
        save_json(multi_llms_test_data, "Benchmark", "Multi_LLM", f"multi_llms_{llm}_test.json")

    # direct prompt baseline
    direct_prompt_train_data = []
    direct_prompt_test_data = []
    for domain in domains:
        human_key = domain_human[domain]
        file_name = f"processed_{domain}_2800.json"
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt"]
        train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=250)
        direct_prompt_train_data += train_data
        direct_prompt_test_data += test_data

    save_json(direct_prompt_train_data, "Benchmark", "Direct_Prompt", f"direct_prompt_train.json")
    save_json(direct_prompt_test_data, "Benchmark", "Direct_Prompt", f"direct_prompt_test.json")

    # """
    # attack, including Prompt, Adversarial, Paraphrase and Data_Mixing.
    # """
    # Prompt
    keys_dict = {
        'prompt_few_shot': ['prompt_few_shot'],
        'prompt_ICO': ['prompt_ICO'],
        'prompt_attacks': ['prompt_few_shot', 'prompt_ICO'],
    }
    for key in keys_dict.keys():
        prompt_attack_train_data = []
        prompt_attack_test_data = []
        for domain in domains:
            human_key = domain_human[domain]
            file_name = f"processed_{domain}_2800.json"
            data = read_data(file_name)
            human_keys = [human_key]
            llm_keys = keys_dict[key]
            train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=250)
            prompt_attack_train_data += train_data
            prompt_attack_test_data += test_data

        save_json(prompt_attack_train_data, "Benchmark", "Prompt_Attacks", f"{key}_train.json")
        save_json(prompt_attack_test_data, "Benchmark", "Prompt_Attacks", f"{key}_test.json")

    # Adversarial
    keys_dict = {
        'perturbation_character_llm': ['perturbation_character_llm'],
        'perturbation_word_llm': ['perturbation_word_llm'],
        'perturbation_sent_llm': ['perturbation_sent_llm'],
        'perturbation_attacks': ['perturbation_character_llm', 'perturbation_word_llm', 'perturbation_sent_llm'],
    }
    for key in keys_dict.keys():
        perturbation_attack_train_data = []
        perturbation_attack_test_data = []
        for domain in domains:
            human_key = domain_human[domain]
            file_name = f"processed_{domain}_2800.json"
            data = read_data(file_name)
            human_keys = [human_key]
            llm_keys = keys_dict[key]
            train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=250)
            perturbation_attack_train_data += train_data
            perturbation_attack_test_data += test_data

        save_json(perturbation_attack_train_data, "Benchmark", "Perturbation_Attacks", f"{key}_train.json")
        save_json(perturbation_attack_test_data, "Benchmark", "Perturbation_Attacks", f"{key}_test.json")

    # Paraphrase
    keys_dict = {
        'paraphrase_dipper_llm': ['paraphrase_dipper_llm'],
        'paraphrase_back_translation_llm': ['paraphrase_back_translation_llm'],
        'paraphrase_polish_llm': ['paraphrase_polish_llm'],
        'paraphrase_attacks': ['paraphrase_dipper_llm', 'paraphrase_back_translation_llm', 'paraphrase_polish_llm'],
    }
    for key in keys_dict.keys():
        paraphrase_attack_train_data = []
        paraphrase_attack_test_data = []
        for domain in domains:
            human_key = domain_human[domain]
            file_name = f"processed_{domain}_2800.json"
            data = read_data(file_name)
            human_keys = [human_key]
            # llm_keys = ['paraphrase_dipper_llm', 'paraphrase_back_translation_llm', 'paraphrase_polish_llm']
            llm_keys = keys_dict[key]
            train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=250)
            paraphrase_attack_train_data += train_data
            paraphrase_attack_test_data += test_data

        save_json(paraphrase_attack_train_data, "Benchmark", "Paraphrase_Attacks", f"{key}_train.json")
        save_json(paraphrase_attack_test_data, "Benchmark", "Paraphrase_Attacks", f"{key}_test.json")

    # Model Data_Mixing
    train_data, test_data = get_llm_mixing(domains, llm_types, domain_human)
    save_json(train_data, "Benchmark", "Data_Mixing", f"multi_llm_mixing_train.json")
    save_json(test_data, "Benchmark", "Data_Mixing", f"multi_llm_mixing_test.json")

    # LLM Centered Data_Mixing
    train_data, test_data = get_llm_centered_mixing(domains, llm_types, domain_human)
    save_json(train_data, "Benchmark", "Data_Mixing", f"llm_centered_mixing_train.json")
    save_json(test_data, "Benchmark", "Data_Mixing", f"llm_centered_mixing_test.json")

    # Data Data_Mixing
    train_llm_mixing_data, test_llm_mixing_data = get_llm_mixing(domains, llm_types, domain_human)
    train_llm_mixing_data_human = train_llm_mixing_data[:len(train_llm_mixing_data) // 2]
    train_llm_mixing_data_llm = train_llm_mixing_data[len(train_llm_mixing_data) // 2:]
    test_llm_mixing_data_human = test_llm_mixing_data[:len(test_llm_mixing_data) // 2]
    test_llm_mixing_data_llm = test_llm_mixing_data[len(test_llm_mixing_data) // 2:]

    train_llm_centered_mixing_data, test_llm_centered_mixing_data = get_llm_centered_mixing(domains, llm_types,
                                                                                         domain_human)
    train_llm_centered_mixing_data_human = train_llm_centered_mixing_data[:len(train_llm_centered_mixing_data) // 2]
    train_llm_centered_mixing_data_llm = train_llm_centered_mixing_data[len(train_llm_centered_mixing_data) // 2:]
    test_llm_centered_mixing_data_human = test_llm_centered_mixing_data[:len(test_llm_centered_mixing_data) // 2]
    test_llm_centered_mixing_data_llm = test_llm_centered_mixing_data[len(test_llm_centered_mixing_data) // 2:]

    train_data = train_llm_mixing_data_human + \
                 train_llm_mixing_data_llm + \
                 train_llm_centered_mixing_data_human + \
                 train_llm_centered_mixing_data_llm + \
                 test_llm_mixing_data_human[int(len(test_llm_mixing_data_human) // 2):] + \
                 test_llm_mixing_data_llm[int(len(test_llm_mixing_data_llm) // 2):] + \
                 test_llm_centered_mixing_data_human[int(len(test_llm_centered_mixing_data_human) // 2):] + \
                 test_llm_centered_mixing_data_llm[int(len(test_llm_centered_mixing_data_llm) // 2):]

    test_data = test_llm_mixing_data_human[:int(len(test_llm_mixing_data_human) // 2)] + \
                test_llm_mixing_data_llm[:int(len(test_llm_mixing_data_llm) // 2)] + \
                test_llm_centered_mixing_data_human[:int(len(test_llm_centered_mixing_data_human) // 2)] + \
                test_llm_centered_mixing_data_llm[:int(len(test_llm_centered_mixing_data_llm) // 2)]

    print(len(train_data))
    print(len(test_data))

    save_json(train_data, "Benchmark", "Data_Mixing", f"data_mixing_attacks_train.json")
    save_json(test_data, "Benchmark", "Data_Mixing", f"data_mixing_attacks_test.json")

    # Cross Length
    sentence1_train_data = []
    sentence1_test_data = []

    sentence2_train_data = []
    sentence2_test_data = []

    sentence3_train_data = []
    sentence3_test_data = []

    sentence4_train_data = []
    sentence4_test_data = []

    multi_domains_test_data = []

    data_distribution = {
        40: [],
        80: [],
        120: [],
        160: [],
        200: [],
        240: [],
        280: [],
        320: [],
        360: [],
    }

    random_data_distribution = {
        40: [],
        80: [],
        120: [],
        160: [],
        200: [],
        240: [],
        280: [],
        320: [],
        360: [],
    }
    sentences_data = []

    for domain in domains:
        human_key = domain_human[domain]
        file_name = f"processed_{domain}_2800.json"
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt", "prompt_few_shot", "prompt_ICO",
                    'paraphrase_dipper_llm', 'paraphrase_back_translation_llm', 'paraphrase_polish_llm',
                    'perturbation_character_llm', 'perturbation_word_llm', 'perturbation_sent_llm']
        train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=250)

        for item in tqdm(train_data + test_data):
            texts = nltk.sent_tokenize(item["text"])

            for i in range(len(texts)):
                for j in range(len(texts) - i):
                    text = " ".join(texts[i:i + j])
                    sentences_data.append({"text": text, "data_type": item['data_type'] , "label" : item['label'], "text_length": len(text.split())})

        print(len(sentences_data))

    for item in sentences_data:
        for i in range(len(data_distribution.keys())):
            if i == 0:
                if 0 < item["text_length"] < 40:
                    data_distribution[40].append(item)
            else:
                if list(data_distribution.keys())[i - 1] < item["text_length"] < list(data_distribution.keys())[i]:
                    data_distribution[list(data_distribution.keys())[i]].append(item)

    random.seed(2023)

    for key in data_distribution.keys():
        if key < 1000:
            human = [j for j in data_distribution[key] if j["label"] == 'human']
            llm = [j for j in data_distribution[key] if j["label"] == 'llm']
            human_sample = random.sample(human, 450)
            llm_sample = []
            for data_type in ["direct_prompt", "prompt_few_shot", "prompt_ICO",
                    'paraphrase_dipper_llm', 'paraphrase_back_translation_llm', 'paraphrase_polish_llm',
                    'perturbation_character_llm', 'perturbation_word_llm', 'perturbation_sent_llm']:
                llm_sample.extend(random.sample([j for j in llm if j["data_type"] == data_type], 50))
            random_data_distribution[key] = human_sample + llm_sample
            print(f"{key}:{len(data_distribution[key])}")

    for key in random_data_distribution:
        save_json(random_data_distribution[key], "Benchmark", "CrossLengths", f"multi_length_{key}_test.json")

    # Adversarial Human
    keys_dict = {
        'perturbation_character_human': ['perturbation_character_human'],
        'perturbation_word_human': ['perturbation_word_human'],
        'perturbation_sent_llm': ['perturbation_sent_human'],
        'perturbation_attacks': ['perturbation_character_human', 'perturbation_word_human',
                                'perturbation_sent_human'],
    }
    for key in keys_dict.keys():
        perturbation_attack_train_data = []
        perturbation_attack_test_data = []
        for domain in domains:
            file_name = f"processed_{domain}_2800.json"
            data = read_data(file_name)
            human_keys = keys_dict[key]
            llm_keys = ["direct_prompt"]
            train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=250)
            perturbation_attack_train_data += train_data
            perturbation_attack_test_data += test_data

        save_json(perturbation_attack_train_data, "Benchmark", "Perturbation_Attacks_Human", f"{key}_train.json")
        save_json(perturbation_attack_test_data, "Benchmark", "Perturbation_Attacks_Human", f"{key}_test.json")


    # Paraphrase Human
    keys_dict = {
        'paraphrase_dipper_llm': ['paraphrase_dipper_human'],
        'paraphrase_back_translation_llm': ['paraphrase_back_translation_human'],
        'paraphrase_polish_llm': ['paraphrase_polish_human'],
        'paraphrase_attacks': ['paraphrase_dipper_llm', 'paraphrase_back_translation_human', 'paraphrase_polish_human'],
    }
    for key in keys_dict.keys():
        paraphrase_attack_train_data = []
        paraphrase_attack_test_data = []
        for domain in domains:
            file_name = f"processed_{domain}_2800.json"
            data = read_data(file_name)
            human_keys = keys_dict[key]
            # llm_keys = ['paraphrase_dipper_llm', 'paraphrase_back_translation_llm', 'paraphrase_polish_llm']
            llm_keys = ["direct_prompt"]
            train_data, test_data = extract_train_test(data, human_keys, llm_keys, llm_types, llm_test_num=250)
            paraphrase_attack_train_data += train_data
            paraphrase_attack_test_data += test_data

        save_json(paraphrase_attack_train_data, "Benchmark", "Paraphrase_Attacks_Human", f"{key}_train.json")
        save_json(paraphrase_attack_test_data, "Benchmark", "Paraphrase_Attacks_Human", f"{key}_test.json")

    # Human Data_Mixing
    train_data, test_data = get_human_mixing(domains, llm_types, domain_human)
    save_json(train_data, "Benchmark", "Data_Mixing_Human", f"multi_human_mixing_train.json")
    save_json(test_data, "Benchmark", "Data_Mixing_Human", f"multi_human_mixing_test.json")
    #
    # Human Centered Data_Mixing
    train_data, test_data = get_human_centered_mixing(domains, llm_types, domain_human)
    save_json(train_data, "Benchmark", "Data_Mixing_Human", f"human_centered_mixing_train.json")
    save_json(test_data, "Benchmark", "Data_Mixing_Human", f"human_centered_mixing_test.json")

    # Data Data_Mixing
    train_human_mixing_data, test_human_mixing_data = get_human_mixing(domains, llm_types, domain_human)
    train_human_mixing_data_human = train_human_mixing_data[:len(train_human_mixing_data) // 2]
    train_human_mixing_data_llm = train_human_mixing_data[len(train_human_mixing_data) // 2:]
    test_human_mixing_data_human = test_human_mixing_data[:len(test_human_mixing_data) // 2]
    test_human_mixing_data_llm = test_human_mixing_data[len(test_human_mixing_data) // 2:]

    train_human_centered_mixing_data, test_human_centered_mixing_data = get_human_centered_mixing(domains, llm_types,
                                                                                         domain_human)
    train_human_centered_mixing_data_human = train_human_centered_mixing_data[:len(train_human_centered_mixing_data) // 2]
    train_human_centered_mixing_data_llm = train_human_centered_mixing_data[len(train_human_centered_mixing_data) // 2:]
    test_human_centered_mixing_data_human = test_human_centered_mixing_data[:len(test_human_centered_mixing_data) // 2]
    test_human_centered_mixing_data_llm = test_human_centered_mixing_data[len(test_human_centered_mixing_data) // 2:]

    train_data = train_human_mixing_data_human + \
                 train_human_mixing_data_llm + \
                 train_human_centered_mixing_data_human + \
                 train_human_centered_mixing_data_llm + \
                 test_human_mixing_data_human[int(len(test_human_mixing_data_human) // 2):] + \
                 test_human_mixing_data_llm[int(len(test_human_mixing_data_llm) // 2):] + \
                 test_human_centered_mixing_data_human[int(len(test_human_centered_mixing_data_human) // 2):] + \
                 test_human_centered_mixing_data_llm[int(len(test_human_centered_mixing_data_llm) // 2):]

    test_data = test_human_mixing_data_human[:int(len(test_human_mixing_data_human) // 2)] + \
                test_human_mixing_data_llm[:int(len(test_human_mixing_data_llm) // 2)] + \
                test_human_centered_mixing_data_human[:int(len(test_human_centered_mixing_data_human) // 2)] + \
                test_human_centered_mixing_data_llm[:int(len(test_human_centered_mixing_data_llm) // 2)]

    print(len(train_data))
    print(len(test_data))

    save_json(train_data, "Benchmark", "Data_Mixing_Human", f"data_mixing_attacks_train.json")
    save_json(test_data, "Benchmark", "Data_Mixing_Human", f"data_mixing_attacks_test.json")
