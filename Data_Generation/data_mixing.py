import json
import math
import random
import nltk


def get_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences


def read_data(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def sample_multiple_of_four(min_length, max_length):
    random.seed(2023)
    min_mult = -(-min_length // 4) * 4
    max_mult = max_length // 4 * 4
    multiples_of_four = list(range(min_mult, max_mult + 1, 4))
    return random.choice(multiples_of_four)


def get_llm_mixed(domains, llm_types, domain_human):
    train_human = []
    test_human = []
    train_llm = []
    test_llm = []
    for domain in domains:
        human_key = domain_human[domain]
        file_name = f"{domain}_2800.json"
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt"]
        human = []
        llm = []
        data = [item for item in data if item["llm_type"] in llm_types]
        for item in data:
            for key in human_keys:
                if item[key] != "":
                    human.append({
                        "text": item[key],
                        "label": "human",
                        "data_type": key,
                        "llm_type": item["llm_type"]
                    })
            for key in llm_keys:
                if item[key] != "":
                    llm.append({
                        "text": item[key],
                        "label": "llm",
                        "data_type": key,
                        "llm_type": item["llm_type"]
                    })

        import random
        random.seed(2023)

        grouped_llm_data = {f"{llm_type}_{llm_key}": [] for llm_type in llm_types for llm_key in llm_keys}
        for item in llm:
            if f"{item['llm_type']}_{item['data_type']}" in grouped_llm_data.keys():
                grouped_llm_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

        llm_data = []
        for llm_key in llm_keys:
            length_range = []
            ChatGPT_data = grouped_llm_data[f"ChatGPT_{llm_key}"]
            PaLM_data = grouped_llm_data[f"Google-PaLM_{llm_key}"]
            Claude_data = grouped_llm_data[f"Claude-instant_{llm_key}"]
            Llama_data = grouped_llm_data[f"Llama-2-70b_{llm_key}"]
            for i in range(len(ChatGPT_data)):
                length_range.append(len(get_sentences_in_paragraph(ChatGPT_data[i]["text"])))
                length_range.append(len(get_sentences_in_paragraph(PaLM_data[i]["text"])))
                length_range.append(len(get_sentences_in_paragraph(Claude_data[i]["text"])))
                length_range.append(len(get_sentences_in_paragraph(Llama_data[i]["text"])))

            max_length = max(length_range)
            min_length = min(length_range)

            max_length = max_length if max_length / 4 <= min_length else min_length * 4
            sample = sample_multiple_of_four(min_length, max_length)
            for i in range(len(ChatGPT_data)):
                ChatGPT_sentences = get_sentences_in_paragraph(ChatGPT_data[i]["text"])
                PaLM_sentences = get_sentences_in_paragraph(PaLM_data[i]["text"])
                Claude_sentences = get_sentences_in_paragraph(Claude_data[i]["text"])
                Llama_sentences = get_sentences_in_paragraph(Llama_data[i]["text"])

                ChatGPT_sentences = random.sample(ChatGPT_sentences, int(sample / 4))
                PaLM_sentences = random.sample(PaLM_sentences, int(sample / 4))
                Claude_sentences = random.sample(Claude_sentences, int(sample / 4))
                Llama_sentences = random.sample(Llama_sentences, int(sample / 4))

                Final_sentences = ChatGPT_sentences + PaLM_sentences + Claude_sentences + Llama_sentences
                random.shuffle(Final_sentences)
                Final_sentences = " ".join(Final_sentences)

                llm_data.append({
                    "text": Final_sentences,
                    "label": "llm",
                    "data_type": llm_key,
                    "llm_type": "4_mixed"
                })

        llm_test_num = 250

        grouped_human_data = {f"{llm_type}_{human_key}": [] for llm_type in llm_types for human_key in human_keys}
        for item in human:
            if f"{item['llm_type']}_{item['data_type']}" in grouped_human_data.keys():
                grouped_human_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

        for data_type, items in grouped_human_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(human_keys) * len(llm_types))))
            sampled_human = [items[i] for i in sampled_indices]
            sampled_indices.sort(reverse=True)
            for index in sampled_indices:
                items.pop(index)
            test_human += sampled_human
            train_human += items

        grouped_llm_data = {f"{llm_key}": [] for llm_key in llm_keys}
        for item in llm:
            if f"{item['data_type']}" in grouped_llm_data.keys():
                grouped_llm_data[f"{item['data_type']}"].append(item)

        for key, items in grouped_llm_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(llm_keys))))
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


def get_human_mixed(domains, llm_types, domain_human):
    train_human = []
    test_human = []
    train_llm = []
    test_llm = []
    for domain in domains:
        file_name = f"{domain}_2800.json"
        human_key = domain_human[domain]
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt"]
        human = []
        llm = []
        data = [item for item in data if item["llm_type"] in llm_types]
        for item in data:
            for key in human_keys:
                if item[key] != "":
                    human.append({
                        "text": item[key],
                        "label": "human",
                        "data_type": key,
                        "llm_type": item["llm_type"]
                    })
            for key in llm_keys:
                if item[key] != "":
                    llm.append({
                        "text": item[key],
                        "label": "llm",
                        "data_type": key,
                        "llm_type": item["llm_type"]
                    })

        import random
        random.seed(2023)

        grouped_llm_data = {f"{llm_type}_{llm_key}": [] for llm_type in llm_types for llm_key in llm_keys}
        for item in llm:
            if f"{item['llm_type']}_{item['data_type']}" in grouped_llm_data.keys():
                grouped_llm_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

        llm_data = []
        for llm_key in llm_keys:
            length_range = []
            ChatGPT_data = grouped_llm_data[f"ChatGPT_{llm_key}"]
            PaLM_data = grouped_llm_data[f"Google-PaLM_{llm_key}"]
            Claude_data = grouped_llm_data[f"Claude-instant_{llm_key}"]
            Llama_data = grouped_llm_data[f"Llama-2-70b_{llm_key}"]
            for i in range(len(ChatGPT_data)):
                length_range.append(len(get_sentences_in_paragraph(ChatGPT_data[i]["text"])))
                length_range.append(len(get_sentences_in_paragraph(PaLM_data[i]["text"])))
                length_range.append(len(get_sentences_in_paragraph(Claude_data[i]["text"])))
                length_range.append(len(get_sentences_in_paragraph(Llama_data[i]["text"])))

            max_length = max(length_range)
            min_length = min(length_range)

            max_length = max_length if max_length / 4 <= min_length else min_length * 4
            sample = sample_multiple_of_four(min_length, max_length)
            for i in range(len(ChatGPT_data)):
                ChatGPT_sentences = get_sentences_in_paragraph(ChatGPT_data[i]["text"])
                PaLM_sentences = get_sentences_in_paragraph(PaLM_data[i]["text"])
                Claude_sentences = get_sentences_in_paragraph(Claude_data[i]["text"])
                Llama_sentences = get_sentences_in_paragraph(Llama_data[i]["text"])

                ChatGPT_sentences = random.sample(ChatGPT_sentences, int(sample / 4))
                PaLM_sentences = random.sample(PaLM_sentences, int(sample / 4))
                Claude_sentences = random.sample(Claude_sentences, int(sample / 4))
                Llama_sentences = random.sample(Llama_sentences, int(sample / 4))

                Final_sentences = ChatGPT_sentences + PaLM_sentences + Claude_sentences + Llama_sentences
                random.shuffle(Final_sentences)
                Final_sentences = " ".join(Final_sentences)

                llm_data.append({
                    "text": Final_sentences,
                    "label": "llm",
                    "data_type": llm_key,
                    "llm_type": "4_mixed"
                })

        llm_test_num = 250

        grouped_human_data = {f"{llm_type}_{human_key}": [] for llm_type in llm_types for human_key in human_keys}
        for item in human:
            if f"{item['llm_type']}_{item['data_type']}" in grouped_human_data.keys():
                grouped_human_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

        for data_type, items in grouped_human_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(human_keys) * len(llm_types))))
            sampled_human = [items[i] for i in sampled_indices]
            sampled_indices.sort(reverse=True)
            for index in sampled_indices:
                items.pop(index)
            test_human += sampled_human
            train_human += items

        grouped_llm_data = {f"{llm_key}": [] for llm_key in llm_keys}
        for item in llm:
            if f"{item['data_type']}" in grouped_llm_data.keys():
                grouped_llm_data[f"{item['data_type']}"].append(item)

        for key, items in grouped_llm_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(llm_keys))))
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


def get_human_centered_mixed(domains, llm_types, domain_human):
    train_human = []
    test_human = []
    train_llm = []
    test_llm = []
    for domain in domains:
        human_key = domain_human[domain]
        file_name = f"{domain}_2800.json"
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt"]

        data = [item for item in data if item["llm_type"] in llm_types]
        human = []
        llm = []
        for item in data:
            for llm_key in llm_keys:
                if item[human_key] != "":
                    human_text = item[human_key]
                    llm_text = item[llm_key]

                    human_texts = get_sentences_in_paragraph(human_text)
                    human_length = len(human_texts)
                    llm_texts = get_sentences_in_paragraph(llm_text)
                    llm_length = len(llm_texts)

                    sample_length = int(human_length * 0.25)
                    if sample_length > llm_length:
                        sample_length = llm_length

                    random.seed(2023)
                    sampled_indices = random.sample(range(human_length), sample_length)
                    for index in sampled_indices:
                        sampled_llm_indices = random.sample(range(len(llm_texts)), 1)[0]
                        human_texts[index] = llm_texts[sampled_llm_indices]
                        llm_texts.pop(sampled_llm_indices)

                    human_text = " ".join(human_texts)
                    human.append({
                        "text": human_text,
                        "label": "human",
                        "data_type": human_key,
                        "llm_type": item["llm_type"]
                    })
                    llm.append({
                        "text": item[llm_key],
                        "label": "llm",
                        "data_type": llm_key,
                        "llm_type": item["llm_type"]
                    })

        llm_test_num = 250

        # human
        grouped_human_data = {f"{llm_type}_{human_key}": [] for llm_type in llm_types for human_key in human_keys}
        for item in human:
            if f"{item['llm_type']}_{item['data_type']}" in grouped_human_data.keys():
                grouped_human_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

        for data_type, items in grouped_human_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(human_keys) * len(llm_types))))
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

        for key, items in grouped_llm_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(llm_keys) * len(llm_types))))
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


def get_llm_centered_mixed(domains, llm_types, domain_human):
    train_human = []
    test_human = []
    train_llm = []
    test_llm = []
    for domain in domains:
        human_key = domain_human[domain]
        file_name = f"{domain}_2800.json"
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt"]

        data = [item for item in data if item["llm_type"] in llm_types]
        human = []
        llm = []
        for item in data:
            for llm_key in llm_keys:
                if item[human_key] != "":
                    human_text = item[human_key]
                    llm_text = item[llm_key]

                    human_texts = get_sentences_in_paragraph(human_text)
                    human_length = len(human_texts)
                    llm_texts = get_sentences_in_paragraph(llm_text)
                    llm_length = len(llm_texts)

                    sample_length = int(llm_length * 0.25)
                    if sample_length > human_length:
                        sample_length = human_length

                    random.seed(2023)
                    sampled_indices = random.sample(range(llm_length), sample_length)
                    for index in sampled_indices:
                        sampled_human_indices = random.sample(range(len(human_texts)), 1)[0]
                        llm_texts[index] = human_texts[sampled_human_indices]
                        human_texts.pop(sampled_human_indices)

                    llm_text = " ".join(llm_texts)
                    human.append({
                        "text": item[human_key],
                        "label": "human",
                        "data_type": human_key,
                        "llm_type": item["llm_type"]
                    })
                    llm.append({
                        "text": llm_text,
                        "label": "llm",
                        "data_type": llm_key,
                        "llm_type": item["llm_type"]
                    })

        llm_test_num = 250

        # human
        grouped_human_data = {f"{llm_type}_{human_key}": [] for llm_type in llm_types for human_key in human_keys}
        for item in human:
            if f"{item['llm_type']}_{item['data_type']}" in grouped_human_data.keys():
                grouped_human_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

        for data_type, items in grouped_human_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(human_keys) * len(llm_types))))
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

        for key, items in grouped_llm_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(llm_keys) * len(llm_types))))
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


def get_sentence_mixed(domains, llm_types, domain_human):
    train_human = []
    test_human = []
    train_llm = []
    test_llm = []
    for domain in domains:
        human_key = domain_human[domain]
        file_name = f"{domain}_2800.json"
        data = read_data(file_name)
        human_keys = [human_key]
        llm_keys = ["direct_prompt", "prompt_few_shot", "prompt_SICO", "paraphrase_polish_llm",
                    'adversarial_character_llm', 'adversarial_word_llm', 'adversarial_character_word_llm']

        data = [item for item in data if item["llm_type"] in llm_types]
        human = []
        llm = []
        for item in data:
            for llm_key in llm_keys:
                if item[human_key] != "":
                    human_text = item[human_key]
                    llm_text = item[llm_key]

                    human_texts = get_sentences_in_paragraph(human_text)
                    human_length = len(human_texts)
                    llm_texts = get_sentences_in_paragraph(llm_text)
                    llm_length = len(llm_texts)

                    cover_human_length = 0
                    for i in range(human_length):
                        if len(human_texts[i].split(",")) >= 4:
                            cover_human_length += 1

                        print(human_texts[i].split(","))

                    print("=" * 10)

                    cover_llm_length = 0
                    for i in range(llm_length):
                        if len(llm_texts[i].split(",")) >= 4:
                            cover_llm_length += 1

                        print(llm_texts[i].split(","))

                    print(cover_human_length, human_length, cover_llm_length, llm_length)

                    sample_length = int(llm_length * 0.25)
                    if sample_length > human_length:
                        sample_length = human_length

                    random.seed(2023)
                    sampled_indices = random.sample(range(llm_length), sample_length)
                    for index in sampled_indices:
                        sampled_human_indices = random.sample(range(len(human_texts)), 1)[0]
                        llm_texts[index] = human_texts[sampled_human_indices]
                        human_texts.pop(sampled_human_indices)

                    llm_text = " ".join(llm_texts)
                    human.append({
                        "text": item[human_key],
                        "label": "human",
                        "data_type": human_key,
                        "llm_type": item["llm_type"]
                    })
                    llm.append({
                        "text": llm_text,
                        "label": "llm",
                        "data_type": llm_key,
                        "llm_type": item["llm_type"]
                    })

        llm_test_num = 250

        # human
        grouped_human_data = {f"{llm_type}_{human_key}": [] for llm_type in llm_types for human_key in human_keys}
        for item in human:
            if f"{item['llm_type']}_{item['data_type']}" in grouped_human_data.keys():
                grouped_human_data[f"{item['llm_type']}_{item['data_type']}"].append(item)

        for data_type, items in grouped_human_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(human_keys) * len(llm_types))))
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

        for key, items in grouped_llm_data.items():
            sampled_indices = random.sample(range(len(items)),
                                            math.ceil(llm_test_num / (len(llm_keys) * len(llm_types))))
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
