import argparse
import json
import logging
import random
import re
import sys
import time
import nltk
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from selenium import webdriver
from Chat_API import chat
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def bt_translation(src, browser):
    zh2en_url = f'https://translate.google.com/?hl=zh&sl=en&tl=zh-CN&text={src}&op=translate'
    browser.get(zh2en_url)  # 访问相对应链接 browser.close#关闭浏览器
    time.sleep(random.randint(1, 2))
    browser.find_element_by_xpath(
        '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[1]/span/span/div/div[2]/div[1]').send_keys(
        src)
    browser.refresh()
    # time.sleep(50)
    time.sleep(random.randint(2, 3))
    text = browser.find_element_by_xpath(
        '/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div/div[8]/div/div[1]/span[1]').text
    en_text = text.replace("翻譯搜尋結果\n", "").replace("\n", "")
    en2zh_url = f'https://translate.google.com/?hl=zh&sl=zh-CN&tl=en&text={en_text}&op=translate'
    browser.get(en2zh_url)  # 访问相对应链接 browser.close#关闭浏览器
    time.sleep(random.randint(1, 2))
    browser.refresh()
    time.sleep(random.randint(2, 3))
    text = browser.find_element_by_xpath(
        '/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div/div[8]/div/div[1]/span[1]').text
    tgt = text.replace("翻譯搜尋結果\n", "").replace("\n", "")
    return tgt


def read_data(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_bm25_retriver(data, human_key):
    corpus = [article[human_key] for article in data[:100000]]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return corpus, bm25


def count_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return len(sentences)


def save_json_data(data, path):
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


def replace_line_breaks(s):
    s = re.sub('\n', ' ', s)
    return s


def truncate_to_last_sentence(s):
    # 从字符串尾部向前找句号的位置
    last_period = s.rfind('.') or s.rfind('!') or s.rfind('?')
    # 如果找到句号，就截断到这个位置（包括句号）
    if last_period != -1:
        s = s[:last_period + 1]
    return s


def check_paragraphs(texts):
    if count_sentences_in_paragraph(texts) >= 4:
        return True
    else:
        return False


def get_prompt(domain, prompt, sentences_num, prompt_type, icl_examples):
    prompt_center = {
        "arxiv": {
            "direct_prompt": f'Given the academic article title, write an academic article abstract with {sentences_num} sentences:\n' \
                             f'title: {prompt}\n' \
                             f'abstract:',
            "prompt_few_shot": f'Here are the academic articles titles and academic articles abstracts pairs:\n\n{icl_examples}\n\n' \
                               f'Given the academic article title, write an academic article abstract with {sentences_num} sentences:\n' \
                               f'title: {prompt}\n' \
                               f'abstract:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                 f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the academic article title, write an academic article abstract with {sentences_num} sentences in human style writings:\n' \
                                 f'title: {prompt}\n' \
                                 f'human:',
            "paraphrase_polish": f'Given the article abstract, polish the writing to meet the academic abstract style with {sentences_num} sentences, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'abstract: {prompt}\n' \
                                 f'polished abstract:'
        },
        "xsum": {
            "direct_prompt": f'Given the news summary, write a news article with {sentences_num} sentences:\n' \
                             f'news summary: {prompt}\n' \
                             f'news article:',
            "prompt_few_shot": f'Here are the news summaries and news articles pairs:\n\n{icl_examples}\n\n' \
                               f'Given the news summary, write a news article with {sentences_num} sentences:\n' \
                               f'news summary: {prompt}\n' \
                               f'news article:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                 f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the news summary, write a news article with {sentences_num} sentences in human style writings:\n' \
                                 f'news summary: {prompt}\n' \
                                 f'human:',
            "paraphrase_polish": f'Given the news article, polish the writing to meet the news article style with {sentences_num} sentences, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'news article: {prompt}\n' \
                                 f'polished news article:'
        },
        "writing_prompt": {
            "direct_prompt": f'Given the writing prompt, write a story with {sentences_num} sentences:\n' \
                             f'writing prompt: {prompt}\n' \
                             f'story:',
            "prompt_few_shot": f'Here are the writing prompts and stories pairs:\n\n{icl_examples}\n\n' \
                               f'Given the writing prompt, write a story with {sentences_num} sentences:\n' \
                               f'writing prompt: {prompt}\n' \
                               f'story:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                 f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the writing prompt, write a story with {sentences_num} sentences in human style writings:\n' \
                                 f'writing prompt: {prompt}\n' \
                                 f'story:',
            "paraphrase_polish": f'Given the story, polish the writing to meet the story style, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'story: {prompt}\n' \
                                 f'polished story:'
        },
        "yelp_review": {
            "direct_prompt": f'Given the review\'s first sentence, please help to continue the review with {sentences_num} sentences (do not reject me):\n' \
                             f'review\'s first sentence: {prompt}\n' \
                             f'continued review:',
            "prompt_few_shot": f'Here are the reviews\' first sentence and continued reviews pairs:\n\n{icl_examples}\n\n' \
                               f'Given the review\'s first sentence, continue the review with {sentences_num} sentences:\n' \
                               f'review\'s first sentence: {prompt}\n' \
                               f'continued review:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                 f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the review\'s first sentence, continue the review with {sentences_num} sentences in human style writings:\n' \
                                 f'review\'s first sentence: {prompt}\n' \
                                 f'continued review:',
            "paraphrase_polish": f'Given the review, polish the writing to meet the review style, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'review: {prompt}\n' \
                                 f'polished review:'
        },
    }
    return prompt_center[domain][prompt_type]


def run(args):
    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    data = read_data(args.input_path)

    # perturbation attacks prepare
    if "perturbation" in args.method:
        sys.path.append(r'/home/runzhezhan/Johnson/RobustnessBenchmark')
        from TextAttack.textattack.augmentation import TextBuggerAugmenter
        from TextAttack.textattack.augmentation import TextFoolerAugmenter
        from TextAttack.textattack.augmentation import DeepWordBugAugmenter
        word_augmenter = TextFoolerAugmenter()
        character_augmenter = DeepWordBugAugmenter()
        word_character_augmenter = TextBuggerAugmenter()

    if args.method == "paraphrase_back_translation":
        browser = webdriver.Chrome('chromedriver.exe')  # 打开浏览器
        time.sleep(10)

    # domain selection
    if "arxiv" in args.input_path:
        domain = "arxiv"
        prompt_key = "title"
        human_key = "abstract"
    elif "xsum" in args.input_path:
        domain = "xsum"
        prompt_key = "summary"
        human_key = "document"
    elif "writing_prompt" in args.input_path:
        domain = "writing_prompt"
        prompt_key = "story_prompt"
        human_key = "story"
    elif "yelp_review" in args.input_path:
        domain = "yelp_review"
        prompt_key = "start"
        human_key = "content"
    else:
        raise ValueError("input_path is not correct")

    logging.info(f"domain: {domain}")

    # in-content learning prepare
    if args.method == "prompt_few_shot":
        all_data = read_data(args.all_data_path)
        if len(all_data) > 1000000:
            all_data = all_data[:1000000]
        all_corpus, all_bm25_retriver = get_bm25_retriver(all_data, human_key)
    elif args.method == "prompt_ICO":
        data_corpus, data_bm25_retriver = get_bm25_retriver(data, human_key)

    if args.check:
        check_key = ["direct_prompt", "prompt_few_shot", "prompt_ICO", "paraphrase_polish_human",
                     "paraphrase_polish_llm", "paraphrase_back_translation_human", "paraphrase_back_translation_llm"]
        for key in check_key:
            logging.info(f"check_key: {key}")
            del_num = 0
            truncate_num = 0
            reject_num = 0
            del_cases = []
            loss_num = 0
            for article in tqdm(data):
                if key not in article:
                    logging.info(f" {key} is not in article: {article[human_key]}")
                    loss_num += 1
                else:
                    if count_sentences_in_paragraph(article[key]) < 4:
                        article[key] = ""
                        del_num += 1
                    if len(article[key]) > 2000:
                        preprocessed_article = article[key]
                        article[key] = article[key][:2000]
                        article[key] = truncate_to_last_sentence(article[key])
                        logging.info(f" {len(preprocessed_article)} is truncated as {len(article[key])}")
                        truncate_num += 1
            response = input(f"{del_num} items will be deleted, Do you want to see del cases? (yes/no): ")
            if response.lower() == 'yes':
                logging.info(f"deleted cases: {del_cases}")
                del_stats = input(f"Do you want to del? (yes/no): ")
                if del_stats.lower() == 'yes':
                    for article in tqdm(data):
                        if key in article.keys() and article[key] in del_cases:
                            logging.info(f"pre deleted article: {article[key]}")
                            article[key] = ""
                            logging.info(f"deleted article: {article[key]}")
            else:
                pass

            logging.info(
                f"check_key: {key} finished, {del_num} items are deleted, {truncate_num} items are truncated, {reject_num} items are rejected, {loss_num} items are lost")

        response = input("Do you want to save? (yes/no): ")
        if response.lower() == 'yes':
            save_json_data(data, args.input_path)
        else:
            pass

    # check
    if args.method != "direct_prompt":
        for item in data:
            assert "direct_prompt" in item.keys() and item['direct_prompt'] != ""
    if not args.check:
        for article in tqdm(data):
            id = article['id']
            prompt = article[prompt_key]
            human = article[human_key]

            sentences_num = count_sentences_in_paragraph(human)

            # llm type selection
            if id in range(1, 701):
                llm_type = "ChatGPT"
            elif id in range(701, 1401):
                llm_type = "Llama-2-70b"
            elif id in range(1401, 2101):
                llm_type = "Claude-instant"
            elif id in range(2101, 2801):
                llm_type = "Google-PaLM"
            else:
                raise ValueError("id is not in range")

            # llms selection
            if llm_type not in args.llm_types:
                pass
            else:
                if args.method == "direct_prompt":
                    direct_prompt = get_prompt(domain=domain, prompt=prompt, sentences_num=sentences_num,
                                               prompt_type="direct_prompt", icl_examples=None)
                    try:
                        if "direct_prompt" in article.keys() and article['direct_prompt'] != "":
                            pass
                        else:
                            logging.info(f"Prompt:{direct_prompt}")
                            response = chat([{"role": "user", "prompt": direct_prompt}], llm_type)
                            logging.info(f"Human Answer: {human}")
                            logging.info(f"Response:{response}")
                            article['direct_prompt'] = response
                            article['llm_type'] = llm_type
                            article['domain'] = domain
                            time.sleep(random.randint(1, 4))

                        save_json_data(data, args.output_path)

                    except Exception as e:
                        logging.error(f"Error:{e}")
                        pass

                if args.method == "prompt_few_shot":
                    # direct prompt is the base for prompt_few_shot
                    if args.method in article.keys() and article[args.method] != "":
                        pass
                    else:
                        try:
                            icl_examples = []
                            tokenized_query = human.split(" ")
                            examples = all_bm25_retriver.get_top_n(tokenized_query, all_corpus, n=4)

                            for example in examples[1:]:
                                for item in all_data:
                                    if item[human_key] == example:
                                        icl_examples.append(
                                            f"{prompt_key}: {item[prompt_key]}\n{human_key}: {item[human_key]}")
                            icl_examples = "\n\n".join(icl_examples)
                            icl_prompt = get_prompt(domain=domain, prompt=prompt, sentences_num=sentences_num,
                                                    prompt_type=args.method, icl_examples=icl_examples)
                            logging.info(f"Prompt:{icl_prompt}")
                            response = chat([{"role": "user", "prompt": icl_prompt}], llm_type)
                            logging.info(f"Response:{response}")
                            article[args.method] = response
                            article['llm_type'] = llm_type
                            article['domain'] = domain
                            time.sleep(random.randint(1, 4))
                        except Exception as e:
                            logging.error(f"Error:{e}")
                            pass

                    save_json_data(data, args.output_path)

                if args.method == "prompt_ICO":
                    if "prompt_ICO" in article.keys() and article["prompt_ICO"] != "":
                        pass
                    else:
                        try:
                            icl_examples = []
                            tokenized_query = human.split(" ")
                            examples = data_bm25_retriver.get_top_n(tokenized_query, data_corpus, n=4)
                            assert len(examples) == 4

                            for example in examples[2:]:
                                for item in data:
                                    if item[human_key] == example:
                                        icl_examples.append(
                                            f"AI writings: {replace_line_breaks(item['direct_prompt'])}\nHuman writings: {replace_line_breaks(item[human_key])}")
                            icl_examples = "\n\n".join(icl_examples)

                            ICO_step1_prompt = get_prompt(domain=domain, prompt=prompt, sentences_num=sentences_num,
                                                           prompt_type=args.method + "_step1",
                                                           icl_examples=icl_examples)
                            logging.info(f"ICO_step1_Prompt:{ICO_step1_prompt}")
                            ICO_step1_response = chat([{"role": "user", "prompt": ICO_step1_prompt}], llm_type)
                            logging.info(f"ICO_step1_Response:{ICO_step1_response}")
                            time.sleep(random.randint(1, 4))

                            ICO_step2_prompt = get_prompt(domain=domain, prompt=prompt, sentences_num=sentences_num,
                                                           prompt_type=args.method + "_step2",
                                                           icl_examples=None)

                            logging.info(f"ICO_step2_Prompt:{ICO_step2_prompt}")
                            ICO_step2_response = chat([{"role": "bot", "prompt": ICO_step1_response},
                                                        {"role": "user", "prompt": ICO_step2_prompt}], llm_type)
                            logging.info(f"ICO_step2_Response:{ICO_step2_response}")
                            time.sleep(random.randint(1, 4))

                            article[args.method] = ICO_step2_response

                            save_json_data(data, args.output_path)

                        except Exception as e:
                            logging.info(f"error: {e}")
                            pass

                if args.method == "paraphrase_polish":
                    llm = article['direct_prompt']
                    try:
                        if args.method + "_human" in article.keys() and article[args.method + "_human"] != "":
                            pass
                        else:
                            polish_prompt = get_prompt(domain=domain, prompt=human, sentences_num=sentences_num,
                                                       prompt_type=args.method, icl_examples=None)

                            logging.info(f"Prompt:{polish_prompt}")
                            response = chat([{"role": "user", "prompt": polish_prompt}], llm_type)
                            logging.info(f"Response:{response}")
                            article[args.method + "_human"] = response
                            time.sleep(random.randint(1, 4))

                            save_json_data(data, args.output_path)

                        if args.method + "_llm" in article.keys() and article[args.method + '_llm'] != "":
                            pass
                        else:
                            polish_prompt = get_prompt(domain=domain, prompt=llm, sentences_num=sentences_num,
                                                       prompt_type=args.method, icl_examples=None)
                            logging.info(f"Prompt:{polish_prompt}")
                            response = chat([{"role": "user", "prompt": polish_prompt}], llm_type)
                            logging.info(f"Response:{response}")
                            article[args.method + "_llm"] = response
                            time.sleep(random.randint(1, 4))

                            save_json_data(data, args.output_path)

                    except Exception as e:
                        logging.info(f"error: {e}")
                        pass

                if args.method == "paraphrase_back_translation":
                    human = human
                    llm = article['direct_prompt']
                    try:
                        if args.method + "_human" in article.keys() and article[args.method + "_human"] != "":
                            pass
                        else:
                            article[args.method + "_human"] = bt_translation(human, browser)
                            logging.info(f'human bt success: {article[args.method + "_human"]}')
                            time.sleep(random.randint(1, 4))

                            save_json_data(data, args.output_path)

                        if args.method + "_llm" in article.keys() and article['_llm'] != "":
                            pass
                        else:
                            article[args.method + "_llm"] = bt_translation(llm, browser)
                            logging.info(f'llm bt success: {article[args.method + "_llm"]}')
                            time.sleep(random.randint(1, 4))

                            save_json_data(data, args.output_path)

                    except Exception as e:
                        print(e)
                        pass

                if "perturbation" in args.method:
                    humans = count_sentences_in_paragraph(human)
                    llms = count_sentences_in_paragraph(article['direct_prompt'])
                    for attack in ["perturbation_character", "perturbation_word", "perturbation_sent"]:
                        if attack == "perturbation_character":
                            augmenter = character_augmenter
                        elif attack == "perturbation_word":
                            augmenter = word_augmenter
                        elif attack == "perturbation_sent":
                            augmenter = word_character_augmenter
                        else:
                            raise ValueError(f"{attack} is not in perturbation_attacks")

                        try:
                            if attack + "human" in article.keys() and article[attack + "human"] != "":
                                pass
                            else:
                                final_data = []
                                for d in humans:
                                    final_data.append(augmenter.augment(d)[0])
                                human_result = ' '.join(final_data)
                                article[attack + "human"] = human_result
                                logging.info(f"{attack} human finished")

                                save_json_data(data, args.output_path)

                            if attack + "llm" in article.keys() and article[attack + "llm"] != "":
                                pass
                            else:
                                final_data = []
                                for d in llms:
                                    final_data.append(augmenter.augment(d)[0])
                                llm_result = ' '.join(final_data)
                                article[attack + "llm"] = llm_result
                                logging.info(f"{attack} llm finished")

                                save_json_data(data, args.output_path)

                        except Exception as e:
                            logging.info(f"error: {e}")
                            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_types', type=list, default=["ChatGPT", "Llama-2-70b", "Claude-instant", "Google-PaLM"])
    parser.add_argument('--input_path', required=False, default="arxiv_2800.json", type=str)
    parser.add_argument('--output_path', required=False, default="arxiv_2800.json", type=str)
    parser.add_argument('--all_data_path', default="arxiv_all.json", type=str, required=False)
    parser.add_argument('--method', default="paraphrase_back_translation", type=str,
                        choices=["direct_prompt", "prompt_few_shot", "prompt_ICO",
                                 "perturbation_character", "perturbation_word", "perturbation_sent",
                                 "paraphrase_back_translation", "paraphrase_polish", "paraphrase_dipper"],
                        required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    parser.add_argument('--check', default=False, type=bool, required=False)
    args = parser.parse_args()
    run(args)
