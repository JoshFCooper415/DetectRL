import re
import json
import nltk
from tqdm import tqdm
from datasets import load_dataset


def clean_text(text):
    text = re.sub(r'\n', r'', text)
    return text


dataset = load_dataset('xsum')

train_dataset = dataset['train']
eval_dataset = dataset['validation']
test_dataset = dataset['test']


def count_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return len(sentences)


data = []
for article in tqdm(train_dataset):
    if len(article['document']) > 800 and len(article['document']) < 2000:
        if count_sentences_in_paragraph(article['document']) >= 4:
            data.append({
                'id': article['id'],
                'summary': clean_text(article['summary']),
                'document': clean_text(article['document'])
            })
for article in tqdm(eval_dataset):
    if len(article['document']) > 800 and len(article['document']) < 2000:
        if count_sentences_in_paragraph(article['document']) >= 4:
            data.append({
                'id': article['id'],
                'summary': clean_text(article['summary']),
                'document': clean_text(article['document'])
            })
for article in tqdm(test_dataset):
    if len(article['document']) > 800 and len(article['document']) < 2000:
        if count_sentences_in_paragraph(article['document']) >= 4:
            data.append({
                'id': article['id'],
                'summary': clean_text(article['summary']),
                'document': clean_text(article['document'])
            })
data = data[:100000]
with open('xsum_all.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

with open('xsum_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
length = []
sentence_num = []
id = 1
new_data = []
for article in tqdm(data):
    length.append(len(article['document']))
    sentence_num.append(count_sentences_in_paragraph(article['document']))
    if len(article['document']) > 800 and len(article['document']) < 2000:
        if count_sentences_in_paragraph(article['document']) >= 4:
            new_data.append({
                'id': id,
                'summary': article['summary'],
                'document': article['document'],
            })
            id += 1

new_data = new_data[:2800]
with open('xsum_2800.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
