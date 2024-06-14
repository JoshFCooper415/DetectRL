import re
import json
import nltk
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset('yelp_review_full')


def count_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return len(sentences)


def clean_text(text):
    abbreviations = {'U. S. A.': 'U.S.A.', 'e. g.': 'e.g.', 'i. e.': 'i.e.', 'U. S.': 'U.S.'}
    text = re.sub(r'([?.!,:])(?!\s|$)', r'\1 ', text)
    text = re.sub(r"\s([?.!,:](?:\s|$))", r'\1', text)
    text = re.sub(r'\s\'s\b', '\'s', text)
    for abbr, replacement in abbreviations.items():
        text = text.replace(abbr, replacement)
    text = re.sub(r'\n', r'', text)
    text = re.sub(r'\\', r'', text)
    text = re.sub(r'\\', r'', text)
    return text


train_dataset = dataset['train']
test_dataset = dataset['test']
data = []
for review in tqdm(train_dataset):
    text = clean_text(review['text'])
    sentences = nltk.sent_tokenize(text)
    if len(" ".join(sentences[1:])) > 800 and len(" ".join(sentences[1:])) < 2000:
        if count_sentences_in_paragraph(" ".join(sentences[1:])) >= 4:
            try:
                data.append({
                    'start': sentences[0],
                    'content': " ".join(sentences[1:]),
                })
            except IndexError:
                pass

for review in tqdm(test_dataset):
    text = clean_text(review['text'])
    sentences = nltk.sent_tokenize(text)
    if len(" ".join(sentences[1:])) > 800 and len(" ".join(sentences[1:])) < 2000:
        if count_sentences_in_paragraph(" ".join(sentences[1:])) >= 4:
            try:
                data.append({
                    'start': sentences[0],
                    'content': " ".join(sentences[1:]),
                })
            except IndexError:
                pass

data = data[:100000]
with open('yelp_review_all.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

with open('yelp_review_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
length = []
sentence_num = []
id = 1
new_data = []
for article in tqdm(data):
    length.append(len(article['content']))
    sentence_num.append(count_sentences_in_paragraph(article['content']))
    if len(article['content']) > 800 and len(article['content']) < 2000:
        if count_sentences_in_paragraph(article['content']) >= 4:
            new_data.append({
                'id': id,
                'start': article['start'],
                'content': article['content'],
            })
            id += 1

new_data = new_data[:2800]
with open('yelp_review_2800.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
