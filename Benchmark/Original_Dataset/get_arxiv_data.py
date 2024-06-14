import json
import re
import nltk
from tqdm import tqdm


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def clean_text(text):
    text = re.sub("\n", r'', text)
    return text


def count_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return len(sentences)


def clean_text(text):
    abbreviations = {'U. S. A.': 'U.S.A.', 'e. g.': 'e.g.', 'i. e.': 'i.e.', 'U. S.': 'U.S.'}
    text = re.sub(r'([?.!,:])(?!\s|$)', r'\1 ', text)
    text = re.sub(r"\s([?.!,:](?:\s|$))", r'\1', text)
    text = re.sub(r'\s\'s\b', '\'s', text)
    text = re.sub(r'\n', '', text)
    for abbr, replacement in abbreviations.items():
        text = text.replace(abbr, replacement)
    return text


# arxiv_papers = read_jsonl("arxiv_papers.json")
#
# data = []
# length = []
# sentence_num = []
# arxiv_papers = arxiv_papers
# num = 0
# for article in tqdm(arxiv_papers):
#     if len(data) < 100000:
#         if len(article['abstract']) > 800 and len(article['abstract']) < 2000:
#             if count_sentences_in_paragraph(article['abstract']) >= 4:
#                 data.append({
#                     'id': clean_text(article['id']),
#                     'title': clean_text(article['title']),
#                     'abstract': clean_text(article['abstract']),
#                 })
#                 num += 1
#                 print(num)
#                 length.append(len(article['abstract']))
#                 sentence_num.append(count_sentences_in_paragraph(article['abstract']))
#     else:
#         break
# with open('arxiv_all.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

with open('arxiv_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in tqdm(data):
        item['title'] = clean_text(item['title'])
        item['abstract'] = clean_text(item['abstract'])

with open('arxiv_all.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
length = []
sentence_num = []
id = 1
new_data = []
for article in tqdm(data):
    length.append(len(article['abstract']))
    sentence_num.append(count_sentences_in_paragraph(article['abstract']))
    if len(article['abstract']) > 800 and len(article['abstract']) < 2000:
        if count_sentences_in_paragraph(article['abstract']) >= 4:
            new_data.append({
                'id': id,
                'title': article['title'],
                'abstract': article['abstract'],
            })
            id += 1

new_data = new_data[:2800]
with open('arxiv_2800.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
