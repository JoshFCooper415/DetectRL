import json
import re
import nltk
from tqdm import tqdm


def clean_text(text):
    abbreviations = {'U. S. A.': 'U.S.A.', 'e. g.': 'e.g.', 'i. e.': 'i.e.', 'U. S.': 'U.S.'}
    text = re.sub(r"([?.!,:;'])(?!\s|$)", r'\1 ', text)
    text = re.sub(r"\s([?.!,:;'](?:\s|$))", r'\1', text)
    text = re.sub(r'\s\'s\b', '\'s', text)
    for abbr, replacement in abbreviations.items():
        text = text.replace(abbr, replacement)
    text = re.sub(r'\n', r'', text)
    text = re.sub(r'\\', r'', text)
    text = re.sub(r'\\', r'', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"``", r'"', text)
    text = re.sub(r"''", r'"', text)
    return text


devide = ["train", "test", "valid"]
data = []
for name in devide:
    with open(name + ".wp_target", encoding="utf-8") as f:
        stories = f.readlines()

    with open(name + ".wp_source", encoding="utf-8") as f:
        stories_prompt = f.readlines()

    for i in tqdm(range(len(stories))):
        data.append({
            'story': clean_text(re.sub(r"<newline>", r"", stories[i])),
            'story_prompt': clean_text(stories_prompt[i][7:])
        })


def count_sentences_in_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return len(sentences)


new_data = []
for item in data:
    if len(item['story']) > 800 and len(item['story']) < 2000:
        if count_sentences_in_paragraph(item['story']) >= 4:
            item['story'] = clean_text(item['story'])
            item['story_prompt'] = clean_text(item['story_prompt'])
            new_data.append(item)

new_data = new_data[:100000]
print(len(new_data))
with open('writing_prompt_all.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

with open('writing_prompt_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
length = []
sentence_num = []
new_data = []
id = 1
for article in tqdm(data):
    length.append(len(article['story']))
    sentence_num.append(count_sentences_in_paragraph(article['story']))
    if len(article['story']) > 800 and len(article['story']) < 2000:
        if count_sentences_in_paragraph(article['story']) >= 4:
            new_data.append({
                'id': id,
                'story': article['story'],
                'story_prompt': article['story_prompt']
            })
            id += 1

print(len(new_data))
new_data = new_data[:2800]
with open('writing_prompt_2800.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)