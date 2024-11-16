from collections import Counter
from matplotlib import pyplot as plt, ticker
import json


def count_ngrams(line, n=1):
    ngram_counter = Counter()
    words = line.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    ngram_counter.update([" ".join(ngram) for ngram in ngrams])
    return ngram_counter


# Initialize counters
unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()

"""
Cross Domains

"""
with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_arxiv_test.json", "r",
          encoding="utf-8") as f:
    arxiv = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_xsum_test.json", "r",
          encoding="utf-8") as f:
    xsum = json.load(f)

with open(
        r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_writing_prompt_test.json",
        "r", encoding="utf-8") as f:
    writing = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_yelp_review_test.json",
          "r", encoding="utf-8") as f:
    review = json.load(f)

arxiv_unigram_counter = Counter()
arxiv_bigram_counter = Counter()
arxiv_trigram_counter = Counter()
for item in arxiv:
    if item["label"] == "llm":
        arxiv_unigram_counter.update(count_ngrams(item["text"], n=1))
        arxiv_bigram_counter.update(count_ngrams(item["text"], n=2))
        arxiv_trigram_counter.update(count_ngrams(item["text"], n=3))

xsum_unigram_counter = Counter()
xsum_bigram_counter = Counter()
xsum_trigram_counter = Counter()
for item in xsum:
    if item["label"] == "llm":
        xsum_unigram_counter.update(count_ngrams(item["text"], n=1))
        xsum_bigram_counter.update(count_ngrams(item["text"], n=2))
        xsum_trigram_counter.update(count_ngrams(item["text"], n=3))

writing_unigram_counter = Counter()
writing_bigram_counter = Counter()
writing_trigram_counter = Counter()
for item in writing:
    if item["label"] == "llm":
        writing_unigram_counter.update(count_ngrams(item["text"], n=1))
        writing_bigram_counter.update(count_ngrams(item["text"], n=2))
        writing_trigram_counter.update(count_ngrams(item["text"], n=3))

review_unigram_counter = Counter()
review_bigram_counter = Counter()
review_trigram_counter = Counter()
for item in review:
    if item["label"] == "llm":
        review_unigram_counter.update(count_ngrams(item["text"], n=1))
        review_bigram_counter.update(count_ngrams(item["text"], n=2))
        review_trigram_counter.update(count_ngrams(item["text"], n=3))

arxiv_list = [len(arxiv_unigram_counter.keys()), len(arxiv_bigram_counter.keys()), len(arxiv_trigram_counter.keys())]
xsum_list = [len(xsum_unigram_counter.keys()), len(xsum_bigram_counter.keys()), len(xsum_trigram_counter.keys())]
writing_list = [len(writing_unigram_counter.keys()), len(writing_bigram_counter.keys()),
                len(writing_trigram_counter.keys())]
review_list = [len(review_unigram_counter.keys()), len(review_bigram_counter.keys()),
               len(review_trigram_counter.keys())]

"""
Cross LLMs

"""
with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_ChatGPT_test.json", "r",
          encoding="utf-8") as f:
    ChatGPT = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_Claude-instant_test.json", "r",
          encoding="utf-8") as f:
    Claude = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_Google-PaLM_test.json", "r",
          encoding="utf-8") as f:
    PaLM = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_Llama-2-70b_test.json", "r",
          encoding="utf-8") as f:
    Llama = json.load(f)

ChatGPT_unigram_counter = Counter()
ChatGPT_bigram_counter = Counter()
ChatGPT_trigram_counter = Counter()
for item in ChatGPT:
    if item["label"] == "llm":
        ChatGPT_unigram_counter.update(count_ngrams(item["text"], n=1))
        ChatGPT_bigram_counter.update(count_ngrams(item["text"], n=2))
        ChatGPT_trigram_counter.update(count_ngrams(item["text"], n=3))

Claude_unigram_counter = Counter()
Claude_bigram_counter = Counter()
Claude_trigram_counter = Counter()
for item in Claude:
    if item["label"] == "llm":
        Claude_unigram_counter.update(count_ngrams(item["text"], n=1))
        Claude_bigram_counter.update(count_ngrams(item["text"], n=2))
        Claude_trigram_counter.update(count_ngrams(item["text"], n=3))

PaLM_unigram_counter = Counter()
PaLM_bigram_counter = Counter()
PaLM_trigram_counter = Counter()
for item in PaLM:
    if item["label"] == "llm":
        PaLM_unigram_counter.update(count_ngrams(item["text"], n=1))
        PaLM_bigram_counter.update(count_ngrams(item["text"], n=2))
        PaLM_trigram_counter.update(count_ngrams(item["text"], n=3))

Llama_unigram_counter = Counter()
Llama_bigram_counter = Counter()
Llama_trigram_counter = Counter()
for item in Llama:
    if item["label"] == "llm":
        Llama_unigram_counter.update(count_ngrams(item["text"], n=1))
        Llama_bigram_counter.update(count_ngrams(item["text"], n=2))
        Llama_trigram_counter.update(count_ngrams(item["text"], n=3))

ChatGPT_list = [len(ChatGPT_unigram_counter.keys()), len(ChatGPT_bigram_counter.keys()),
                len(ChatGPT_trigram_counter.keys())]
Claude_list = [len(Claude_unigram_counter.keys()), len(Claude_bigram_counter.keys()),
               len(Claude_trigram_counter.keys())]
PaLM_list = [len(PaLM_unigram_counter.keys()), len(PaLM_bigram_counter.keys()), len(PaLM_trigram_counter.keys())]
Llama_list = [len(Llama_unigram_counter.keys()), len(Llama_bigram_counter.keys()), len(Llama_trigram_counter.keys())]

"""
Cross Attacks

"""
with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\DirectPrompt\direct_prompt_test.json", "r",
          encoding="utf-8") as f:
    direct_prompt = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\PromptAttacks\prompt_attacks_test.json", "r",
          encoding="utf-8") as f:
    prompt_attacks = json.load(f)

with open(
        r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\ParaphraseAttacks\paraphrase_attacks_attack_test.json",
        "r", encoding="utf-8") as f:
    paraphrase_attacks = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\AdversarialAttacks\adversarial_attacks_test.json",
          "r", encoding="utf-8") as f:
    perturbation_attacks = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\Mixed\mixed_attack_test.json", "r",
          encoding="utf-8") as f:
    data_mixing = json.load(f)

direct_prompt_unigram_counter = Counter()
direct_prompt_bigram_counter = Counter()
direct_prompt_trigram_counter = Counter()

human_unigram_counter = Counter()
human_bigram_counter = Counter()
human_trigram_counter = Counter()
for item in direct_prompt:
    if item["label"] == "llm":
        direct_prompt_unigram_counter.update(count_ngrams(item["text"], n=1))
        direct_prompt_bigram_counter.update(count_ngrams(item["text"], n=2))
        direct_prompt_trigram_counter.update(count_ngrams(item["text"], n=3))
    elif item["label"] == "human":
        human_unigram_counter.update(count_ngrams(item["text"], n=1))
        human_bigram_counter.update(count_ngrams(item["text"], n=2))
        human_trigram_counter.update(count_ngrams(item["text"], n=3))

prompt_attacks_unigram_counter = Counter()
prompt_attacks_bigram_counter = Counter()
prompt_attacks_trigram_counter = Counter()
for item in prompt_attacks:
    if item["label"] == "llm":
        prompt_attacks_unigram_counter.update(count_ngrams(item["text"], n=1))
        prompt_attacks_bigram_counter.update(count_ngrams(item["text"], n=2))
        prompt_attacks_trigram_counter.update(count_ngrams(item["text"], n=3))

paraphrase_attacks_unigram_counter = Counter()
paraphrase_attacks_bigram_counter = Counter()
paraphrase_attacks_trigram_counter = Counter()
for item in paraphrase_attacks:
    if item["label"] == "llm":
        paraphrase_attacks_unigram_counter.update(count_ngrams(item["text"], n=1))
        paraphrase_attacks_bigram_counter.update(count_ngrams(item["text"], n=2))
        paraphrase_attacks_trigram_counter.update(count_ngrams(item["text"], n=3))

perturbation_attacks_unigram_counter = Counter()
perturbation_attacks_bigram_counter = Counter()
perturbation_attacks_trigram_counter = Counter()
for item in perturbation_attacks:
    if item["label"] == "llm":
        perturbation_attacks_unigram_counter.update(count_ngrams(item["text"], n=1))
        perturbation_attacks_bigram_counter.update(count_ngrams(item["text"], n=2))
        perturbation_attacks_trigram_counter.update(count_ngrams(item["text"], n=3))

data_mixing_unigram_counter = Counter()
data_mixing_bigram_counter = Counter()
data_mixing_trigram_counter = Counter()
for item in data_mixing:
    if item["label"] == "llm":
        data_mixing_unigram_counter.update(count_ngrams(item["text"], n=1))
        data_mixing_bigram_counter.update(count_ngrams(item["text"], n=2))
        data_mixing_trigram_counter.update(count_ngrams(item["text"], n=3))

direct_prompt_list = [len(direct_prompt_unigram_counter.keys()), len(direct_prompt_bigram_counter.keys()),
                      len(direct_prompt_trigram_counter.keys())]
prompt_attacks_list = [len(prompt_attacks_unigram_counter.keys()), len(prompt_attacks_bigram_counter.keys()),
                       len(prompt_attacks_trigram_counter.keys())]
paraphrase_attacks_list = [len(paraphrase_attacks_unigram_counter.keys()),
                           len(paraphrase_attacks_bigram_counter.keys()),
                           len(paraphrase_attacks_trigram_counter.keys())]
perturbation_attacks_list = [len(perturbation_attacks_unigram_counter.keys()),
                             len(perturbation_attacks_bigram_counter.keys()),
                             len(perturbation_attacks_trigram_counter.keys())]
data_mixing_list = [len(data_mixing_unigram_counter.keys()), len(data_mixing_bigram_counter.keys()),
                    len(data_mixing_trigram_counter.keys())]

human_list = [len(human_unigram_counter.keys()), len(human_bigram_counter.keys()), len(human_trigram_counter.keys())]

datasets = ['Uni.', 'Bigram', 'Tri.']

plt.rcParams.update({'font.size': 15})

# 创建一个更大的图和三个子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.4), sharey=True)  # 这里的figsize可以根据需要调整

# 设置一些通用的参数
bar_width = 0.15
opacity = 0.6
# error_config = {'ecolor': '0.3'}

# 第一个子图
index = range(len(datasets))
ax1.bar(index, arxiv_list, bar_width, alpha=opacity, color='#8ECFC9', label='Arxiv')
ax1.bar([p + bar_width for p in index], xsum_list, bar_width, alpha=opacity, color='#FFBE7A', label='Xsum')
ax1.bar([p + bar_width * 2 for p in index], writing_list, bar_width, alpha=opacity, color='#FA7F6F', label='Writing')
ax1.bar([p + bar_width * 3 for p in index], review_list, bar_width, alpha=opacity, color='#82B0D2', label='Review')

ax1.set_xlabel('')
ax1.set_ylabel('Counts', rotation=90)
ax1.set_ylim(0, 500000)
ax1.set_title('Multi-Domain', fontsize=15)
ax1.set_xticks([p + bar_width * 1.5 for p in index])
ax1.set_xticklabels(datasets)
ax1.legend(fontsize=11, framealpha=0.2)
ax1.tick_params(axis='y', labelrotation=90)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x*1e-3)))

# 第二个子图
ax2.bar(index, ChatGPT_list, bar_width, alpha=opacity, color='#8ECFC9', label='GPT-3.5')
ax2.bar([p + bar_width for p in index], Claude_list, bar_width, alpha=opacity, color='#FFBE7A', label='Claude')
ax2.bar([p + bar_width * 2 for p in index], PaLM_list, bar_width, alpha=opacity, color='#FA7F6F', label='PaLM-2')
ax2.bar([p + bar_width * 3 for p in index], Llama_list, bar_width, alpha=opacity, color='#82B0D2', label='Llama-2')

ax2.set_xlabel('N-Gram Distribution')
ax2.set_ylabel('')
ax2.set_ylim(0, 500000)
ax2.set_title('Multi-LLM', fontsize=15)
ax2.set_xticks([p + bar_width * 1.5 for p in index])
ax2.set_xticklabels(datasets)
ax2.legend(fontsize=11, framealpha=0.2)

# 第三个子图
ax3.bar(index, direct_prompt_list, bar_width, alpha=opacity, color='#8ECFC9', label='Direct')
ax3.bar([p + bar_width for p in index], prompt_attacks_list, bar_width, alpha=opacity, color='#FFBE7A', label='Prompt')
ax3.bar([p + bar_width * 2 for p in index], paraphrase_attacks_list, bar_width, alpha=opacity, color='#FA7F6F', label='Paraph.')
ax3.bar([p + bar_width * 3 for p in index], perturbation_attacks_list, bar_width, alpha=opacity, color='#82B0D2', label='Perturb')
ax3.bar([p + bar_width * 4 for p in index], data_mixing_list, bar_width, alpha=opacity, color='#BEB8DC', label='Mixing')
ax3.bar([p + bar_width * 5 for p in index], human_list, bar_width, alpha=opacity, color='#FFC0CB', label='Human')

ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_ylim(0, 500000)
ax3.set_title('Multi-Attack', fontsize=15)
ax3.set_xticks([p + bar_width * 2 for p in index])
ax3.set_xticklabels(datasets)
ax3.legend(fontsize=11, framealpha=0.2)

# # 调整子图之间的间距和布局
# plt.tight_layout(pad=3.0)
plt.subplots_adjust(left=0.07, right=0.99, wspace=0.05, top=0.88, bottom=0.25)
plt.savefig("n_gram.pdf")
plt.show()