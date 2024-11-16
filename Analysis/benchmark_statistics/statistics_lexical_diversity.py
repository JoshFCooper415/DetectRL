import seaborn as sns
from matplotlib import pyplot as plt
import json


def lexical_diversity(text):
    tokens = text.split()
    types = set(tokens)
    diversity_score = len(types) / len(tokens)
    return diversity_score


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

arxiv_list = []
xsum_list = []
writing_list = []
review_list = []
for item in arxiv:
    if item["label"] == "llm":
        arxiv_list.append(lexical_diversity(item["text"]))
for item in xsum:
    if item["label"] == "llm":
        xsum_list.append(lexical_diversity(item["text"]))
for item in writing:
    if item["label"] == "llm":
        writing_list.append(lexical_diversity(item["text"]))
for item in review:
    if item["label"] == "llm":
        review_list.append(lexical_diversity(item["text"]))

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

ChatGPT_list = []
Claude_list = []
PaLM_list = []
Llama_list = []

for item in ChatGPT:
    if item["label"] == "llm":
        ChatGPT_list.append(lexical_diversity(item["text"]))
for item in Claude:
    if item["label"] == "llm":
        Claude_list.append(lexical_diversity(item["text"]))
for item in PaLM:
    if item["label"] == "llm":
        PaLM_list.append(lexical_diversity(item["text"]))
for item in Llama:
    if item["label"] == "llm":
        Llama_list.append(lexical_diversity(item["text"]))

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

direct_prompt_list = []
prompt_attacks_list = []
paraphrase_attacks_list = []
perturbation_attacks_list = []
data_mixing_list = []
human_list = []

for item in direct_prompt:
    if item["label"] == "llm":
        direct_prompt_list.append(lexical_diversity(item["text"]))
    elif item["label"] == "human":
        human_list.append(lexical_diversity(item["text"]))
for item in prompt_attacks:
    if item["label"] == "llm":
        prompt_attacks_list.append(lexical_diversity(item["text"]))
for item in paraphrase_attacks:
    if item["label"] == "llm":
        paraphrase_attacks_list.append(lexical_diversity(item["text"]))
for item in perturbation_attacks:
    if item["label"] == "llm":
        perturbation_attacks_list.append(lexical_diversity(item["text"]))
for item in data_mixing:
    if item["label"] == "llm":
        data_mixing_list.append(lexical_diversity(item["text"]))

plt.rcParams.update({'font.size': 15})

fig, axes = plt.subplots(1, 3, figsize=(8, 2.4), sharey=True)

# Plot comparison for Text Length Distribution - Domains
sns.kdeplot(arxiv_list, color='#8ECFC9', label='ArXiv', ax=axes[0])
sns.kdeplot(xsum_list, color='#FFBE7A', label='XSum', ax=axes[0])
sns.kdeplot(writing_list, color='#FA7F6F', label='Writing', ax=axes[0])
sns.kdeplot(review_list, color='#82B0D2', label='Review', ax=axes[0])
axes[0].set_title('Multi-Domain', fontsize=15)
axes[0].set_xlabel('')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0.0, 1.2)
axes[0].set_ylim(0, 8)
axes[0].legend(fontsize=11, framealpha=0.2)

# Plot comparison for Text Length Distribution - LLMs
sns.kdeplot(ChatGPT_list, color='#8ECFC9', label='GPT-3.5', ax=axes[1])
sns.kdeplot(Claude_list, color='#FFBE7A', label='Claude', ax=axes[1])
sns.kdeplot(PaLM_list, color='#FA7F6F', label='PaLM-2', ax=axes[1])
sns.kdeplot(Llama_list, color='#82B0D2', label='Llama-2', ax=axes[1])
axes[1].set_title('Multi-LLM', fontsize=15)
axes[1].set_xlabel('Lexical Diversity Distribution', fontsize=15)
axes[1].set_ylabel('')
axes[1].set_xlim(0.0, 1.2)
axes[1].set_ylim(0, 8)
axes[1].legend(fontsize=11, framealpha=0.2)

# Plot comparison for Text Length Distribution - Attacks
sns.kdeplot(direct_prompt_list, color='#8ECFC9', label='Direct', ax=axes[2])
sns.kdeplot(prompt_attacks_list, color='#FFBE7A', label='Prompt', ax=axes[2])
sns.kdeplot(paraphrase_attacks_list, color='#FA7F6F', label='Paraph.', ax=axes[2])
sns.kdeplot(perturbation_attacks_list, color='#82B0D2', label='Perturb.', ax=axes[2])
sns.kdeplot(data_mixing_list, color='#BEB8DC', label='Mixing', ax=axes[2])
sns.kdeplot(human_list, color='#FFC0CB', label='Human', ax=axes[2])
axes[2].set_title('Multi-Attack', fontsize=15)
axes[2].set_xlabel('')
axes[2].set_ylabel('')
axes[2].set_xlim(0.0, 1.2)
axes[2].set_ylim(0, 8)
axes[2].legend(fontsize=11, framealpha=0.2)

plt.subplots_adjust(left=0.07, right=0.99, wspace=0.05, top=0.88, bottom=0.25)

# save as lexical_diversity.pdf
plt.savefig("lexical_diversity.pdf")

plt.show()
