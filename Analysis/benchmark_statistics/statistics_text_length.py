from matplotlib import pyplot as plt
import seaborn as sns
import json


"""
Cross Domains

"""
with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_arxiv_test.json", "r", encoding="utf-8") as f:
    arxiv = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_xsum_test.json", "r", encoding="utf-8") as f:
    xsum = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_writing_prompt_test.json", "r", encoding="utf-8") as f:
    writing = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossDomains\cross_domains_yelp_review_test.json", "r", encoding="utf-8") as f:
    review = json.load(f)

human_arxiv_lengths = []
human_xsum_lengths = []
human_writing_lengths = []
human_review_lengths = []
llm_arxiv_lengths = []
llm_xsum_lengths = []
llm_writing_lengths = []
llm_review_lengths = []

for item in arxiv:
    if item["label"] == "llm":
        llm_arxiv_lengths.append(len(item["text"].split()))
    elif item["label"] == "human":
        human_arxiv_lengths.append(len(item["text"].split()))

for item in xsum:
    if item["label"] == "llm":
        llm_xsum_lengths.append(len(item["text"].split()))
    elif item["label"] == "human":
        human_xsum_lengths.append(len(item["text"].split()))

for item in writing:
    if item["label"] == "llm":
        llm_writing_lengths.append(len(item["text"].split()))
    elif item["label"] == "human":
        human_writing_lengths.append(len(item["text"].split()))

for item in review:
    if item["label"] == "llm":
        llm_review_lengths.append(len(item["text"].split()))
    elif item["label"] == "human":
        human_review_lengths.append(len(item["text"].split()))

"""
Cross LLMs

"""
with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_ChatGPT_test.json", "r", encoding="utf-8") as f:
    ChatGPT = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_Claude-instant_test.json", "r", encoding="utf-8") as f:
    Claude = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_Google-PaLM_test.json", "r", encoding="utf-8") as f:
    PaLM = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\CrossLLMs\cross_llms_Llama-2-70b_test.json", "r", encoding="utf-8") as f:
    Llama = json.load(f)


ChatGPT_lengths = []
Claude_lengths = []
PaLM_lengths = []
Llama_lengths = []

for item in ChatGPT:
    if item["label"] == "llm":
        ChatGPT_lengths.append(len(item["text"].split()))

for item in Claude:
    if item["label"] == "llm":
        Claude_lengths.append(len(item["text"].split()))

for item in PaLM:
    if item["label"] == "llm":
        PaLM_lengths.append(len(item["text"].split()))

for item in Llama:
    if item["label"] == "llm":
        Llama_lengths.append(len(item["text"].split()))

"""
Cross Attacks

"""
with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\DirectPrompt\direct_prompt_test.json", "r", encoding="utf-8") as f:
    direct_prompt = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\PromptAttacks\prompt_attacks_test.json", "r", encoding="utf-8") as f:
    prompt_attacks = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\ParaphraseAttacks\paraphrase_attacks_attack_test.json", "r", encoding="utf-8") as f:
    paraphrase_attacks = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\AdversarialAttacks\adversarial_attacks_test.json", "r", encoding="utf-8") as f:
    perturbation_attacks = json.load(f)

with open(r"D:\gitHome\WrokFlow\Robustness_Benchmark\Eval_Bench\Doc\Mixed\mixed_attack_test.json", "r", encoding="utf-8") as f:
    data_mixing = json.load(f)

human_lengths = []
direct_prompt_lengths = []
prompt_attacks_lengths = []
paraphrase_attacks_lengths = []
perturbation_attacks_lengths = []
data_mixing_attacks_lengths = []

for item in direct_prompt:
    if item["label"] == "llm":
        direct_prompt_lengths.append(len(item["text"].split()))
    elif item["label"] == "human":
        human_lengths.append(len(item["text"].split()))

for item in prompt_attacks:
    if item["label"] == "llm":
        prompt_attacks_lengths.append(len(item["text"].split()))

for item in paraphrase_attacks:
    if item["label"] == "llm":
        paraphrase_attacks_lengths.append(len(item["text"].split()))

for item in perturbation_attacks:
    if item["label"] == "llm":
        perturbation_attacks_lengths.append(len(item["text"].split()))

for item in data_mixing:
    if item["label"] == "llm":
        data_mixing_attacks_lengths.append(len(item["text"].split()))

plt.rcParams.update({'font.size': 15})
# 3 pics
fig, axes = plt.subplots(1, 3, figsize=(8, 2.4), sharey=True)

# Plot comparison for Text Length Distribution - Domains
sns.kdeplot(llm_arxiv_lengths, color='#8ECFC9', label='ArXiv', ax=axes[0])
sns.kdeplot(llm_xsum_lengths, color='#FFBE7A', label='XSum', ax=axes[0])
sns.kdeplot(llm_writing_lengths, color='#FA7F6F', label='Writing', ax=axes[0])
sns.kdeplot(llm_review_lengths, color='#82B0D2', label='Review', ax=axes[0])

axes[0].set_title('Multi-Domain', fontsize=15)
axes[0].set_xlabel('', fontsize=15)
axes[0].set_ylabel('Frequency', fontsize=15, rotation=90)
axes[0].set_xlim(0, 500)
axes[0].set_ylim(0, 0.01)
axes[0].legend(fontsize=11, loc='upper right', framealpha=0.2)
axes[0].tick_params(axis='y', labelrotation=90)

# Plot comparison for Text Length Distribution - LLMs
sns.kdeplot(ChatGPT_lengths, color='#8ECFC9', label='GPT-3.5', ax=axes[1])
sns.kdeplot(Claude_lengths, color='#FFBE7A', label='Claude', ax=axes[1])
sns.kdeplot(PaLM_lengths, color='#FA7F6F', label='PaLM-2', ax=axes[1])
sns.kdeplot(Llama_lengths, color='#82B0D2', label='Llama-2', ax=axes[1])
axes[1].set_title('Multi-LLM', fontsize=14)
axes[1].set_xlabel('Text Length Distribution', fontsize=15)
axes[1].set_ylabel('', fontsize=15)
axes[1].set_xlim(0, 500)
axes[1].set_ylim(0, 0.01)
axes[1].legend(fontsize=11, loc='upper right', framealpha=0.2)

# Plot comparison for Text Length Distribution - Attacks
sns.kdeplot(direct_prompt_lengths, color='#8ECFC9', label='Direct', ax=axes[2])
sns.kdeplot(prompt_attacks_lengths, color='#FFBE7A', label='Prompt', ax=axes[2])
sns.kdeplot(paraphrase_attacks_lengths, color='#FA7F6F', label='Paraph.', ax=axes[2])
sns.kdeplot(perturbation_attacks_lengths, color='#82B0D2', label='Perturb', ax=axes[2])
sns.kdeplot(data_mixing_attacks_lengths, color='#BEB8DC', label='Mixing', ax=axes[2])
sns.kdeplot(human_lengths, color='#FFC0CB', label='Human', ax=axes[2])

axes[2].set_title('Multi-Attack', fontsize=15)
axes[2].set_xlabel('', fontsize=15)
axes[2].set_ylabel('', fontsize=15)
axes[2].set_xlim(0, 500)
axes[2].set_ylim(0, 0.01)
axes[2].legend(fontsize=11, loc='upper right', framealpha=0.2)

plt.subplots_adjust(left=0.07, right=0.99, wspace=0.05, top=0.88, bottom=0.25)
plt.savefig("text_length.pdf")
plt.show()