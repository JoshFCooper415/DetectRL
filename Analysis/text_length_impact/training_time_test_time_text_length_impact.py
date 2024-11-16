import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

fixed_label = {
    "likelihood": "Likelihood",
    "rank": "Rank",
    "logRank": "LogRank",
    "LRR": "LRR",
    "NPR": "NPR",
    "DetectGPT": "DetectGPT",
    "dna_gpt": "DNA-GPT",
    "revise": "Revise-Detect.",
    "bino": "Binoculars",
    "Fast_DetectGPT": "Fast-DetectGPT",
    "RoB-Base": "RoB-Base",
    "X-RoB-Base": "X-RoB-Base"
}

data = {}
for metric in ["likelihood", "rank", "logRank", "LRR", "NPR", "DetectGPT", "dna_gpt", "revise", "bino", "Fast_DetectGPT", "RoB-Base", "X-RoB-Base"]:
    res = []
    for length in ['20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '220', '240', '260', '280', '300', '320', '340', '360']:
        if "RoB-Base" == metric:
            with open(f"cross_length_{length}_test.jsonroberta_base_classifier/cross_length_180_test.json.roberta-base_result.json", "r") as f:
                score = json.load(f)["f1"]
                res.append(score)
        elif "X-RoB-Base" == metric:
            with open(f"cross_length_{length}_test.jsonxlm_roberta_base_classifier/cross_length_180_test.json.xlm-roberta-base_result.json", "r") as f:
                score = json.load(f)["f1"]
                res.append(score)
        else:
            with open(f"cross_length_{length}_test_transfer_result.json", "r") as f:
                score = json.load(f)[metric]["cross_length_180"]["f1"]
                res.append(score)
    data[metric] = res
    print(f"{metric}: {sum(res)/len(res)}")

colors = sns.color_palette("husl", len(data))

markers = [
    'o', 's', 'D', '^', 'v',
    '<', '>', 'p', '*',
    'X', 'H', 'P'  # New markers
]

x = range(len(data['likelihood']))

x_labels = ['20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '220', '240', '260', '280', '300', '320', '340', '360']

plt.figure(figsize=(10, 3))

# first figure
plt.subplot(1, 2, 1)
for idx, (label, y) in enumerate(data.items()):
    plt.plot(x, y, label=fixed_label[label], color=colors[idx], marker=markers[idx % len(markers)], markersize=5, linewidth=2)
plt.xticks(ticks=x, labels=[x_labels[i] if i % 3 == 0 else '' for i in range(len(x_labels))], fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 1.0)
plt.xlabel('Text Length', fontsize=14)
plt.ylabel('AUROC', fontsize=14)
plt.title('(a) Training-Time', fontsize=14)

data = {}
for metric in ["likelihood", "rank", "logRank", "LRR", "NPR", "DetectGPT", "dna_gpt", "revise", "bino", "Fast_DetectGPT", "RoB-Base", "X-RoB-Base"]:
    res = []
    for length in ['20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '220', '240', '260', '280', '300', '320', '340', '360']:
        if "RoB-Base" == metric:
            with open(f"cross_length_180_test.jsonroberta_base_classifier/cross_length_{length}_test.json.roberta-base_result.json", "r") as f:
                score = json.load(f)["f1"]
                res.append(score)
        elif "X-RoB-Base" == metric:
            with open(f"cross_length_180_test.jsonxlm_roberta_base_classifier/cross_length_{length}_test.json.xlm-roberta-base_result.json", "r") as f:
                score = json.load(f)["f1"]
                res.append(score)
        else:
            with open(f"cross_length_180_test_transfer_result.json", "r") as f:
                score = json.load(f)[metric][f"cross_length_{length}"]["f1"]
                res.append(score)
        data[metric] = res
        print(f"{metric}: {sum(res) / len(res)}")

    # second figure
    plt.subplot(1, 2, 2)
    for idx, (label, y) in enumerate(data.items()):
        plt.plot(x, y, label=fixed_label[label], color=colors[idx], marker=markers[idx % len(markers)], markersize=5, linewidth=2)
    plt.xticks(ticks=x, labels=[x_labels[i] if i % 3 == 0 else '' for i in range(len(x_labels))],fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1.0)
    plt.xlabel('Text Length', fontsize=14)
    plt.title('(b) Test-Time', fontsize=14)

    # plt.subplots_adjust(bottom=0.32, wspace=0.4)

    # shared legend
    lines, labels = plt.subplot(1, 2, 1).get_legend_handles_labels()
    plt.figlegend(lines, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.01), fontsize=11)

    plt.subplots_adjust(hspace = 0.2 ,wspace=0.2, top=0.9, bottom=0.37)

    plt.savefig('training_time_test_time_text_length_impact.pdf')
    plt.show()