import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Calibri
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.unicode_minus'] = False

# 数据
data = {
    "Likelihood": [63.71, 62.97, 67.97, 53.37, 51.77, 50.73, 57.92, 59.28, 88.48],
    "Rank": [51.34, 50.33, 57.08, 42.61, 41.49, 38.84, 41.67, 46.65, 83.86],
    "Entropy": [46.02, 46.97, 43.75, 25.06, 31.07, 16.53, 13.38, 15.99, 22.39],
    "LogRank": [64.43, 63.75, 68.52, 55.10, 52.78, 51.28, 57.44, 59.74,  88.46],
    "LRR": [65.47, 64.93, 68.53, 54.61, 52.73, 57.41, 57.09, 58.15, 85.99],
    "NPR": [48.37, 47.27, 53.49, 38.58, 38.83, 36.10, 37.60, 42.17, 80.03],
    "DetectGPT": [34.43, 34.93, 36.19, 11.54, 13.11, 11.84, 35.78, 34.69, 60.86],
    "DNA-GPT ": [64.92, 64.36, 68.36, 51.51, 47.09, 41.98, 57.63, 62.43, 87.80],
    "Revise-Detect.": [67.24, 66.36, 70.89, 54.50, 53.28, 50.63, 65.71, 67.96, 83.29],
    "Binoculars": [83.95, 83.30, 85.05, 77.47, 74.10, 74.70, 73.82, 74.34, 90.68],
    "Fast-Detect.": [58.52, 59.58, 60.70, 48.35, 36.56, 49.47, 61.31, 55.08, 76.03],
    "RoB-base": [99.98, 99.93, 99.56, 83.00, 91.81, 92.37, 79.99, 74.00, 97.34],
    "X-RoB-base": [99.92, 99.14, 98.49, 75.97, 92.73, 90.58, 84.25, 73.83, 93.43],
    "RoB-large": [99.78, 95.16, 99.87, 77.20, 82.85, 83.96, 86.08, 85.23, 96.68],
    "X-RoB-large": [99.01, 97.40, 99.31, 76.14, 85.89, 73.42, 86.35, 79.83, 97.21],
}

colors = sns.color_palette("husl", len(data))

labels = np.array(['Multi-Domain', 'Multi-LLM', 'Multi-Attack', 'Domain Generlization', 'LLM Generlization', 'Attack Generlization', 'Training-Time', 'Test-Time', 'Human Writing'])

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

fig, ax = plt.subplots(figsize=(9.6, 6), subplot_kw=dict(polar=True))
ax.set_facecolor('white')  # 设置雷达图背景颜色

markers = ['o', 'v', '^', 's', 'p', '*', 'h', 'H', 'D', 'X', 'd', 'P', '+', '.', 'x']

for idx, (baseline, scores) in enumerate(data.items()):
    scores_full = np.concatenate((scores, [scores[0]]))  # 完成数据闭环
    ax.fill(angles, scores_full, color=colors[idx], alpha=0.05)  # 填充颜色，增加透明度
    line = ax.plot(angles, scores_full, label=baseline, linewidth=3, color=colors[idx])  # 绘制线条
    plt.setp(line, linestyle='-')  # 设置线型、标记样式和大小

ax.set_xticks(angles[:-1])
ax.set_xticklabels([])

for i, angle in enumerate(angles[:-1]):
    angle_rad = angle + np.pi / len(labels)
    ha = 'center'
    if angle_rad > np.pi:
        angle_rad -= np.pi
        ha = 'center'
    ax.text(angle, ax.get_ylim()[1] + 5, labels[i], size=20, horizontalalignment=ha,
            verticalalignment='center', fontweight='bold')

ax.yaxis.grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.5)
ax.xaxis.grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.5)

ax.tick_params(axis='y', labelsize=18)

# plt.subplots_adjust(top=0.85, bottom=0.15)

ax.fill(angles, [0.1] * len(angles), color='gray', alpha=0.1)

legend = ax.legend(loc='center left', bbox_to_anchor=(1.17, 0.5), ncol=1, fontsize=18, framealpha=0.3, edgecolor='black')
frame = legend.get_frame()
frame.set_color('white')
frame.set_edgecolor('black')
frame.set_linewidth(1.5)

plt.subplots_adjust(left=-0.01, right=0.8, top=0.9, bottom=0.1)

fig.savefig('radar.png', bbox_inches='tight', transparent=True)
plt.show()