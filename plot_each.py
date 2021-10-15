from config import  News20Config, CnewsConfig
import json
import seaborn as sns
import matplotlib.pyplot as plt

cfg = CnewsConfig()
dataset = []
with open(cfg.each_acc_dir, 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        dataset.append(line)


sns.set_theme()
sns.set(font_scale=1.2)
# sns.set(font="simhei")
fig = plt.figure(figsize=(10,6))
markers = ['o', '^', 's']
for idx, line in enumerate(dataset):
    x = line['labels']
    y = line['acc']
    label = line['model_name']
    ax = sns.lineplot(x=x, y=y, label=label, marker=markers[idx], dashes=True,
                      sizes=500)
    ax.lines[idx].set_linestyle('--')
plt.xlabel('Label')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('label_acc.png')
plt.show()
