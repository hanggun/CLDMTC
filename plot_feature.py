from config import News20Config
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.cm as cm

cfg = News20Config()
vectors = []
label_ids = []
labels = []
with open(cfg.batch_feature_dir, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        line = json.loads(line)
        vectors.append(line['vector'])
        label_ids.append(line['label_id'])
        if line['label'] == 'hard':
            labels.append('hard negative mixing')
        elif idx == 15:
            labels.append(line['label'].split('.')[-1]+'-Aug')
        else:
            if line['label'].split('.')[-1] == 'misc':
                labels.append('.'.join(line['label'].split('-')[-1].split('.')[-2:]))
            elif line['label'].split('.')[-1] == 'x':
                labels.append(line['label'].split('.')[-2])
            else:
                labels.append(line['label'].split('.')[-1])
vectors = np.array(vectors)[:-7]
labels = labels[:-7]
label_ids = label_ids[:-7]
id2label = {id:label for id, label in zip(label_ids, labels)}

fig, ax = plt.subplots(1, 1, figsize=(6,6), dpi=300)
sns.set(font_scale=1)
# fig = plt.figure(figsize=(18,6))
# axes = fig.add_subplot(1, 2, 1)
cos_sim = cosine_similarity(vectors, vectors)
dt = pd.DataFrame(np.round(cos_sim, 2), columns=labels, index=labels)
sns.heatmap(dt, annot=False, fmt=".2f", linewidths=.5, cmap='OrRd', square=True, cbar=True, cbar_kws={"shrink": .72})
# plt.title('Scatter plot pythonspot.com', y=-0.01)
# plt.text(15, -1, "Correlation Graph between Citation & Favorite Count")
plt.tight_layout()
plt.savefig('pic.png', dpi=300)
plt.show()

# axes1 = fig.add_subplot(1, 2, 2)
X_embedded = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(vectors)
X_embedded /= 100
fig, ax = plt.subplots(1, 1, figsize=(10,6))
set_labels = []
for label in labels:
    if label not in set_labels:
        set_labels.append(label)
length = len(set_labels)
x = np.arange(length)
ys = [i+x+(i*x)**2 for i in range(length)]
colors = ['red', 'orange', 'gold', 'yellow', 'green', 'aquamarine',
          'cyan', 'skyblue', 'blue', 'mediumorchid', 'magenta', 'pink']
cdict = {label:c for label, c in zip(set_labels, colors)}
for g in np.unique(label_ids):
    ix = np.where(label_ids == g)
    ax.scatter(X_embedded[:,0][ix], X_embedded[:,1][ix], c=cdict[id2label[g]], label=id2label[g], s=100)

for i, txt in enumerate(labels):
    if i == 16:
        ax.annotate(txt, (X_embedded[:, 0][i], X_embedded[:, 1][i]), fontsize=16)
    elif i == 18:
        ax.annotate(txt, (X_embedded[:, 0][i], X_embedded[:, 1][i]), fontsize=16)
    else:
        ax.annotate(txt, (X_embedded[:,0][i], X_embedded[:,1][i]), fontsize=16)
plt.axis('off')
plt.tight_layout()
# plt.subplots_adjust(left=0, wspace=0.3)
plt.savefig('pic1.png')
plt.show()