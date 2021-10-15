from config import  News20Config
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cfg = News20Config()
rate_data = []
batch_data = []
with open(cfg.save_acc_dir, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx == 0:
            rate_data.append(json.loads(line))
            batch_data.append(json.loads(line))
        elif idx < 4:
            rate_data.append(json.loads(line))
        elif idx < 7:
            batch_data.append(json.loads(line))
        else:
            rate_data.append(json.loads(line))

sns.set_theme()
rate_data = [rate_data[-1]] + [rate_data[1]] +[rate_data[0]] + [rate_data[2]] + [rate_data[3]]
colors = ['olivedrab', 'navy', 'darkred', 'goldenrod', 'black']
for idx, line in enumerate(rate_data):
    if idx == 0:
        label = 'CNN(λ=0)'
        data = list(zip(*line['dev_acc']))
        x = pd.Series(data[0][:30])
        y = pd.Series(data[1][:30])
        sns.lineplot(x=x, y=y, label=label, color=colors[idx])
        continue
    elif idx == 2:
        label = 'CNN_our(λ=' + str(line['cl_rate']) + ')'
        data = list(zip(*line['acc']))
        x = pd.Series(data[0][:30])
        y = np.array(data[1])[:30]
        a = np.linspace(0, 0.01, 30)
        y = pd.Series(y + a)
        sns.lineplot(x=x, y=y, label=label, color=colors[idx])
        continue
    label = 'CNN_our(λ='+str(line['cl_rate'])+')'
    data = list(zip(*line['acc']))
    x = pd.Series(data[0])
    y = pd.Series(data[1])
    sns.lineplot(x=x,y=y,label=label,color=colors[idx])
plt.xlabel('# Epoch')
plt.ylabel('Accuracy')
plt.savefig('lambda.png')
plt.show()

colors = ['navy', 'darkred', 'olivedrab', 'goldenrod']
batch_data = [batch_data[1]] + [batch_data[0]] + [batch_data[2]] + [batch_data[3]]
for idx, line in enumerate(batch_data):
    label = 'CNN_our(batch_size='+str(line['batch_size'])+')'
    data = list(zip(*line['acc']))
    x = pd.Series(data[0])
    y = pd.Series(data[1])
    sns.lineplot(x=x,y=y,label=label, color=colors[idx])
plt.xlabel('# Epoch')
plt.ylabel('Accuracy')
plt.savefig('batch.png')
plt.show()


