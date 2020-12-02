#! python3
# plot_sp_es.py - Curvas janela de tolerância vs. {sensibilidade, especificidade}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.style.use('ggplot')

data_fnn = pd.read_csv('se_es_fnn.csv', index_col=0)
data_hmm = pd.read_csv('se_es_hmm.csv', index_col=0)
data_lstm = pd.read_csv('se_es_lstm.csv', index_col=0)

fig, axs = plt.subplots(2, 4, figsize=(12,5), constrained_layout=True, sharex=True, sharey=True)

fig.suptitle('Sensibilidade e especificidade com tolerâncias de erro em número de amostras', weight='bold', size='x-large')

labels = ['FF', 'HO', 'SW', 'HS']
markers = ['o', 'X', 's']

for i in range(4):
    axs[0][i].plot(data_fnn.iloc[:, i], marker=markers[0], c=colors[0])
    axs[1][i].plot(data_fnn.iloc[:, 4+i], marker=markers[0], c=colors[0])

    axs[0][i].plot(data_hmm.iloc[:, i], marker=markers[1], c=colors[1])
    axs[1][i].plot(data_hmm.iloc[:, 4+i], marker=markers[1], c=colors[1])

    axs[0][i].plot(data_lstm.iloc[:, i], marker=markers[1], c=colors[2])
    axs[1][i].plot(data_lstm.iloc[:, 4+i], marker=markers[1], c=colors[2])

    axs[0][i].axhline(0.95, c='r')
    axs[1][i].axhline(0.95, c='r')
    axs[0][i].set_title(labels[i])


axs[0][0].set_ylabel('Sensibilidade')
axs[1][0].set_ylabel('Especificidade')
axs[1][0].set_xlabel('Tolerância')
axs[1][1].set_xlabel('Tolerância')
axs[1][2].set_xlabel('Tolerância')
axs[1][3].set_xlabel('Tolerância')


fig.legend(['FNN', 'HMM', 'LSTM'], loc='center', ncol=3, bbox_to_anchor=(0.5, 0.15))
plt.show()

