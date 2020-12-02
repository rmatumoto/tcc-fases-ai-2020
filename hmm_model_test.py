#! python3
# hmm_model_test.py - Aplica o modelo HMM treinado em um novo conjunto de dados.

import pomegranate as pg
from pomegranate import hmm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix, classification_report

from low_filter import LowPass

pd.set_option('mode.chained_assignment', None)

#? Configurações para plot.
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#? Corte de regiões de transição.
data_slice = {
    'h5a.csv': slice(520, 3520),
    'h5b.csv': slice(510, 3530),
    'h5c.csv': slice(470, 3450),
    'h5d.csv': slice(500, 3400),
    'h5e.csv': slice(550, 3170),
    'h5f.csv': slice(500, 3510),
    #
    'h2a.csv': slice(630, 2050),
    'h2b.csv': slice(630, 3480),
    'h2c.csv': slice(630, 3480),
    'h2d.csv': slice(640, 3600),
    'h2e.csv': slice(660, 3630),
    'h2f.csv': slice(600, 3410),
    'h2g.csv': slice(830, 3800),
    #
    'h3a.csv': slice(610, 3660),
    'h3b.csv': slice(540, 3460),
    'h3c.csv': slice(600, 3800),
    'h3d.csv': slice(560, 3500),
    'h3e.csv': slice(540, 3640),
    'h3f.csv': slice(540, 2770),
    'h3g.csv': slice(570, 3480),
    #
    'h0a.csv': slice(590, 2680),
    'h0b.csv': slice(480, 2720),
    'h0c.csv': slice(480, 2730),
    'h0d.csv': slice(560, 2850),
    'h0e.csv': slice(550, 3260),
    'h0f.csv': slice(560, 3440),
    'h0g.csv': slice(620, 3460),
    'h0h.csv': slice(610, 3420),
    'h0i.csv': slice(600, 3430)
}


def load_data(path, test=False):
    """Carrega os dados e os coloca em um pd.DataFrame. São agregadas todas as séries de medição.

    Args:
        path (str): Diretório dos arquivos.
        test (bool, optional): Se serão usados os dados de teste ou não. Defaults to False.

    Returns:
        pd.DataFrame: Data frame com os dados.
    """

    if test:
        path = path + 'test/'
    else:
        path = path + 'train/'

    data = pd.read_csv(path + os.listdir(path)[0], index_col=0)[data_slice[os.listdir(path)[0]]].reset_index(drop=True)

    for filename in os.listdir(path)[1:]:
        temp = pd.read_csv(path + filename, index_col=0)[data_slice[filename]].reset_index(drop=True)

        data = pd.concat([data, temp], axis=0, ignore_index=True)

    return data


def lp_filter(cutoff_freq, sampl_freq, approx_order, data):
    """Filtro passa-baixa.

    Args:
        cutoff_freq (float): Frequência de corte.
        sampl_freq (float): Frequência de amostragem.
        approx_order (int): Ordem para aproximação da resposta.
        data (array_like): Dados que serão filtrados.

    Returns:
        array_like: Dados filtrados.
    """

    filtro = LowPass(cutoff_freq, sampl_freq, approx_order)
    b, a = filtro.get_tf()
    filt_data = filtro.get_response(b, a, data)

    return filt_data


def create_sets(data, n_obs_per_seq) -> np.ndarray:
    """Modifica formato dos dados para serem fornecidos ao modelo.

    Args:
        data (pd.DataFrame): Data frame com dados originais.
        n_obs_per_seq (int): Número de observações em cada sequência.

    Returns:
        np.ndarray: Dados formatados.
    """

    # Remoção de valores finais para adequação no formato requerido.
    n_rows_drop = data.shape[0] % n_obs_per_seq
    data = data[:-n_rows_drop]

    # Vetor de rótulos.
    labels = np.array(data.loc[:, 'new_stL'] - 1, dtype=str)

    # Métricas de normalização provindas do treinamento.
    norm_path = 'hmm_model_stats.csv'

    mu_sigma = pd.read_csv(norm_path, index_col=0)
    mean = mu_sigma.iloc[0, :].values
    std = mu_sigma.iloc[1, :].values

    # Reshape para formato: (#sequências, #observações) e (#sequências, #observações, #features).
    observations = np.array(data.loc[:, ['gxL', 'gyL', 'gzL', 'axL', 'ayL', 'azL']])
    observations = (observations - mean.T) / std.T
    observations = observations.reshape((-1, n_obs_per_seq, 6))

    labels = labels.reshape((-1, n_obs_per_seq))

    # Labels numéricas para plot da curva.
    num_labels = np.array(labels.flatten(), dtype=np.int32)

    return observations, labels, num_labels


def load_model():
    """Carrega o modelo treinado.

    Returns:
        hmm.HiddenMarkovModel: HMM.
    """

    model = hmm.HiddenMarkovModel().from_json('hmm_model.json')

    return model


def predict_plot(model, observ, labels, lim, tol=0):
    """Comparação entre os estados mais prováveis e os reais. Plota matriz de confusão dos resultados. Gera sensibilidade e especificidade.

    Args:
        model (hmm.HiddenMarkovModel): HMM.
        observ (list): Sequência de observações.
        labels (list): Sequência dos estados ocultos.
        lim (list): Limites para visualização das curvas.
        tol (int, optional): Largura da janela de rejeição de erros ao redor dos rótulos. Defaults to 0.
    """

    preds = list()

    # Predição pelo algoritmo de Viterbi.
    for i in observ:
        p = model.predict(i, algorithm='viterbi')[1:]
        preds.append(p)

    preds = np.array(preds, dtype=np.int32)
    states = preds.flatten()

    if tol:
        states = tolerance_predictions(labels, tol, states)

    gs_kw = dict(width_ratios=[3, 1], height_ratios=[2, 1])
    fig, axs = plt.subplots(2, 2, figsize=(12,5), constrained_layout=True, gridspec_kw=gs_kw)

    fig.suptitle('Classificação de fases com HMM', weight='bold', size='x-large')

    axs[0][0].set_title('Fases classificadas vs. fases reais')
    axs[0][0].set_ylabel('Fases')
    axs[0][0].set_yticks([0, 1, 2, 3])
    axs[0][0].set_yticklabels(['FF', 'HO', 'SW', 'HS'])
    axs[0][0].set_xlim(lim)

    axs[0][0].scatter([i for i in range(len(states))], states, marker='o', edgecolor='k', label='Predições', c='#2ca02c', s=40)

    axs[0][0].plot(labels, marker='.', label='Rótulos')

    axs[1][0].set_title('Erro na classificação')
    axs[1][0].set_ylabel('Diferença')
    axs[1][0].set_xlabel('Amostras')
    axs[1][0].set_xlim(lim)

    axs[1][0].get_shared_x_axes().join(axs[0][0], axs[1][0])

    axs[1][0].plot(labels - states, c=colors[3])

    # Matriz de confusão dos resultados.
    cm = confusion_matrix(labels, states)

    # Sensibilidade e especificidade.
    sens, spec = sens_spec(cm)
    print(sens, spec)

    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axs[0][1])

    axs[0][1].set_title('Matriz de confusão')
    axs[0][1].set_ylabel('Fases reais')
    axs[0][1].set_xlabel('Fases classificadas')

    handles, labels = axs[0][0].get_legend_handles_labels()

    axs[1][1].legend(handles, labels, loc='center')
    axs[1][1].axis('off')


def get_predictions(model, observ):
    """Dados o modelo e sequências de observações, retorna a sequência de estados ocultos.

    Args:
        model (hmm.HiddenMarkovModel): HMM.
        observ (list): Lista de observações.

    Returns:
        list: Sequência de estados ocultos.
    """
    preds = list()

    # Predição pelo algoritmo de Viterbi.
    for i in observ:
        p = model.predict(i, algorithm='viterbi')[1:]
        preds.append(p)

    preds = np.array(preds, dtype=np.int32)
    states = preds.flatten()

    return states


def tolerance_predictions(labels, width, predictions):
    """Correção nas predições considerando intervalo de largura definida em torno dos rótulos.

    Args:
        labels (list): Rótulos.
        width (int): Largura da janela de rejeição.
        predictions (list): Sequência de estados ocultos.

    Returns:
        list: Rótulos corrigidos.
    """

    label_shift_left, label_shift_right = shift_labels(labels, width)

    dif = labels - predictions
    better_predictions = list()

    for i in range(len(dif)):
        if dif[i]:
            if predictions[i] == label_shift_left[i]:
                better_predictions.append(labels[i])
            else:
                if predictions[i] == label_shift_right[i]:
                    better_predictions.append(labels[i])
                else:
                    better_predictions.append(predictions[i])
        else:
            better_predictions.append(predictions[i])

    return better_predictions


def shift_labels(labels, width):
    """Cria cópias da sequência de rótulos à direita e à esquerda da original.

    Args:
        labels (list): Sequência de estados ocultos.
        width (int): Largula da janela de rejeição.

    Returns:
        list: Rótulos deslocados.
    """

    # Meia janela à esquerda.
    label_shift_left = np.roll(labels, -width)
    label_shift_left[-width:] = label_shift_left[-(width+1)]

    # Meia janela à direita.
    label_shift_right = np.roll(labels, width)
    label_shift_right[:width] = label_shift_right[width]

    return label_shift_left, label_shift_right


def sens_spec(cm):
    """Retorna valores da sensibilidade e especificidade da detecção de cada estado oculto.

    Args:
        cm (array_like): Matriz de confusão.

    Returns:
        list: Sensibilidade, especificidade.
    """

    sens = list()
    spec = list()

    for i in range(4):
        tp = cm[i][i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(np.diag(cm)) - tp

        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))

    return sens, spec


#? -----------------------
#? Carregamento dos dados.
#? -----------------------
data_path = '../walking/data2/'
data = load_data(data_path, test=True)


#? --------------------------------------
#? Filtragem com Butterworth passa-baixa.
#? --------------------------------------
order = 2
fs = 50.0
cutoff_w = 20
cutoff_a = 2

data.loc[:, ['gxL', 'gyL', 'gzL']] = lp_filter(cutoff_w, fs, order, data.loc[:, ['gxL', 'gyL', 'gzL']])
data.loc[:, ['axL', 'ayL', 'azL']] = lp_filter(cutoff_a, fs, order, data.loc[:, ['axL', 'ayL', 'azL']])

# print(data.groupby(by='new_stL').count())

#? ----------------------
#? Divisão dos conjuntos.
#? ----------------------
observations, labels, num_labels = create_sets(data, 25)
tol_width = 0
num_labels = np.array(num_labels).flatten()

#? ------------------------
#? Previsão de novos dados.
#? ------------------------
# Carregamento do modelo.
hmm_model = load_model()

# Estados ocultos previstos pelo algoritmo de Viterbi.
predictions = get_predictions(hmm_model, np.array(list(observations)))

# print(classification_report(num_labels, predictions, target_names=['FF', 'HO', 'SW', 'HS']))

# Limites para visualização dos 500 pontos centrais.
limites = [int(len(data)/2) - 250, int(len(data)/2) + 250]


# Resultado da classificação + matriz de confusão.
predict_plot(hmm_model, np.array(list(observations)), num_labels, limites, tol_width)
plt.show()