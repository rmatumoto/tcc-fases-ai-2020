#! python3
# lstm_model_test.py -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error,  explained_variance_score

from low_pass import LowPass

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
    """Carrega os dados e os coloca em um DataFrame. São agregados todas as séries de medição

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


def create_sets(data, train_features, norm_path=None):
    """Divide dados em conjuntos de treinamento e validação.

    Args:
        data (pd.DataFrame): Data frame com dados.
        train_features (list): Lista das colunas que contém as features desejadas.
        normalize (str, optional): Caminho para vetor que normaliza o conjunto de dados nas métricas de treinamento. Defaults to None.

    Returns:
        pd.DataFrame: Data frames contendo todos os dados, somentes os de treino e os de validação.
    """

    features = data[train_features]

    if norm_path:
        mu_sigma = pd.read_csv(norm_path, index_col=0)
        train_mean = mu_sigma.iloc[0, :]
        train_std = mu_sigma.iloc[1, :]

        features = (features - train_mean) / train_std

    return features


def split_sets(data, train_features, norm_path, input_size, output_size, class_labels):
    """Separa dados de teste nos grupos de previsão e de rótulos.

    Args:
        data (pd.DataFrame): Data frame com dados.
        train_features (list): Lista com features que serão usadas.
        input_size (int): Quantidade de amostras que será utilizada para previsão.
        output_size (int): Qual ponto a partir do histórico será previsto.

    Returns:
        array_like: Dados para serem estruturados como sequências.
    """

    # Dados crus.
    feat = create_sets(data, train_features, norm_path)

    # Índices para corte dos dados.
    start_idx = input_size + output_size
    end_idx = len(feat) - input_size - output_size

    test_data = feat.iloc[:end_idx].values
    test_label = feat.iloc[start_idx:]
    test_class_labels = class_labels.iloc[start_idx:]

    return test_data, test_label, test_class_labels


def create_sequences(features, targets, seq_len, batch_size):
    """Transforma dados tabelados em dados sequenciais.

    Por exemplo, se 3 dados são usados como histórico para previsão do seguinte, tem-se, com lotes de tamanho:

    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    features = [ [0, 1, 2], [1, 2, 3] ],
               [ [2, 3, 4], [3, 4, 5] ],
               [ [4, 5, 6] ]

    targets = [ [3, 4],
                [5, 6],
                [7] ]

    Args:
        features (array_like): Array com dados históricos.
        targets (array_like): Array com valores que serão previstos.
        seq_len (int): Comprimento da sequência.
        batch_size (int): Tamanho do minilote.

    Returns:
        tf.data.Dataset: Dataset estruturado.
    """
    dataset = keras.preprocessing.timeseries_dataset_from_array(
        features,
        targets,
        sequence_length=seq_len,
        batch_size=batch_size
    )

    return dataset


def load_model(mode):
    """Carrega o modelo treinado.

    Args:
        mode (int): Tipo de resposta, classificatória (0) ou preditiva (1).

    Returns:
        tf.keras.Model: Modelo LSTM.
    """

    if mode:
        model_name = 'lstm_model_pred.h5'
    else:
        model_name = 'lstm_model_class.h5'

    model = keras.models.load_model(model_name)

    return model


def comp_prediction(model, dataset, lim):
    """Comparação das previsões feitas pelo modelo e valores reais.

    Args:
        model (tf.keras.Model): Modelo LSTM.
        dataset (tf.data.Dataset): Dataset avaliado pelo modelo.
        lim (list): Limites para plot da curva para melhor visualização.
    """

    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12,5), constrained_layout=True)

    lbls = [r'$ \omega_x $', r'$ \omega_y $', r'$ \omega_z $', r'$ a_x $', r'$ a_y $', r'$ a_z $']

    for i in range(6):
        for x, y in dataset.take(1):
            axs[i//3, i%3].plot(np.transpose(y)[i])
            axs[i//3, i%3].plot(np.transpose(model.predict(x))[i])
            axs[i//3, i%3].set_ylabel(lbls[i])
            axs[i//3, i%3].legend(['Rótulos', 'Predições'], loc=4)

    axs[0][0].set_xlim(lim)
    axs[1][0].set_xlabel('Amostras')
    axs[1][1].set_xlabel('Amostras')
    axs[1][2].set_xlabel('Amostras')

    fig.suptitle('Previsão dos sinais inerciais vs. rótulos', size='x-large', weight='bold')


def comp_classification(model, dataset, lim, tol=0):
    """Comparação das classificações feitas pelo modelo e valores reais. Retorna matriz de confusão também.

    Args:
        model (tf.keras.Model): Modelo LSTM.
        dataset (tf.data.Dataset): Dataset avaliado pelo modelo.
        lim (list): Limites para plot da curva para melhor visualização.
        tol (int, optional): Largura da janela de tolerância. Defautls to 0.
    """

    gs_kw = dict(width_ratios=[3, 1], height_ratios=[2, 1])
    fig, axs = plt.subplots(2, 2, figsize=(12,5), constrained_layout=True, gridspec_kw=gs_kw)

    fig.suptitle('Classificação de fases com rede LSTM', weight='bold', size='x-large')

    for inputs, targets in dataset.take(1):

        n = len(targets)
        predictions = model.predict(inputs).argmax(axis=1)

        if tol:
            pred = tolerance_predictions(targets, tol, predictions.flatten())
            predictions = np.array(pred, dtype=int)

        axs[0][0].set_title('Fases classificadas vs. fases reais')
        axs[0][0].set_ylabel('Fases')
        axs[0][0].set_yticks([0, 1, 2, 3])
        axs[0][0].set_yticklabels(['FF', 'HO', 'SW', 'HS'])
        axs[0][0].set_xlim(lim)

        axs[0][0].scatter([i for i in range(n)], predictions, marker='o', edgecolor='k', label='Predições', c='#2ca02c', s=40)

        axs[0][0].plot(targets, marker='.', label='Rótulos')

        axs[1][0].set_title('Erro na classificação')
        axs[1][0].set_ylabel('Diferença')
        axs[1][0].set_xlabel('Amostras')
        axs[1][0].set_xlim(lim)

        axs[1][0].plot([i for i in range(n)], targets - predictions, c=colors[3])
        axs[1][0].get_shared_x_axes().join(axs[0][0], axs[1][0])

        # Matriz de confusão dos resultados.
        cm = confusion_matrix(targets, predictions)

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


def comp_plots(model, dataset, mode, lim, tol=0):
    """Agrega curvas de previsão ou de classificação.

    Args:
        model (tf.keras.Model): Modelo LSTM.
        dataset (tf.data.Dataset): Dataset de teste.
        mode (int): Classificação (0) ou regressão (1).
        lim (list): Limites para plot da curva para melhor visualização.
        tol (int, optional): Largura da janela de tolerância. Defautls to 0.
    """

    if mode:
        comp_prediction(lstm_model, dataset, lim)
    else:
        comp_classification(lstm_model, dataset, lim, tol)


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
        labels (list): Rótulos.
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


def tolerance_plot(labels, width, predictions, lim):
    """Igual predict_plot, mas ilustra a janela de rejeição.

    Args:
        labels (list): Rótulos.
        width (int): Largura da janela de rejeição.
        predictions (list): Predições da rede.
        lim (list): Limites para visualização das curvas.
    """

    tol_left, tol_right = shift_labels(labels, width)
    new_predictions = np.array(tolerance_predictions(labels, width, predictions), dtype=int)

    gs_kw = dict(width_ratios=[3, 1], height_ratios=[2, 1])
    fig, axs = plt.subplots(2, 2, figsize=(12,5), constrained_layout=True, gridspec_kw=gs_kw)

    fig.suptitle(r'Classificação de fases com tolerância de erro de $ \pm $ '+str(width)+' amostras', weight='bold', size='x-large')

    axs[0][0].set_title('Fases classificadas vs. fases reais')
    axs[0][0].set_ylabel('Fases')
    axs[0][0].set_yticks([0, 1, 2, 3])
    axs[0][0].set_yticklabels(['FF', 'HO', 'SW', 'HS'])
    axs[0][0].set_xlim(lim)

    axs[0][0].scatter([i for i in range(len(predictions))], predictions, marker='o', edgecolor='k', label='Predições', c='#2ca02c', s=40)

    axs[0][0].fill_between([i for i in range(len(predictions))], tol_left, tol_right, alpha=0.5, color=colors[1], label='Tolerância')

    axs[0][0].plot(labels, marker='.', label='Rótulos')
    axs[0][0].plot(new_predictions, marker='x', label='Novas predições')

    axs[1][0].set_title('Erro na classificação')
    axs[1][0].set_ylabel('Diferença')
    axs[1][0].set_xlabel('Amostras')
    axs[1][0].set_xlim(lim)

    axs[1][0].plot(labels - new_predictions, c=colors[3])
    axs[1][0].get_shared_x_axes().join(axs[0][0], axs[1][0])

    # Matriz de confusão dos resultados.
    cm = confusion_matrix(labels, new_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axs[0][1])

    axs[0][1].set_title('Matriz de confusão')
    axs[0][1].set_ylabel('Fases reais')
    axs[0][1].set_xlabel('Fases classificadas')

    handles, labels = axs[0][0].get_legend_handles_labels()

    axs[1][1].legend(handles, labels, loc='center')
    axs[1][1].axis('off')


def sens_spec(cm):
    """Retorna valores da sensibilidade e especificidade da detecção de cada fase.

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


#? ------------------
#? Preparo dos dados.
#? ------------------
# Parâmetros para formatação das entradas.
input_feat_size = 25    # 25 pontos passados
output_pred_dist = 1    # prediz o 26o ponto

# Parâmetros para treinamento.
BATCH_SIZE = len(data)
FEAT = [data.columns[i] for i in [1, 2, 3, 4, 5, 6]]
norm_path = 'lstm_model_stats.csv'
class_labels = data.new_stL - 1

test_data, test_label, test_class_label = split_sets(data, FEAT, norm_path, input_feat_size, output_pred_dist, class_labels)


#? --------------------------------
#? Carregamento do modelo treinado.
#? --------------------------------
MODE = 1

if MODE:
    dataset = create_sequences(test_data, test_label, input_feat_size, BATCH_SIZE)
else:
    dataset = create_sequences(test_data, test_class_label, input_feat_size, BATCH_SIZE)


lstm_model = load_model(MODE)


#? ----------------------------------
#? Geração de figura para resultados.
#? ----------------------------------
# limites = [int(len(data)/2) - 500, int(len(data)/2) + 500]
limites = [int(len(data)/2) - 250, int(len(data)/2) + 250]
tol_window = 0

# comp_plots(lstm_model, dataset, MODE, limites, tol_window)
# plt.show()

# for inputs, targets in dataset:
    # predictions = lstm_model.predict(inputs).argmax(axis=1)
    # labels = targets


for inputs, targets in dataset:
    predictions = np.transpose(lstm_model.predict(inputs))
    actuals = np.transpose(targets)
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(data.axL)
# axs[1].plot(targets[:, 3])

# axs[0].set_xlim(limites)
plt.show()


# Performance da regressão.
metrics = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(6):
    # print(explained_variance_score(actuals[i], predictions[i]))
    # print(r2_score(actuals[i], predictions[i]))
    metrics[i][1] = mean_absolute_error(actuals[i], predictions[i])
    metrics[i][2] = mean_squared_error(actuals[i], predictions[i])

for i in range(6):
    print(f'{metrics[i][0]:.4f} & {metrics[i][1]:.4f} & {metrics[i][2]:.4f} \\\\')