#! python3
# fnn_model_test.py - Carrega um conjunto de dados inédito para avaliação do modelo previamente treinado.

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

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


def load_model(model_name):
    """Carrega modelo FNN treinado.

    Args:
        model_name (str): Caminho até modelo.

    Returns:
        tf.keras.Model: Modelo FNN.
    """

    model = keras.models.load_model(model_name)

    return model


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

    data['diff_axL'] = signal_diff(data.loc[:, 'axL'])
    data['diff_ayL'] = signal_diff(data.loc[:, 'ayL'])
    data['diff_azL'] = signal_diff(data.loc[:, 'azL'])

    for filename in os.listdir(path)[1:]:

        temp = pd.read_csv(path + filename, index_col=0)[data_slice[filename]].reset_index(drop=True)

        temp['diff_axL'] = signal_diff(temp.loc[:, 'axL'])
        temp['diff_ayL'] = signal_diff(temp.loc[:, 'ayL'])
        temp['diff_azL'] = signal_diff(temp.loc[:, 'azL'])

        data = pd.concat([data, temp], axis=0, ignore_index=True)

    return data


def signal_diff(data_to_diff):
    """Diferença entre pontos sucessivos no sinal.

    Args:
        data_to_diff (list): Lista com sinais a serem diferenciados.

    Returns:
        list: Lista com diferenças.
    """

    dt = 0.02

    diff = list()
    diff.append(0)

    for i in range(1, len(data_to_diff)):
        temp = (data_to_diff[i] - data_to_diff[i - 1]) / dt
        diff.append(temp)

    return diff


def training_features(type='1111111111111'):
    """Determina quais observações serão usadas como entradas da rede.

    Args:
        type (str, optional): Lista com booleans que correspondem a cada observação. Defaults to '1111111111111'.

    Returns:
        list: Lista com nomes das colunas que serão alimentadas.
    """

    FEATURES = ['gxL', 'gyL', 'gzL', 'axL', 'ayL', 'azL', 'aL', 'angle_x', 'angle_y', 'angle_z', 'diff_axL', 'diff_ayL', 'diff_azL']
    feat_list = list()

    f = list(map(int, list(type)))

    for i in range(len(type)):
        if f[i]:
            feat_list.append(FEATURES[i])
    feat_list.append('new_stL')

    return feat_list


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


def magnitude_a(ax, ay, az):
    """Aceleração resultante.

    Args:
        ax (array_like): Aceleração medida no eixo x.
        ay (array_like): Aceleração medida no eixo y.
        az (array_like): Aceleração medida no eixo z.

    Returns:
        array_like: Lista com aceleração resultante.
    """

    return np.linalg.norm([ax, ay, az])


def accel_angle_z(ax, ay, az):
    """Ângulo definido em torno do eixo z.

    Args:
        ax (array_like): Aceleração medida no eixo x.
        ay (array_like): Aceleração medida no eixo y.
        az (array_like): Aceleração medida no eixo z.

    Returns:
        array_like: Lista contendo os ângulos definidos em função das acelerações, medidos no eixo z.
    """

    return np.arctan2(np.linalg.norm([ax, ay]), az)


def accel_angle_x(ax, ay, az):
    """Ângulo definido em torno do eixo x.

    Args:
        ax (array_like): Aceleração medida no eixo x.
        ay (array_like): Aceleração medida no eixo y.
        az (array_like): Aceleração medida no eixo z.

    Returns:
        array_like: Lista contendo os ângulos definidos em função das acelerações, medidos no eixo x.
    """

    return np.arctan2(np.linalg.norm([ay, az]), ax)


def accel_angle_y(ax, ay, az):
    """Ângulo definido em torno do eixo y.

    Args:
        ax (array_like): Aceleração medida no eixo x.
        ay (array_like): Aceleração medida no eixo y.
        az (array_like): Aceleração medida no eixo z.

    Returns:
        array_like: Lista contendo os ângulos definidos em função das acelerações, medidos no eixo y.
    """

    return np.arctan2(ay, np.linalg.norm([ax, az]))


def angle(accel_angle, gyro_vel, alpha):
    """Calcula o ângulo da inclinação de um eixo pela composição dos ângulos definidos com o giroscópio e acelerômetro.

    Args:
        accel_angle (array_like): Ângulos medidos em relação a um dos eixos.
        gyro_vel (array_like): Velocidade angular medida em torno deste mesmo eixo.
        alpha (float): Proporção complementar entre os ângulos de cada sensor.

    Returns:
        array_like: Inclinação do sensor em torno de um eixo.
    """

    ang = list()
    gyro_ang = 0

    for i in range(len(accel_angle)):
        temp = alpha * gyro_ang + (1 - alpha) * accel_angle[i]
        ang.append(temp)
        gyro_ang = temp + gyro_vel[i] * 0.02

    return ang


def comp_classification(predictions, labels, lim, tol=0):
    """Comparação das classificações feitas pelo modelo e valores reais. Retorna matriz de confusão também.

    Args:
        model (tf.keras.Model): Modelo LSTM.
        dataset (tf.data.Dataset): Dataset avaliado pelo modelo.
        lim (list): Limites para plot da curva para melhor visualização.
    """

    gs_kw = dict(width_ratios=[3, 1], height_ratios=[2, 1])
    fig, axs = plt.subplots(2, 2, figsize=(12,5), constrained_layout=True, gridspec_kw=gs_kw)

    fig.suptitle('Classificação de fases com rede FNN', weight='bold', size='x-large')

    n = len(labels)

    pred = predictions

    if tol:
        pred = tolerance_predictions(labels, tol, predictions.flatten())
        pred = np.array(pred, dtype=int)

    axs[0][0].set_title('Fases classificadas vs. fases reais')
    axs[0][0].set_ylabel('Fases')
    axs[0][0].set_yticks([0, 1, 2, 3])
    axs[0][0].set_yticklabels(['FF', 'HO', 'SW', 'HS'])
    axs[0][0].set_xlim(lim)

    axs[0][0].scatter([i for i in range(n)], pred, marker='o', edgecolor='k', label='Predições', c='#2ca02c', s=40)

    axs[0][0].plot(labels, marker='.', label='Rótulos')

    axs[1][0].set_title('Erro na classificação')
    axs[1][0].set_ylabel('Diferença')
    axs[1][0].set_xlabel('Amostras')
    axs[1][0].set_xlim(lim)

    axs[1][0].plot([i for i in range(n)], labels - pred, c=colors[3])
    axs[1][0].get_shared_x_axes().join(axs[0][0], axs[1][0])

    # Matriz de confusão dos resultados.
    cm = confusion_matrix(labels, pred)

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


#? --------------------------------
#? Cálculo de variáveis adicionais.
#? --------------------------------
# Ângulo por complementary filter -- combinação dos ângulos determinados pelo giroscópio (curto prazo, drift) e acelerômetro (longo prazo, ruído).
tau = 0.14
alpha = tau / (tau + 0.02)

# Ângulo em x; pitch.
data['rho'] = data.apply(lambda x: accel_angle_x(x.axL, x.ayL, x.azL), axis=1)

# Ângulo em y.
data['theta'] = data.apply(lambda x: accel_angle_y(x.axL, x.ayL, x.azL), axis=1)

# Ângulo em z; roll.
data['phi'] = data.apply(lambda x: accel_angle_z(x.axL, x.ayL, x.azL), axis=1)

# Composição do ângulo para cada eixo.
angle_x = angle(data.rho, data.gxL, alpha)
angle_y = angle(data.theta, data.gyL, alpha)
angle_z = angle(data.phi, data.gzL, alpha)

data['angle_x'] = angle_x
data['angle_y'] = angle_y
data['angle_z'] = angle_z

# Aceleração resultante.
data['aL'] = data.apply(lambda x: magnitude_a(x.axL, x.ayL, x.azL), axis=1)


#? ------------------
#? Preparo dos dados.
#? ------------------
train_feat = '1001110111111'
# train_feat = '1111110000000'

feat_list = training_features(type=train_feat)
test_data = data.loc[:, feat_list]
test_target = np.array(test_data.pop(feat_list[-1])) - 1

# Normalização usando dados do conjunto de treino.
norm_par = 'fnn_model_stats.csv'

mu_sigma = pd.read_csv(norm_par, index_col=0)
train_mean = mu_sigma.iloc[0, :]
train_std = mu_sigma.iloc[1, :]

test_data = (test_data - train_mean) / train_std

#? --------------------------------
#? Carregamento do modelo treinado.
#? --------------------------------
model_name = 'fnn_model.h5'
model = load_model(model_name)


# Predições com dados inéditos.
BATCH_SIZE = 512
test_predictions = model.predict(test_data, batch_size=BATCH_SIZE).argmax(axis=1)

test_results = model.evaluate(test_data, test_target, batch_size=BATCH_SIZE, verbose=1)

# print(test_results)
tol_window = 0
limites = [int(len(test_data)/2) - 250, int(len(test_data)/2) + 250]
comp_classification(test_predictions, test_target, limites, tol_window)
plt.show()

# print(classification_report(test_target, test_predictions, target_names=['FF', 'HO', 'SW', 'HS']))