#! python3
# fnn_model_gen.py - Gerador de rede neural feedforward para classificação de fases do caminhar nos sinais inerciais obtidos em pacientes saudáveis.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import metrics, optimizers, regularizers

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


def create_fnn(n_input, lr=0.0001):
    """Construção da FNN. Otimização da função custo entropia cruzada com algoritmo Adam.

    Args:
        n_input (int): Número de neurônios de entrada.
        lr (float, optional): Taxa de aprendizado. Defaults to 0.001.

    Returns:
        tf.keras.Model: Modelo FNN.
    """

    kinematic_in = keras.Input(shape=(n_input,), name='vel_accel')

    x = layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(kinematic_in)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.Dropout(0.2)(x)

    phase_out = layers.Dense(4, activation='softmax', name='phase_prob')(x)

    gait_classifier = keras.Model(kinematic_in, phase_out, name='gait_classifier')


    metrics = [keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseCategoricalCrossentropy()]

    gait_classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics
    )

    print(gait_classifier.summary())

    return gait_classifier


def fit_to_data(model, data, label, batch_size, epochs, val_data, val_label):
    """Ajuste dos dados para determinação dos parâmetros da rede.

    Args:
        model (tf.keras.Model): Modelo FNN a ser ajustado.
        data (pd.DataFrame): Data frame com as features.
        label (pd.DataFrame): Vetor com labels.
        batch_size (int): Tamanho do mini-lote para apresentação dos exemplos.
        epochs (int): Número máximo de épocas de treinamento.
        val_data (pd.DataFrame): Exemplos de validação, não apresentados no treinamento.
        val_label (pd.DataFrame): Rótulos para validação.

    Returns:
        tf.keras.callbacks.History: Histórico de treinamento.
    """

    # Callbacks do treinamento.
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_crossentropy',
        verbose=1,
        patience=15,
        mode='min',
        restore_best_weights=True
    )

    # Pesos de cada classe proporcionais ao inverso de sua parcela no total de exemplos.
    class_weights = {
        0: 5.326,
        1: 5.571,
        2: 2.095,
        3: 6.413
    }

    history = model.fit(
        data,
        label,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[callback],
        validation_data=(val_data, val_label),
        class_weight=class_weights
    )

    return history


def load_data(path, test=False):
    """Carrega os dados e os coloca em um pd.DataFrame. São agregados todas as séries de medição.

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


def save_model(model, model_name):
    """Salva parâmetros do modelo em arquivo .h5.

    Args:
        model (tf.keras.Model): Modelo FNN.
        model_name (str): Nome do modelo .h5.
    """

    model.save(model_name)


def comp_classification(predictions, labels, lim):
    """Curva comparativa da classificação de fases.

    Args:
        predictions (array_like): Predições do modelo.
        labels (array_like): Fases reais.
    """

    gs_kw = dict(width_ratios=[3, 1], height_ratios=[2, 1])
    fig, axs = plt.subplots(2, 2, figsize=(12,5), constrained_layout=True, gridspec_kw=gs_kw)

    fig.suptitle('Classificação de fases com FNN', weight='bold', size='x-large')

    axs[0][0].set_title('Fases classificadas vs. fases reais')
    axs[0][0].set_ylabel('Fases')
    axs[0][0].set_yticks([0, 1, 2, 3])
    axs[0][0].set_yticklabels(['FF', 'HO', 'SW', 'HS'])
    axs[0][0].set_xlim(lim)

    axs[0][0].scatter([i for i in range(len(predictions))], predictions, marker='o', edgecolor='k', label='Predições', c='#2ca02c', s=40)

    axs[0][0].plot(labels, marker='.', label='Rótulos')

    axs[1][0].set_title('Erro na classificação')
    axs[1][0].set_ylabel('Diferença')
    axs[1][0].set_xlabel('Amostras')
    axs[1][0].set_xlim(lim)

    axs[1][0].plot(labels - predictions, c=colors[3])

    # Matriz de confusão dos resultados.
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axs[0][1])

    axs[0][1].set_title('Matriz de confusão')
    axs[0][1].set_ylabel('Fases reais')
    axs[0][1].set_xlabel('Fases classificadas')

    handles, labels = axs[0][0].get_legend_handles_labels()

    axs[1][1].legend(handles, labels, loc='center')
    axs[1][1].axis('off')


def visualize_loss(history):
    """Plota curva da evolução do custo com base no histórico de treinamento.

    Args:
        history (tf.keras.callbacks.History): Histórico de treinamento.
    """

    fig, axs = plt.subplots(1, 1, figsize=(12, 5))

    loss_name = 'Entropia cruzada'
    loss = history.history['sparse_categorical_crossentropy']
    val_loss = history.history['val_sparse_categorical_crossentropy']


    epochs = range(len(loss))
    axs.plot(epochs, loss, label='Treino')
    axs.plot(epochs, val_loss, label='Validação')

    fig.suptitle(loss_name + ' durante treinamento e validação', weight='bold', size='x-large')
    axs.set_xlabel('Épocas')
    axs.set_ylabel('Custo')
    fig.legend(loc=7)
    plt.subplots_adjust(right=0.88)


#? -----------------------
#? Carregamento dos dados.
#? -----------------------
data_path = '../walking/data2/'
data = load_data(data_path)


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

data['aL'] = data.apply(lambda x: magnitude_a(x.axL, x.ayL, x.azL), axis=1)


#? -------------------------------------------------------------------
#? Divisão em conjuntos de treino (80%), validação (15%) e teste (5%).
#? -------------------------------------------------------------------
# gx gy gz ax ay az |a| ang_x ang_y ang_z d(ax) d(ay) d(az)
train_feat = '1001110111111'
# train_feat = '1111110000000'

n = len(data)
feat_list = training_features(type=train_feat)

# Treino.
train_data = data.loc[0:int(.8*n), feat_list]
train_target = np.array(train_data.pop(feat_list[-1])) - 1

# Validação.
val_data = data.loc[int(.8*n):int(.95*n), feat_list]
val_target = np.array(val_data.pop(feat_list[-1])) - 1

# Teste.
test_data = data.loc[int(.95*n):, feat_list]
test_target = np.array(test_data.pop(feat_list[-1])) - 1


#? ------------------------------------------------
#? Normalização usando dados do conjunto de treino.
#? ------------------------------------------------
train_mean = train_data.mean()
train_std = train_data.std()

# Exportar média e desvio padrão para utilizar no teste com novos dados
stat = pd.concat([train_mean, train_std], axis=1, ignore_index=True).T
stat.to_csv('fnn_model_stats.csv', sep=',')

# Normalização com métricas do conjunto de treino.
train_data = (train_data - train_mean) / train_std
val_data = (val_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std


#? ------------
#? Treinamento.
#? ------------
MAX_EPOCHS = 500
BATCH_SIZE = 512

model = create_fnn(n_input=len(feat_list)-1)

history = fit_to_data(model, train_data, train_target, BATCH_SIZE, MAX_EPOCHS, val_data, val_target)

save_model(model, 'fnn_model.h5')


# Predições e avaliação do modelo.
predictions = model.predict(test_data, batch_size=BATCH_SIZE).argmax(axis=1)
results = model.evaluate(test_data, test_target, batch_size=BATCH_SIZE, verbose=1)
print(results)
print(classification_report(test_target, predictions, target_names=['FF', 'HO', 'SW', 'HS']))


# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)

limites = [int(len(test_data)/2) - 250, int(len(test_data)/2) + 250]

comp_classification(predictions, test_target, limites)
visualize_loss(history)
plt.show()
