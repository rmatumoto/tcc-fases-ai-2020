#! python3
# lstm_model_gen.py - Gerador de modelo LSTM para regressão e classificação.

import tensorflow as tf
from tensorflow import keras

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


def create_sets(data, train_features, split, normalize=True):
    """Divide dados em conjuntos de treinamento e validação.

    Args:
        data (pd.DataFrame): Data frame com dados.
        train_features (list): Lista das colunas que contém as features desejadas.
        split (float): Percentual usado para treinamento.
        normalize (boolean, optional): Normaliza o conjunto de dados. Defaults to True.

    Returns:
        pd.DataFrame: Data frames contendo todos os dados, somentes os de treino e os de validação.
    """

    train_split = int(split * int(data.shape[0]))

    features = data[train_features]

    train_set = features.loc[0: train_split - 1]
    val_set = features.loc[train_split:]

    if normalize:
        mean = train_set.mean()
        std = train_set.std()

        stat = pd.concat([mean, std], axis=1, ignore_index=True).T
        stat.to_csv('lstm_model_stats.csv', sep=',')

        features = (features - mean) / std
        train_set = (train_set - mean) / std
        val_set = (val_set - mean) / std

    return features, train_set, val_set


def split_sets(data, train_features, input_size, output_size, split, class_labels):
    """Divide dados em conjuntos de treinamento e validação.

    Args:
        data (pd.DataFrame): Data frame com dados.
        train_features (list): Lista com features que serão usadas.
        input_size (int): Quantidade de amostras que será utilizada para previsão.
        output_size (int): Qual ponto a partir do histórico será previsto.
        split (float): Proporção dos dados usados em treinamento.

    Returns:
        array_like: Conjuntos de treinamento e validação, ordenados para serem alimentados em sequência.
    """

    train_split = int(split * int(data.shape[0]))

    # Grupos de treino e validação.
    feat, train, val = create_sets(data, train_features, split)

    # Índices para corte dos dados.
    # início dos rótulos de treino.
    start_train_idx = input_size + output_size
    end_train_idx = start_train_idx + train_split
    # rótulos de validação.
    start_val_idx = train_split + input_size + output_size
    end_val_idx = len(val) - input_size - output_size

    # Conjuntos de treinamento e validação, prontos para serem agrupados em lotes.
    train_data = train.values
    train_label_pred = feat.iloc[start_train_idx: end_train_idx]
    train_label_class = np.array(class_labels.iloc[start_train_idx: end_train_idx], dtype=int)

    val_data = val.iloc[:end_val_idx].values
    val_label_pred = feat.iloc[start_val_idx:]
    val_label_class = np.array(class_labels.iloc[start_val_idx:], dtype=int)


    return train_data, train_label_pred, val_data, val_label_pred, train_label_class, val_label_class


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


def create_lstm(train_data, n_units, mode):
    """Cria a LSTM para previsão de valores futuros.

    Args:
        train_data (tf.data.Dataset): Dataset com features e labels.
        n_units (int): Quantidade de células LSTM.

    Returns:
        tf.keras.Model: Modelo LSTM.
    """

    for batch in train_data.take(1):
        inputs, _ = batch

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(n_units)(inputs)

    if mode:
        outputs = keras.layers.Dense(6)(lstm_out)
    else:
        outputs = keras.layers.Dense(4, activation='softmax')(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    return model


def compile_fit(model, lr, train_data, val_data, epochs, batch_size, mode):
    """Ajusta dados ao modelo. Função custo erro quadrático médio, otimizada com Adam.

    Args:
        model (tf.keras.Model): Modelo LSTM.
        lr (float): Taxa de aprendizado.
        train_data (tf.data.Dataset): Dataset para treino.
        val_data (tf.data.Dataset): Dataset para validação.
        epochs (int): Número máximo de épocas de treinamento.

    Returns:
        tf.keras.callbacks.History: Histórico de treinamento.
    """

    if mode:
        # regressão
        es_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='mse'
        )

    else:
        # classificação
        es_callback = keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_crossentropy',
        patience=10,
        mode='min',
        restore_best_weights=True
    )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=keras.metrics.SparseCategoricalCrossentropy()
        )

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        batch_size=batch_size,
        callbacks=[es_callback]
    )

    return history


def visualize_loss(history, mode):
    """Plota curva da evolução do custo com base no histórico de treinamento.

    Args:
        history (tf.keras.callbacks.History): Histórico de treinamento.
        mode (int): 0 - classificação, 1 - regressão.
    """

    fig, axs = plt.subplots(1, 1, figsize=(12, 5))

    if mode:
        loss_name = 'Erro quadrático médio'
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    else:
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


def comp_prediction(model, dataset):
    """Comparação das previsões feitas pelo modelo e valores reais.

    Args:
        model (tf.keras.Model): Modelo LSTM.
        dataset (tf.data.Dataset): Dataset avaliado pelo modelo.
    """

    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12,5))

    lbls = [r'$ \omega_x $', r'$ \omega_y $', r'$ \omega_z $', r'$ a_x $', r'$ a_y $', r'$ a_z $']

    for i in range(6):
        for x, y in dataset.take(1):
            axs[i//3, i%3].plot(np.transpose(y)[i])
            axs[i//3, i%3].plot(np.transpose(lstm_model.predict(x))[i])
            axs[i//3, i%3].set_ylabel(lbls[i])

    fig.suptitle('Previsão dos sinais inerciais vs. rótulos')
    fig.legend(['Rótulos', 'Predições'], loc=7)


def comp_classification(model, dataset):
    """Comparação das fases classificadas pelo modelo e as reais.

    Args:
        model (tf.keras.Model): Modelo LSTM.
        dataset (tf.data.Dataset): Dataset avaliado pelo modelo.
    """

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,5))

    fig.suptitle('Identificação de fases com rede LSTM', weight='bold', size='x-large')

    for inputs, targets in dataset.take(1):
        n = len(targets)
        axs[0].plot(targets, marker='.', label='Rótulos')
        axs[0].scatter([i for i in range(n)], model.predict(inputs).argmax(axis=1), label='Predições', marker='o', edgecolor='k', c='#2ca02c', s=40)
        axs[0].set_ylabel('Fases')
        axs[0].set_yticks([0, 1, 2, 3])
        axs[0].set_yticklabels(['FF', 'HO', 'SW', 'HS'])
        axs[0].set_title('Fases classificadas vs. fases reais')

        axs[1].plot([i for i in range(n)], targets - model.predict(inputs).argmax(axis=1))
        axs[1].set_ylabel('Diferença')
        axs[1].set_title('Erro na classificação')
        axs[1].set_xlabel('Amostras')

    fig.legend(loc=7)
    plt.subplots_adjust(left=0.10, right=0.88)


#? -----------------------
#? Carregamento dos dados.
#? -----------------------
data_path = '../walking/data2/'
data = load_data(data_path, test=False)


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
# Conjuntos de treinamento e validação.
split_fraction = 0.8
FEAT = [data.columns[i] for i in [1, 2, 3, 4, 5, 6]]

# Parâmetros para formatação das entradas.
input_feat_size = 25    # 25 pontos passados
output_pred_dist = 1    # prediz/classifica o 26o ponto

# Parâmetros para treinamento.
BATCH_SIZE = 512
LEARNING_RATE = 0.001
MAX_EPOCHS = 500

# Formatação em sequências.
sequence_length = input_feat_size
classif_labels = data.new_stL - 1

# Formato: (batch_size, #timesteps, #features)
x_train, y_train, x_val, y_val, label_train, label_val = split_sets(data, FEAT, input_feat_size, output_pred_dist, split_fraction, classif_labels)


#? ------------
#? Treinamento.
#? ------------
# 1: previsão, 0: classificação.
MODE = 1

# Dados de treino e validação.
if MODE:
    dataset_train = create_sequences(x_train, y_train, sequence_length, BATCH_SIZE)
    dataset_val = create_sequences(x_val, y_val, sequence_length, BATCH_SIZE)
    name = '_pred.h5'
else:
    dataset_train = create_sequences(x_train, label_train, sequence_length, BATCH_SIZE)
    dataset_val = create_sequences(x_val, label_val, sequence_length, BATCH_SIZE)
    name = '_class.h5'

# Modelo
lstm_model = create_lstm(dataset_train, 32, MODE)

history = compile_fit(lstm_model, LEARNING_RATE, dataset_train, dataset_val, MAX_EPOCHS, BATCH_SIZE, MODE)

lstm_model.save('lstm_model' + name)

visualize_loss(history, MODE)

if MODE:
    comp_prediction(lstm_model, dataset_val)
else:
    comp_classification(lstm_model, dataset_val)

plt.show()