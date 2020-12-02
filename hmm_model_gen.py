#! python3
# hmm_model_gen.py - Gerador de modelo HMM para classificação de eventos da caminhada humana saudável.

import pomegranate as pg
from pomegranate import distributions
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


def create_sets(data, n_obs_per_seq):
    """Divide os dados dos sensores em grupos de treinamento e validação, adequando-os ao formato requerido para o ajuste dos parâmetros.

    Args:
        data (pd.DataFrame): Data frame com dados originais.

    Returns:
        np.ndarray: Grupos de treinamento e validação.
    """

    # Remoção de valores finais para adequação no formato requerido.
    n_rows_drop = data.shape[0] % n_obs_per_seq
    data = data[:-n_rows_drop]

    # Vetor de rótulos.
    label = np.array(data.loc[:, 'new_stL'] - 1, dtype=str)

    # Sequências de observações (atributos).
    observations = np.array(data.loc[:, ['gxL', 'gyL', 'gzL', 'axL', 'ayL', 'azL']])

    # Reshape para formato: (#sequências, #observações) e (#sequências, #observações, #features).
    observations = observations.reshape((-1, n_obs_per_seq, 6))
    label = label.reshape((-1, n_obs_per_seq))

    # start_state = np.array(['None-start' for i in range(label.shape[0])])
    # teste = np.column_stack([start_state, label])

    # 80% para treino, 20% para validação.
    n = len(observations)
    split = 0.8
    obs_train, obs_test = observations[0:int(split*n), :], observations[int(split*n):, :]

    # Normalizar nos dados de treino.
    mean = obs_train.mean(axis=0).mean(axis=0)
    std = obs_train.std(axis=0).std(axis=0)

    obs_train = (obs_train - mean) / std
    obs_test = (obs_test - mean) / std

    # Exportar para teste em nova sequência.
    stat = pd.concat([pd.Series(mean), pd.Series(std)], axis=1, ignore_index=True).T
    stat.to_csv('hmm_model_stats.csv', sep=',')

    label_train, label_test = label[0:int(split*n), :], label[int(split*n):, :]

    # Labels numéricas para plot da curva.
    num_label = np.array(label_test.flatten(), dtype=np.int32)

    return obs_train, obs_test, label_train, label_test, num_label


def load_init_model(load=False):
    """Carrega modelo treinado ou inicializa um novo.

    Args:
        load (bool, optional): Se True, carrega modelo .json treinado. Defaults to False.

    Returns:
        hmm.HiddenMarkovModel: HMM.
    """

    if load:
        model = hmm.HiddenMarkovModel().from_json('hmm_model.json')
    else:
        model = model_init()

    return model


def model_init():
    """Inicializa a estrutura do modelo, zerando transições indesejadas.

    Returns:
        hmm.HiddenMarkovModel: HMM.
    """

    # Distribuições de probabilidade de emissão das observações.
    means = np.zeros((6, 1))
    cov = np.diag(np.ones(6))

    d0 = distributions.MultivariateGaussianDistribution(means, cov)
    d1 = distributions.MultivariateGaussianDistribution(means, cov)
    d2 = distributions.MultivariateGaussianDistribution(means, cov)
    d3 = distributions.MultivariateGaussianDistribution(means, cov)

    # Estados ocultos relacionados a cada emissão.
    s0 = pg.State(d0, name='0')
    s1 = pg.State(d1, name='1')
    s2 = pg.State(d2, name='2')
    s3 = pg.State(d3, name='3')

    model = hmm.HiddenMarkovModel()
    model.add_states([s0, s1, s2, s3])

    # Distribuições iniciais.
    model.add_transition(model.start, s0, 0.25)
    model.add_transition(model.start, s1, 0.25)
    model.add_transition(model.start, s2, 0.25)
    model.add_transition(model.start, s3, 0.25)

    # Probabilidades de transição.
    model.add_transition(s0, s0, 0.8)
    model.add_transition(s0, s1, 0.2)

    model.add_transition(s1, s1, 0.8)
    model.add_transition(s1, s2, 0.2)

    model.add_transition(s2, s2, 0.8)
    model.add_transition(s2, s3, 0.2)

    model.add_transition(s3, s3, 0.8)
    model.add_transition(s3, s0, 0.2)

    model.bake(verbose=True)

    return model


def fit_and_save(model, observations, label, save=True):
    """Ajuste dos parâmetros do modelo, salvo em arquivo .json.

    Args:
        model (hmm.HiddenMarkovModel): HMM.
        observations (np.ndarray): Sequência de observações.
        label (np.ndarray): Sequência dos estados ocultos correspondentes.
        save (bool, optional): Se True, salva o modelo. Defaults to True.
    """

    model, history = model.fit(
        sequences=observations,
        labels=label,
        max_iterations=500,
        algorithm='labeled',
        verbose=True,
        distribution_inertia=0.2,
        edge_inertia=0.2,
        # use_pseudocount=True,
        batches_per_epoch=512,
        return_history=True
    )

    # Arquivo .json com especificações do modelo.
    if save:
        with open('hmm_model.json', 'w') as f:
            f.write(model.to_json())

    return history


def predict_plot(model, observ, labels):
    """Comparação entre os estados mais prováveis e os reais.

    Args:
        model (hmm.HiddenMarkovModel): HMM.
        observ (list): Sequência de observações.
        labels (list): Sequência dos estados ocultos.
    """

    preds = list()

    # Predição pelo algoritmo de Viterbi.
    for i in observ:
        p = model.predict(i, algorithm='viterbi')[1:]
        preds.append(p)

    preds = np.array(preds, dtype=np.int32)
    states = preds.flatten()

    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    fig.suptitle('Identificação de fases com HMM', weight='bold', size='x-large')

    axs[0].set_title('Fases classificadas vs. fases reais')
    axs[0].set_ylabel('Fases')
    axs[0].set_yticks([0, 1, 2, 3])
    axs[0].set_yticklabels(['FF', 'HO', 'SW', 'HS'])

    axs[0].scatter([i for i in range(len(states))], states, marker='o', edgecolor='k', label='Predições', c='#2ca02c', s=40)

    axs[0].plot(labels, marker='.', label='Rótulos')

    axs[1].set_title('Erro na classificação')
    axs[1].set_ylabel('Diferença')
    axs[1].set_xlabel('Amostras')

    axs[1].plot(labels - states)

    fig.legend(loc=7)
    plt.subplots_adjust(right=0.88)

    # Matriz de confusão dos resultados.
    plt.figure()
    cm = confusion_matrix(labels, states)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.ylabel('Fases reais')
    plt.xlabel('Fases classificadas')


def logprob_gain(history):
    """Curva da log-probabilidade relativa entre modelos sucessivos.

    Args:
        history (dict): Dicionário contendo evolução do treinamento por época.
    """

    epochs = history.epochs
    log_improv = history.improvements

    total_improv = np.cumsum(log_improv)

    fig, axs = plt.subplots(1, 1, figsize=(12, 5))

    fig.suptitle('Log-probabilidades das observações terem sido geradas pelo modelo', weight='bold')

    axs.set_xlabel('Épocas')
    axs.set_ylabel(r'$ \frac{logP_{t}}{logP_{t-1}} $', size='x-large')

    axs.plot(epochs, total_improv, marker='.')


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


#? ----------------------
#? Divisão dos conjuntos.
#? ----------------------
obs_train, obs_test, label_train, label_test, num_labels = create_sets(data, 25)


#? ------------
#? Treinamento.
#? ------------
# Cria modelo HMM.
model = load_init_model()

history = fit_and_save(model, np.array(list(obs_train)), np.array(list(label_train)), save=True)

# print(model.dense_transition_matrix())
# print(model.states[0].distribution)

# Comparação dos resultados.
predict_plot(model, np.array(list(obs_test)), num_labels)
# logprob_gain(history)
plt.show()