import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


class LowPass():
    """Filtro Butterworth passa-baixa.
    """

    def __init__(self, cutoff, fs, order):
        """Inicializa filtro.

        Args:
            cutoff (float): Frequência de corte, Hz.
            fs (float): Frequência de amostragem, Hz.
            order (int): Ordem para aproximação.
        """

        self.cutoff = cutoff
        self.fs = fs
        self.order = order


    def get_tf(self):
        """Determina coeficientes do numerador e denominador da função transferência do filtro. Resposta é normalizada na frequência de Nyquist [0, 1]

        Returns:
            array_like: Vetor com coeficientes do numerador e denominador da função transferência do filtro.
        """

        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq

        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)

        return b, a


    def freq_response(self, b, a):
        """Resposta em frequência do filtro.

        Args:
            b (array_like): Coeficientes do numerador da FT.
            a (array_like): Coeficientes do denominador da FT.

        Returns:
            array_like: Resposta em frequência do filtro digital.
        """

        w, h = freqz(b, a, worN=8000)  # worN = número de pontos de frequência.

        return w, h


    def get_response(self, b, a, data):
        """Aplica filtro no dado e retorna sua resposta.

        Args:
            b (array_like): Coeficientes do numerador da FT.
            a (array_like): Coeficientes do denominador da FT.
            data (pd.DataFrame): Matrix com os dados a serem filtrados.

        Returns:
            pd.DataFrame: Matrix de mesmo tamanho da de origem com dados filtrados.
        """

        y = data[:]
        for col in data:
            y[col] = lfilter(b, a, data[col])

        return y