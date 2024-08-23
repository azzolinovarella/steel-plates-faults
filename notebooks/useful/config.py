# Bibliotecas padrão
import warnings
import os

# Bibliotecas utilitárias de terceiros
import numpy as np
import pandas as pd

# Importações de constantes personalizadas
from useful.constants import SEED


def set_default_configs():
    # Para remover warnings desnecessários do scikit-learn que só poluem o notebook
    warnings.simplefilter('ignore'); os.environ['PYTHONWARNINGS'] = 'ignore'  

    # Para permitir ver todas as colunas do pandas
    pd.set_option('display.max_columns', None)

    # Para garantir reprodutibilidade nos experimentos
    np.random.seed(SEED)