import numpy as np
import math

class selfAttention:
    def __init__(self, query, key, value) -> None:
        self.query = query
        self.key = key
        self.value = value

    def calculateSelfA(self, masking=True):

        d_k = self.query.shape[-1]
        ## Matriz de Atencion
        matrix_attention = np.dot(self.query, self.key.T)
        # Ajuste/escalamiento para reducir los valores de probabilidad
        scaling = matrix_attention / math.sqrt(d_k)

        if masking:
            ## Masking elimina el contexto de palabras futuras 
            size = self.query.shape[0]
            mask = np.tril(np.ones( (size, size )))
            mask[mask == 0] = -np.infty
            mask[mask == 1] = 0
            scaling += mask
        
        ## Normalizacion con softmax
        attention = self.softmax(scaling)
        #obteniendo los nuevos valores de las palabras
        new_value = np.dot(matrix_attention, self.value)

        return attention, new_value
    
    def softmax(self, x):
        return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T







