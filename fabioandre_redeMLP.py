from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import set_random_seed
import numpy as np
from pip._vendor.typing_extensions import Self
from tensorflow.python.ops.numpy_ops import np_array_ops



class redeMLP():
    def __init__(self, arquitetura, dadosEntrada,dadosSaida):
        # seta a mesma semente para tornar o código reprodutível
        set_random_seed(0)
        # iniciliza criando separacao dos dados para treino e teste
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(dadosEntrada, dadosSaida, test_size=0.30, random_state=4)
        # normalizacao dos dados de entrada para treino e teste
        scaler = StandardScaler()
        scaler.fit(self.X_treino)
        self.X_treino = scaler.transform(self.X_treino)
        self.X_teste  = scaler.transform(self.X_teste)
        #criação do modelo da rede 
        self.modelo = Sequential()
        self.modelo.add(Dense(input_dim=arquitetura[0], units=arquitetura[1], activation='tanh'))
        self.modelo.add(Dense(units=arquitetura[2], activation='tanh'))
        self.modelo.add(Dense(units=arquitetura[3]))
        otimzador = Adam(learning_rate = 0.016)
        self.modelo.compile(optimizer=otimzador, loss='mse')
        
    def treinar(self):
        #treinamento do modelo
        history = self.modelo.fit(self.X_treino, self.y_treino, epochs=100, batch_size=5, verbose=0)
        predicao  = self.modelo.predict(self.X_teste)
        return r2_score(self.y_teste,predicao)    
    
    def getPesos(self):
        pesos=[]
        i = 0
        for layer in self.modelo.layers:
            print(i)
            i = i +1
            if isinstance(layer, Dense):
                pesos.append(layer.get_weights())
        return pesos
    
    def formatarPesos(self,pesosNovos):
        novosPesos = []
        novosBias = []
        #camada zero
        novosPesos.append(np.array([[pesosNovos[0],pesosNovos[1],pesosNovos[2]],
                                    [pesosNovos[3],pesosNovos[4],pesosNovos[5]]]))
        novosBias.append(np.array([pesosNovos[6],pesosNovos[7],pesosNovos[8]])) 
        #camada 1
        novosPesos.append(np.array([
                                    [pesosNovos[9] ,pesosNovos[10],pesosNovos[11]],
                                    [pesosNovos[12],pesosNovos[13],pesosNovos[14]],
                                    [pesosNovos[15],pesosNovos[16],pesosNovos[17]]
                                    ]))
        novosBias.append(np.array([pesosNovos[18],pesosNovos[19],pesosNovos[20]]))                          
        #camada dois
        novosPesos.append(np.array([
                                    [pesosNovos[21]],
                                    [pesosNovos[22]],
                                    [pesosNovos[23]]
                                    ]))
        novosBias.append(np.array( [pesosNovos[24]] )) 
        return novosPesos, novosBias
    
    def testarPesos(self, formatarPesos):
        #pesosOriginais = []
        camada = 0
        pesos, bias = self.formatarPesos(formatarPesos)
        for layer in self.modelo.layers:
            if isinstance(layer, Dense):
                #pesosOriginais.append(layer.get_weights())
                new_weights = pesos[camada]
                new_biases = bias[camada]
                layer.set_weights([new_weights, new_biases])
                camada = camada + 1
        #print(self.getPesos())        
        predicao  = self.modelo.predict(self.X_teste, verbose=0)
        return r2_score(self.y_teste,predicao)   
        
class redeMLP2():
    def __init__(self, arquitetura, dadosEntrada,dadosSaida):
        # seta a mesma semente para tornar o código reprodutível
        set_random_seed(0)
        # iniciliza criando separacao dos dados para treino e teste
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(dadosEntrada, dadosSaida, test_size=0.30, random_state=4)
        # normalizacao dos dados de entrada para treino e teste
        scaler = StandardScaler()
        scaler.fit(self.X_treino)
        self.X_treino = scaler.transform(self.X_treino)
        self.X_teste  = scaler.transform(self.X_teste)
        #criação do modelo da rede 
        self.modelo = Sequential()
        self.modelo.add(Dense(input_dim=arquitetura[0], units=arquitetura[1], activation='tanh'))
        self.modelo.add(Dense(units=arquitetura[2], activation='tanh'))        
        self.modelo.add(Dense(units=arquitetura[3]))
        otimzador = Adam(learning_rate = 0.016)
        self.modelo.compile(optimizer=otimzador, loss='mse')
        
    def treinar(self):
        #treinamento do modelo
        history = self.modelo.fit(self.X_treino, self.y_treino, epochs=100, batch_size=5, verbose=0)
        predicao  = self.modelo.predict(self.X_teste)
        return r2_score(self.y_teste,predicao)    
    
    def getPesos(self):
        pesos=[]
        i = 0
        for layer in self.modelo.layers:
            print(i)
            i = i +1
            if isinstance(layer, Dense):
                pesos.append(layer.get_weights())
        return pesos
    
    def formatarPesos(self,pesosNovos):
        novosPesos = []
        novosBias = []
        #camada zero
        novosPesos.append(np.array([[pesosNovos[0], pesosNovos[1], pesosNovos[2], pesosNovos[3], pesosNovos[4]],
                                    [pesosNovos[5], pesosNovos[6], pesosNovos[7], pesosNovos[8], pesosNovos[9]]]))    
        novosBias.append(np.array([pesosNovos[10],pesosNovos[11],pesosNovos[12],pesosNovos[13],pesosNovos[14]])) 
        #camada 1
        novosPesos.append(np.array([[pesosNovos[15],pesosNovos[16],pesosNovos[17],pesosNovos[18],pesosNovos[19]],
                                    [pesosNovos[20],pesosNovos[21],pesosNovos[22],pesosNovos[23],pesosNovos[24]],
                                    [pesosNovos[25],pesosNovos[26],pesosNovos[27],pesosNovos[28],pesosNovos[29]],
                                    [pesosNovos[30],pesosNovos[31],pesosNovos[32],pesosNovos[33],pesosNovos[34]],
                                    [pesosNovos[35],pesosNovos[36],pesosNovos[37],pesosNovos[38],pesosNovos[39]]]))    
        novosBias.append(np.array([pesosNovos[40],pesosNovos[41],pesosNovos[42],pesosNovos[43],pesosNovos[44]]))       
        #camada de saída    
        novosPesos.append(np.array([[pesosNovos[45]],
                                    [pesosNovos[46]],
                                    [pesosNovos[47]],
                                    [pesosNovos[48]],
                                    [pesosNovos[49]]
                                    ]))
        novosBias.append(np.array( [pesosNovos[50]] )) 
        return novosPesos, novosBias

    
    def testarPesos(self, formatarPesos):
        #pesosOriginais = []
        camada = 0
        pesos, bias = self.formatarPesos(formatarPesos)
        for layer in self.modelo.layers:
            if isinstance(layer, Dense):
                #pesosOriginais.append(layer.get_weights())
                new_weights = pesos[camada]
                new_biases = bias[camada]
                layer.set_weights([new_weights, new_biases])
                camada = camada + 1
        #print(self.getPesos())        
        predicao  = self.modelo.predict(self.X_teste, verbose=0)
        return r2_score(self.y_teste,predicao)           