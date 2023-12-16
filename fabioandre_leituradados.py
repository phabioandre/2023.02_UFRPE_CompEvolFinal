import pandas as pd


##################################################################################################
# Tratamento e leitura dos dados para uso na Rede Neural Artificial
##################################################################################################
def leituraDeDados(n_dias):
    dataFramePandas = pd.read_csv("dados.csv", delimiter=";")
    dados = dataFramePandas[:]
    dados = dados[['Timestamp','EXO-pH','EXO-TempC','EXO-DOmgL','EXO-DOPerSat']]
    dados=dados.drop(0)
    dados['Timestamp']=pd.to_datetime(dados['Timestamp'], format="%d/%m/%Y %H:%M")
    dados['EXO-pH']=pd.to_numeric(dados['EXO-pH'])
    dados['EXO-TempC']=pd.to_numeric(dados['EXO-TempC'])
    dados['EXO-DOmgL']=pd.to_numeric(dados['EXO-DOmgL'])
    dados['EXO-DOPerSat']=pd.to_numeric(dados['EXO-DOPerSat'])
    dados=dados.dropna(axis=0)
    saidas = dados[['EXO-DOmgL']]
    entradas = dados[['EXO-pH','EXO-TempC']]
    
    entradas = entradas[0:n_dias*144]
    saidas   = saidas[0:n_dias*144]
    
#    plt.plot(entradas[0:len(entradas.index)])
#    plt.plot(saidas[0:len(entradas.index)])   
    # COnverte a hora do dia em minutos decorridos desde a meia noite
   # entradas['Timestamp'] = entradas['Timestamp'].dt.hour * 60 + entradas['Timestamp'].dt.minute

    return entradas, saidas