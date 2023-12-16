import fabioandre_leituradados
import fabioandre_redeMLP
import tensorflow as tf
from deap import  base, creator, tools, algorithms
import random
import numpy as np
from datetime import date, datetime 
import pandas as pd

def avaliaFitness(individuo):
    rQuadrado = redeTestada.testarPesos(individuo)
    return rQuadrado,

##################################################################################
# 1 - Carregamento da base de dados
##################################################################################
nDiasAvaliados = 30
dadosEntrada, dadosSaida = fabioandre_leituradados.leituraDeDados(nDiasAvaliados)

##################################################################################
# 2 - Carregamento da Rede Neural
##################################################################################
arquiteturaRedeMLP = [2,5,5,1]
nGenes = 51 # escrever código para calcular automaticante com a arquitetura passada
#nGenes = 25 # escrever código para calcular automaticante com a arquitetura passada
redeTestada = fabioandre_redeMLP.redeMLP2(arquiteturaRedeMLP,dadosEntrada,dadosSaida)
rQuadrado = redeTestada.treinar()
print('valor de r² = ', rQuadrado)

##################################################################################
# 3 - Carregamento do Algorítmo Evolutivo -> Estratégia de Evolução (mi, lambda)
# representação: arranjo com pesos e bias de todas as camadas e neurônios da rede (reais)
# avaliação: r quadrado nos dados de testes da rede neural MLP
# mutação: perturbação gausiana
# seleção : melhores fitness
# pressão mi/lambda = 1/5
##################################################################################
MI  = 5
LAMBDA = 25
QTDAVALIACOES = 40000
PASSOEVOLUCAO = 0.2
mutpb_usado = 0.4
indpb_usado = 0.2
cxpb_usado = 0.2
limites_usado = 6

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("atrIndividuo", random.uniform,-limites_usado,limites_usado)                    # o atributo do indivíduo será um numero aleatório no intervalor determinado
toolbox.register("individual", tools.initRepeat,creator.Individual,toolbox.atrIndividuo,nGenes)  # Registra indivíduo e sua função de criação de atributos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)                       # Criação da população de indivíduos
toolbox.register("evaluate", avaliaFitness)                                                      # Registra função objetivo para avaliar aptdião
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma= PASSOEVOLUCAO, indpb=indpb_usado)
toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("select", tools.selRoulette)
#selecao = 'tools.selRoulette'
toolbox.register("select", tools.selTournament, tournsize=3)
selecao = 'tools.selTournament, tournsize=3'
#toolbox.register("select", tools.selBest)
#selecao = 'tools.selBest'
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
pop = toolbox.population(n=MI)
horaInicio = datetime.now()
print('hora de início ...> ',horaInicio)

'''
populacao, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MI, lambda_=LAMBDA, 
        cxpb=cxpb_usado, mutpb=mutpb_usado, ngen=QTDAVALIACOES, stats=stats, halloffame=hof)
algoritmo = 'eaMuCommaLambda'        
'''
populacao, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb_usado, mutpb=mutpb_usado, ngen=QTDAVALIACOES, stats=stats, halloffame=hof)
algoritmo = 'eaSimple'

#geracoes = logbook.select("gen")
#fit_maximos = logbook.chapters["fitness"].select("max")
#plt.plot(geracoes, fit_maximos, "b-", label="Melhores fitness")
horaFinal = datetime.now()
date_time = horaFinal.strftime("%Y%m%d_%H%M%S")
df_log = pd.DataFrame(logbook)
nome=f'C:\\Users\\phabi\\OneDrive\\Cursos\\PPGIA\\PPGIA7303 - COMPUTAÇÃO EVOLUTIVA\\Final\\Resultados{date_time}'
df_log.to_csv(f'{nome}.csv', index=False) # Writing to a CSV file

with open(f'{nome}.txt', "w") as arquivo:
    arquivo.write(f"Hora inicial       = {horaInicio}\n")
    arquivo.write(f"Hora final         = {horaFinal}\n")
    arquivo.write(f"nDiasAvaliados     = {nDiasAvaliados}\n")
    arquivo.write(f"arquiteturaRedeMLP = {arquiteturaRedeMLP}\n")
    arquivo.write(f"algoritmo          = {algoritmo}\n")  
    arquivo.write(f"selecao            = {selecao}\n") 
    arquivo.write(f"MI                 = {MI}\n")
    arquivo.write(f"LAMBDA             = {LAMBDA}\n")
    arquivo.write(f"QTDAVALIACOES      = {QTDAVALIACOES}\n")
    arquivo.write(f"PASSOEVOLUCAO      = {PASSOEVOLUCAO}\n")             
    arquivo.write(f"cxpb               = {cxpb_usado}\n")  
    arquivo.write(f"mutpb              = {mutpb_usado}\n") 
    arquivo.write(f"indpb              = {indpb_usado}\n")     
    arquivo.write(f"rQuadrado          = {rQuadrado}\n") 
    arquivo.write(f"limites_usado      = {limites_usado}\n")    
    arquivo.write(f"melhor             = {hof[0].fitness}\n")    
    arquivo.write(f"hof                = {hof}\n")
print('hora de finalização ...> ',datetime.now())
print(hof)
print('fim')
