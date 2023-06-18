from typing import List
import gymnasium
import numpy as np
import random
import collections
import pickle

random.seed(0)
np.random.seed(0)

numEpocas =  
fatorDesconto = 0.8
taxaAprendizado = 0.9
intervaloRelatorio = 500

# verifica se existe um arquivo pickle com a tabela Q_table
# se existir carrega a tabela Q_table
try:
    pickle_file = open('treinamento.pickle', 'rb')
    Q = pickle.load(pickle_file)
    pickle_file.close()
# se não existir cria uma nova tabela Q_table
except:
    print("Não encontrei nada, irei criar um novo arquivo")
    Q = collections.defaultdict(float)

# converte um número ou uma sequência de números em uma representação binária compacta
def converter(n):
    out = 0
    for num in n:
        for bit in np.binary_repr(num, width=8):
            out = (out << 1) | int(bit)
    out = out & 0xFFFFFFFFFFFFFFFF

    return out

# retorna o maior valor de Q_table
def max_q(s, Q, env):
    max_q = float('-inf')
    for acao in range(env.action_space.n):
        estadoAcao = (s, acao)
        if Q[estadoAcao] > max_q:
            max_q = Q[estadoAcao]

    return max_q

def main():
    # cria o ambiente
    env = gymnasium.make("ALE/Enduro-v5", obs_type='ram', render_mode="human")
    env.seed(0)
    # cria uma lista para armazenar as recompensas
    recompensas = []
    
    # loop para treinar o agente
    for epoca in range(1, numEpocas + 1):
        # reseta o ambiente e converte o estado para binário
        estado = converter(env.reset()[0]) 
        # variável para armazenar a recompensa da época
        epocaRecompensa = 0
        # printa a epoca atual
        print("Epoca :", epoca)
        while True:
            # numero aleatório para escolher a ação
            randAcao = np.random.random()
            # usado para determinar se o agente irá explorar ou aproveitar o conhecimento existent
            if randAcao < 0.50:
                # o agente escolhe uma ação aleatória
                acao = env.action_space.sample()
            else:
                # o agente aproveita o conhecimento existente para selecionar a ação
                # escolhe a ação com o maior valor Q usando
                acao = np.argmax([Q[(estado, a)] for a in range(env.action_space.n)])

            #executa a ação escolhida
            passo = env.step(acao) 
            # extrai o novo estado da tupla e converte-o para o formato binário
            novoEstado = converter(passo[0])
            # obtém a recompensa do passo atual do jogo 
            recompensa = passo[1]
            # obtém o valor booleano indicando se o jogo terminou após o passo atual
            fim = passo[2]

            
            # calcula o valor de Q-alvo para o par estado-ação atual
            Qtarget = recompensa + fatorDesconto * max_q(estado, Q, env)
            # atualiza o valor de Q para o par estado-ação atual
            Q[(estado, acao)] = (1-taxaAprendizado) * Q[(estado, acao)] + taxaAprendizado * Qtarget
            # adiciona a recompensa imediata obtida na etapa atual à recompensa total 
            epocaRecompensa += recompensa
            # atualiza o estado atual
            estado = novoEstado
            # verifica se há recompensa
            if recompensa != 0:
                print("Recompensa: ", recompensa)
                print("Estado: ", estado)
                print("Q_table[Estado, Acao]: ", Q[(estado, acao)])
            # verifica se o jogo acabou
            if fim:
                recompensas.append(epocaRecompensa)
                break

    pickle_file = open('treinamento.pickle', 'wb')
    pickle.dump(Q, pickle_file)
    pickle_file.close()

if __name__ == '__main__':
    main()