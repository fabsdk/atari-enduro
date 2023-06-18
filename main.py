from typing import List
import gymnasium
import numpy as np
import random
import collections
import pickle

random.seed(0)
np.random.seed(0)

numEpocas = 500 
fatorDesconto = 0.8
taxaAprendizado = 0.9
intervaloRelatorio = 500

# 
try:
    pickle_file = open('treinamento.pickle', 'rb')
    Q = pickle.load(pickle_file)
    pickle_file.close()
except:
    print("Não encontrei nada, irei criar um novo arquivo")
    Q = collections.defaultdict(float)


def converter(n):
    out = 0
    for num in n:
        for bit in np.binary_repr(num, width=8):
            out = (out << 1) | int(bit)
    out = out & 0xFFFFFFFFFFFFFFFF

    return out

def max_q(s, Q, env):
    max_q = float('-inf')
    for acao in range(env.action_space.n):
        estadoAcao = (s, acao)
        if Q[estadoAcao] > max_q:
            max_q = Q[estadoAcao]

    return max_q

def main():
    env = gymnasium.make("ALE/Enduro-v5", obs_type='ram', render_mode="human")
    env.seed(0)
    recompensas = []
    
    for epoca in range(1, numEpocas + 1):
        estado = converter(env.reset()[0]) 
        epocaRecompensa = 0
        print("Epoca :", epoca)
        while True:
            
            randAcao = np.random.random()
            if randAcao < 0.50:
                acao = env.action_space.sample()
            else:
                acao = np.argmax([Q[(estado, a)] for a in range(env.action_space.n)])

            passo = env.step(acao) 
            novoEstado = converter(passo[0])
            recompensa = passo[1]
            fim = passo[2]

            Qtarget = recompensa + fatorDesconto * max_q(estado, Q, env)
            Q[(estado, acao)] = (1-taxaAprendizado) * Q[(estado, acao)] + taxaAprendizado * Qtarget
            epocaRecompensa += recompensa
            estado = novoEstado
            if recompensa != 0:
                print("Recompensa: ", recompensa)
                print("Estado: ", estado)
                print("Q_table[Estado, Acao]: ", Q[(estado, acao)])
            if fim:
                recompensas.append(epocaRecompensa)
                break

    pickle_file = open('treinamento.pickle', 'wb')
    pickle.dump(Q, pickle_file)
    pickle_file.close()

if __name__ == '__main__':
    main()