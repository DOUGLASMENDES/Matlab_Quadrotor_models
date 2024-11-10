import numpy as np
import random

import matplotlib
matplotlib.use('TkAgg')  # para geração não-interativa: usar 'Agg' 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define o ambiente do labirinto (3D grid)
# 1: caminho livre, 0: obstáculo, 2: objetivo
labirinto = np.array([
    [   # cada vetor representa o plano x,z
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1]
    ],
    [
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1]
    ],
    [
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ],
    [
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ],
    [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 2]
    ]
])

# Parâmetros de aprendizado por reforço
max_rodadas = 200
gamma = 0.9
alpha = 0.1
epsilon = 1.0
decay_rate = 0.995
min_epsilon = 0.01


# Define a tabela Q
# Configuração do Q-Learning: Inicializa-se a tabela Q com zeros, 
# define-se as ações e parâmetros de aprendizado
q_table = np.zeros((*labirinto.shape, 6))
actions = [(0, 0, 1), (0, 0, -1), 
           (0, 1, 0), (0, -1, 0), 
           (1, 0, 0), (-1, 0, 0)]
todas_rodadas = []

# Função de recompensa
def obter_recompensa(pos):
    if labirinto[pos] == 2:     # chegou no destino
        return 100
    elif labirinto[pos] == 0:   # bateu em obstáculo
        return -100
    else:
        return -1

# Verificar se a posição é válida
# (posição não é válida caso passe dos limites 
# do ambiente ou encontre um obstáculo)
def posicao_valida(pos):
    x, y, z = pos
    return 0 <= x < labirinto.shape[0] and \
            0 <= y < labirinto.shape[1] and \
            0 <= z < labirinto.shape[2] and \
            labirinto[pos] != 0

# Atualiza posição com base na ação
def atualizar_posicao(pos, action):
    x, y, z = pos
    dx, dy, dz = actions[action]
    nova_pos = (x + dx, y + dy, z + dz)
    return nova_pos if posicao_valida(nova_pos) else pos

def obtem_rodada_menor_custo(rodadas):
    menor = 100000000
    for rodada in rodadas:
        if rodada.custo < menor:
            menor = rodada.custo
            rodada_menor_custo = rodada
    return rodada_menor_custo

def obtem_comandos(cam):
    for c in cam:
        print(actions[c[1]])

# Função para plotar o labirinto e o drone
def plot_labirinto(rodada_voo):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotando obstáculos e objetivo
    for x in range(labirinto.shape[0]):
        for y in range(labirinto.shape[1]):
            for z in range(labirinto.shape[2]):
                if labirinto[x, y, z] == 0:
                    ax.scatter(x, y, z, color='black', s=300, marker='s',
                               label="Obstáculo" if (x, y, z) == (0, 0, 0) else "")
                elif labirinto[x, y, z] == 2:
                    ax.scatter(x, y, z, color='yellow', s=100, label="Objetivo")
    
    # Plotar o caminho do drone
    x_vals, y_vals, z_vals = zip(*[pos for pos, _ in rodada_voo.caminho])
    ax.plot(x_vals, y_vals, z_vals, color="blue", linewidth=2, label="Caminho do Drone")
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color="blue", s=100, label="Drone")

    # Configurações do gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Rodada {rodada_voo.rodada} | Custo {rodada_voo.custo:.2f}')
    ax.legend()
    plt.show()    
    
class Rodada:
    def __init__(self, cam, epi, eps, cus):
        self.caminho = cam
        self.rodada = epi
        self.epsilon = eps
        self.custo = cus

# Treinamento Q-Learning
# Executa múltiplas rodadas de treinamento, atualizando a tabela Q 
# com base nas recompensas e penalizações recebidas. O parâmetro epsilon
# é reduzido ao longo do tempo para equilibrar exploração e exploração.
def treinar_ql(rodadas, posicao_inicial):
    global epsilon
    global custo
    for rodada in range(1, rodadas + 1):
        pos = posicao_inicial
        caminho = []
        custo = 0

        while True:
            # Escolhe ação
            if random.uniform(0, 1) < epsilon:
                # escolhe uma ação qualquer dentro das possibilidades
                action = random.choice(range(len(actions)))
            else:
                action = np.argmax(q_table[pos])

            # Obter próxima posição e recompensa
            nova_pos = atualizar_posicao(pos, action)
            recompensa = obter_recompensa(nova_pos)
            custo = custo + 1

            # Atualizar Q-valor
            q_table[pos][action] = q_table[pos][action] + \
                alpha * (recompensa + gamma * \
                    np.max(q_table[nova_pos]) - q_table[pos][action])

            caminho.append((pos, action))  # Guarda o caminho

            # Atualiza posição
            pos = nova_pos

            # Termina a rodada
            if recompensa == 100 or recompensa == -100:
                break

        esta_rodada = Rodada(caminho, rodada, epsilon, custo)

        # Plota o caminho após cada rodada
        #plot_labirinto(esta_rodada)

        # Atualiza epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        todas_rodadas.append(esta_rodada)
        print(f"Rodada: {rodada}, Epsilon: {epsilon:.2f}, Custo: {custo}")

# Executa o treinamento (encontra o melhor caminho)
posicao_inicial = (0, 0, 0)
treinar_ql(max_rodadas, posicao_inicial)

rodada_menor_custo = obtem_rodada_menor_custo(todas_rodadas)

print(f"Rodada com menor custo: {rodada_menor_custo.rodada}")
print("Comandos:")
obtem_comandos(rodada_menor_custo.caminho)
print("Caminho:")
print(rodada_menor_custo.caminho)

plot_labirinto(rodada_menor_custo)

print("FIM")
