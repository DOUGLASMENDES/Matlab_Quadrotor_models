import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import matplotlib
matplotlib.use('TkAgg')  # para geração não-interativa: usar 'Agg' 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir o ambiente do labirinto (7x7x3 grid 3D)
labirinto = np.array([
    [
        [1, 1, 1, 1, 0, 1, 2],
        [1, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ],
    [
        [1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1]
    ],
    [
        [1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
])

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

# Parâmetros DQN
max_episodios = 10
gamma = 0.95
alpha = 0.001
epsilon = 1.0
decay_rate = 0.995
min_epsilon = 0.01
batch_size = 32
experience_replay_capacity = 1000

# Definir Replay Buffer
experience_replay = deque(maxlen=experience_replay_capacity)

# Rede Neural para estimar valores Q
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Inicializar a rede e otimizador
state_size = 3  # Coordenadas x, y, z como estado
action_size = 6  # Número de ações possíveis
policy_net = DQN(state_size, action_size)
optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# Ações possíveis: [acima, abaixo, esquerda, direita, frente, trás]
actions = [(0, 0, 1), (0, 0, -1), 
           (0, 1, 0), (0, -1, 0), 
           (1, 0, 0), (-1, 0, 0)]
todas_rodadas = []

# Funções para o ambiente
def obter_recompensa(pos):
    if labirinto[pos] == 2:
        return 100
    elif labirinto[pos] == 0:
        return -100
    else:
        return -1

def posicao_valida(pos):
    x, y, z = pos
    return 0 <= x < labirinto.shape[0] and 0 <= y < labirinto.shape[1] and 0 <= z < labirinto.shape[2] and labirinto[pos] != 0

def atualizar_posicao(pos, action):
    x, y, z = pos
    dx, dy, dz = actions[action]
    nova_pos = (x + dx, y + dy, z + dz)
    return nova_pos if posicao_valida(nova_pos) else pos

def obtem_comandos(cam):
    for c in cam:
        print(actions[c[1]])

# Função para plotar o labirinto e o drone
def plot_labirinto_dqn(rodada_voo):
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
    
    # Plotando o caminho do drone
    x_vals, y_vals, z_vals = zip(*[pos for pos, _ in rodada_voo.caminho])
    ax.plot(x_vals, y_vals, z_vals, color="blue", linewidth=2, label="Caminho do Drone")
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color="blue", s=200, label="Drone")

    # Configurações do gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Rodada {rodada_voo.rodada} | Custo {rodada_voo.custo:.2f}')
    ax.legend()
    plt.show()    
    
class Rodada:
    def __init__(self, cam, epi, eps, cus, rec):
        self.caminho = cam
        self.rodada = epi
        self.epsilon = eps
        self.custo = cus
        self.recompensa = rec

def obtem_rodada_maior_recompensa(rodadas):
    maior = 0
    for rodada in rodadas:
        if rodada.recompensa > maior:
            maior = rodada.recompensa
            rodada_maior_recompensa = rodada
    return rodada_maior_recompensa

# Função de treinamento do DQN
def treinar_dqn(episodios, posicao_inicial):
    global epsilon
    global custo
    for episodio in range(episodios):
        pos = posicao_inicial
        total_reward = 0
        caminho = []
        custo = 0
        
        while True:
            state = torch.tensor(pos, dtype=torch.float32)
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(action_size))
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
            
            nova_pos = atualizar_posicao(pos, action)
            recompensa = obter_recompensa(nova_pos)
            total_reward += recompensa
            done = recompensa == 100 or recompensa == -100
            custo = custo + 1
                        
            # Adicionar a transição ao replay buffer
            experience_replay.append((pos, action, recompensa, nova_pos, done))
            caminho.append((pos, action))
            
            pos = nova_pos
            
            # Treinamento com replay de experiência
            if len(experience_replay) >= batch_size:
                batch = random.sample(experience_replay, batch_size)
                estados, acoes, recompensas, proximos_estados, finais = zip(*batch)
                
                estados_tensor = torch.tensor(estados, dtype=torch.float32)
                acoes_tensor = torch.tensor(acoes, dtype=torch.long).unsqueeze(1)
                recompensas_tensor = torch.tensor(recompensas, dtype=torch.float32)
                proximos_estados_tensor = torch.tensor(proximos_estados, dtype=torch.float32)
                finais_tensor = torch.tensor(finais, dtype=torch.float32)
                
                q_vals = policy_net(estados_tensor).gather(1, acoes_tensor).squeeze()
                max_q_futuro = policy_net(proximos_estados_tensor).max(1)[0]
                q_alvo = recompensas_tensor + gamma * max_q_futuro * (1 - finais_tensor)
                
                loss = loss_fn(q_vals, q_alvo.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break

        esta_rodada = Rodada(caminho, episodio, epsilon, custo, recompensa)
        todas_rodadas.append(esta_rodada)

        # Plotar o ambiente e o caminho do drone após cada episódio
        #plot_labirinto(caminho, episodio+1)
        #plot_labirinto(esta_rodada)
        
        # Decaimento de epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)
        
        print(f"Episódio {episodio+1}, Recompensa Total: {total_reward}")

# Iniciar o treinamento
posicao_inicial = (0, 0, 0)
treinar_dqn(max_episodios, posicao_inicial)


rodada_maior_recompensa = obtem_rodada_maior_recompensa(todas_rodadas)

print("Comandos:")
obtem_comandos(rodada_maior_recompensa.caminho)
print("Caminho:")
print(rodada_maior_recompensa.caminho)
print(f"Rodada com maior recompensa: { \
            rodada_maior_recompensa.rodada}, Recompensa: { \
            rodada_maior_recompensa.recompensa}")


plot_labirinto_dqn(rodada_maior_recompensa)

print("FIM")
