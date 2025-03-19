import numpy as np
import xml.etree.ElementTree as ET
import pyproj
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import random
import matplotlib
matplotlib.use('TkAgg')  # para geração não-interativa: usar 'Agg' 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime

# Nome do arquivo .osm (exportado do OpenStreetMap.org):
osm_file = "openstreetmap_masp.osm"

# Coordenadas dos limites a considerar (long min, lat min, long max, lat max):
coord_limites = (-46.6615860, -23.5633580, -46.6506850, -23.5576730)

# Define posição inicial (saída do drone) em metros
inicio_ix, inicio_iy, inicio_iz = 1, 1, 1

# Define posição destino (objetivo) em metros
dest_ix, dest_iy, dest_iz = (
    700, # posição x
    700, # posição y
    1,  # posição z
)
dest_pos = (dest_ix, dest_iy, dest_iz)

# Parâmetros de aprendizado por reforço
max_rodadas = 50
gamma = 0.5 #0.9
alpha = 0.1
epsilon = 1.0
decay_rate = 0.99 #0.995
min_epsilon = 0.01
recompensa_aproximar = 50

# Ações possíveis (frente, acima etc.)
actions = [(0, 0, 1), (0, 0, -1), 
           (0, 1, 0), (0, -1, 0), 
           (1, 0, 0), (-1, 0, 0),
           
           #(-1, -1, 0),(-1, 1, 0),(1, 1, 0),(1, -1, 0),
           #(-1, 0, -1),(-1, 0, 1),(1, 0, 1),(1, 0, -1),
           #(0, -1, -1),(0, -1, 1),(0, 1, 1),(0, 1, -1),

           #(1, 1, 1),(1, 1, -1),(1, -1, 1),(-1, 1, 1),
           #(-1, -1, -1),(-1, 1, -1),(1, -1, -1)           
           ]
todas_rodadas = []


def parse_osm_buildings(osm_file, coord_limites=None):
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # Coleta nós (nodes)
    nodes = {}
    for node in root.findall("node"):
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        if coord_limites:
            min_lon, min_lat, max_lon, max_lat = coord_limites
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
        node_id = node.get("id")
        nodes[node_id] = (lat, lon)

    # Coleta edifícios
    buildings = []
    for way in root.findall("way"):
        is_building = False
        height = None
        levels = None
        tags = {}
        for tag in way.findall("tag"):
            k = tag.get("k")
            v = tag.get("v")
            tags[k] = v
            if k == "building":
                is_building = True
            if k == "height":
                try:
                    height = float(v)
                except ValueError:
                    pass
            if k == "building:levels":
                try:
                    levels = int(v)
                except ValueError:
                    pass
        if is_building:
            nds = []
            for nd in way.findall("nd"):
                ref = nd.get("ref")
                if ref in nodes:
                    nds.append(nodes[ref])
            if len(nds) > 2:
                building = {
                    "nodes": nds,
                    "height": height,
                    "levels": levels,
                    "tags": tags,
                }
                buildings.append(building)

    return buildings


def get_projected_buildings(buildings, transformer):
    projected_buildings = []
    for b in buildings:
        nodes = b["nodes"]
        coords_lonlat = [(lon, lat) for lat, lon in nodes]
        # Converte para coordenadas projetadas
        coords_proj = [transformer.transform(lon, lat) for lon, lat in coords_lonlat]
        poly = Polygon(coords_proj)
        height = b["height"]
        levels = b["levels"]
        if height is None and levels is not None:
            height = levels * 3
        elif height is None:
            height = 10
        projected_buildings.append({"polygon": poly, "height": height})
    return projected_buildings


def get_grid_bounds_and_resolution(buildings, grid_resolution=1.0):
    min_x = min(b["polygon"].bounds[0] for b in buildings)
    min_y = min(b["polygon"].bounds[1] for b in buildings)
    max_x = max(b["polygon"].bounds[2] for b in buildings)
    max_y = max(b["polygon"].bounds[3] for b in buildings)
    max_height = max(b["height"] for b in buildings)
    x_range = np.arange(min_x, max_x, grid_resolution)
    y_range = np.arange(min_y, max_y, grid_resolution)
    z_range = np.arange(0, max_height + grid_resolution, grid_resolution)
    return x_range, y_range, z_range, min_x, min_y, max_height


def coord_to_index(coord, min_coord, resolution):
    return int((coord - min_coord) // resolution)


def create_labirinto_array(
    buildings, x_range, y_range, z_range, min_x, min_y, grid_resolution
):
    nx = len(x_range)
    ny = len(y_range)
    nz = len(z_range)
    labirinto = np.ones((nx, ny, nz), dtype=np.int8)
    for b in buildings:
        poly = b["polygon"]
        height = b["height"]
        minx, miny, maxx, maxy = poly.bounds
        ix_min = coord_to_index(minx, min_x, grid_resolution)
        iy_min = coord_to_index(miny, min_y, grid_resolution)
        ix_max = coord_to_index(maxx, min_x, grid_resolution)
        iy_max = coord_to_index(maxy, min_y, grid_resolution)
        iz_max = coord_to_index(height, 0, grid_resolution)
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                x = min_x + ix * grid_resolution + grid_resolution / 2
                y = min_y + iy * grid_resolution + grid_resolution / 2
                point = Point(x, y)
                if poly.contains(point):
                    for iz in range(iz_max + 1):
                        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                            labirinto[ix, iy, iz] = 0
    return labirinto


# Função de recompensa
def obter_recompensa(pos, pos_anterior):
    if labirinto[pos] == 2:     # chegou no destino
        return 10000000
    elif labirinto[pos] == 0:   # bateu em obstáculo
        return -100
    else:
        # Calcula a distância euclidiana até o destino
        distancia_anterior = np.linalg.norm(np.array(pos_anterior) - np.array(dest_pos))
        distancia_atual = np.linalg.norm(np.array(pos) - np.array(dest_pos))

        if distancia_atual < distancia_anterior:
            return recompensa_aproximar  # Recompensa por se aproximar
        else:
            return (recompensa_aproximar*(-1))  # Penalização por se afastar

        #return (distancia_atual*(-1))


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
        print('=' * 80)
        print(f"RODADA: {rodada}, hora: {datetime.datetime.now()}")
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

            #recompensa = obter_recompensa(nova_pos)
            recompensa = obter_recompensa(nova_pos, pos)  # Passa a posição atual e anterior

            custo = custo + 1

            # Atualizar Q-valor
            q_table[pos][action] = q_table[pos][action] + \
                alpha * (recompensa + gamma * \
                    np.max(q_table[nova_pos]) - q_table[pos][action])

            caminho.append((pos, action))  # Guarda o caminho

            # Atualiza posição
            pos = nova_pos
            #print(pos)

            # Termina a rodada
            if recompensa == 10000000 or recompensa == -100:
                break

        esta_rodada = Rodada(caminho, rodada, epsilon, custo)

        # Plota o caminho após cada rodada
        #plot_labirinto(esta_rodada)

        # Atualiza epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

        todas_rodadas.append(esta_rodada)
        print(f"Rodada: {rodada}, Epsilon: {epsilon:.2f}, Custo: {custo}")


print('Convertendo o arquivo .osm para uma matriz simbólica...')
buildings = parse_osm_buildings(osm_file, coord_limites=coord_limites)

wgs84 = pyproj.CRS("EPSG:4326")
utm_zone = pyproj.CRS("EPSG:32618")
transformer = pyproj.Transformer.from_crs(wgs84, utm_zone, always_xy=True)
projected_buildings = get_projected_buildings(buildings, transformer)
x_range, y_range, z_range, min_x, min_y, max_height = get_grid_bounds_and_resolution(
    projected_buildings, grid_resolution=1.0
)

# Define o ambiente do labirinto (3D grid)
# 1: caminho livre, 0: obstáculo, 2: objetivo
# Usa como origem o arquivo .osm para criar a matriz 3d:
labirinto = create_labirinto_array(
    projected_buildings, x_range, y_range, z_range, min_x, min_y, grid_resolution=1.0
)
print('Matriz simbólica obtida a partir do arquivo .osm!')


# Garante que as posições sejam válidas
if labirinto[inicio_ix, inicio_iy, inicio_iz] == 1:
    labirinto[inicio_ix, inicio_iy, inicio_iz] = 1
else:
    print("Ponto inicial não está em espaço livre.")

if labirinto[dest_ix, dest_iy, dest_iz] == 1:
    labirinto[dest_ix, dest_iy, dest_iz] = 2
else:
    print("Ponto objetivo não está em espaço livre.")

# Define a tabela Q
# Configuração do Q-Learning: Inicializa-se a tabela Q com zeros, 
# define-se as ações e parâmetros de aprendizado
q_table = np.zeros((*labirinto.shape, 6))

# Executa o treinamento (encontra o melhor caminho)
posicao_inicial = (inicio_ix, inicio_iy, inicio_iz)

print('Iniciando treino do drone (agente)...')
treinar_ql(max_rodadas, posicao_inicial)

print(f"FIM DO TREINO, hora: {datetime.datetime.now()}")
print('Treino do drone (agente) concluído...')


rodada_menor_custo = obtem_rodada_menor_custo(todas_rodadas)

print(f"Rodada com menor custo: {rodada_menor_custo.rodada}")
print("Comandos:")
obtem_comandos(rodada_menor_custo.caminho)
print("Caminho:")
print(rodada_menor_custo.caminho)

plot_labirinto(rodada_menor_custo)

print("FIM")
