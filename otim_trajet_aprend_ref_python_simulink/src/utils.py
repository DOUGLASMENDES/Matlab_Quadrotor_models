import numpy as np
import xml.etree.ElementTree as ET
import pyproj
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import random
import json
import matplotlib

matplotlib.use("TkAgg")  # para geração não-interativa: usar 'Agg'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime

resolucao_grade = 1.0

# Arquivo de log com os comandos gravados:
arquivo_log_comandos = 'log_comandos.txt'

# Arquivo de saída:
arquivo_log = 'log.txt'

# Nome do arquivo .osm (exportado do OpenStreetMap.org):
arquivo_osm = "openstreetmap_masp.osm"

# Coordenadas dos limites a considerar (long min, lat min, long max, lat max):
# Engloba a região da Av. Paulista, MASP e entorno dentro do município de São Paulo.
coord_limites = (-46.6615860, -23.5633580, -46.6506850, -23.5576730)

# Define posição inicial (partida do drone) em metros (na grade 3d)
inicio_ix, inicio_iy, inicio_iz = 1, 1, 1
posicao_inicial = (inicio_ix, inicio_iy, inicio_iz)

# Define uma posição destino (objetivo) arbitrária em metros
# apenas para teste do algoritmo (na grade 3d)
dest_ix, dest_iy, dest_iz = (600, 600, 1)
dest_pos = (dest_ix, dest_iy, dest_iz)

# Parâmetros da grade 3d:
resolucao_grade = 1.0  # ajustar para melhorar/reduzir performance da busca

# Parâmetros do aprendizado por reforço (AI - Reinforcement Learning)
max_rodadas = 100
gamma = 0.5  # 0.9
alpha = 0.1
epsilon = 1.0
taxa_decaimento = 0.99  # 0.995
epsilon_min = 0.01
recompensa_aproximar = 50
# vamos definir um custo máximo para uma determinada rodada e
# evitar que o agente fique preso ou perdido durante o treino:
custo_maximo = 20000000

# Ações possíveis do drone (frente, acima, abaixo, lado esq etc.) nos eixos x,y,z
comandos = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
#comandos = [(0, 0, 1), (0, 0, -1), 
#           (0, 1, 0), (0, -1, 0), 
#           (1, 0, 0), (-1, 0, 0),
#           
#           #(-1, -1, 0),(-1, 1, 0),(1, 1, 0),(1, -1, 0),
#           #(-1, 0, -1),(-1, 0, 1),(1, 0, 1),(1, 0, -1),
#           #(0, -1, -1),(0, -1, 1),(0, 1, 1),(0, 1, -1),

           #(1, 1, 1),(1, 1, -1),(1, -1, 1),(-1, 1, 1),
           #(-1, -1, -1),(-1, 1, -1),(1, -1, -1)           
#           ]

todas_rodadas = []


# Classe python que identifica uma rodada:
class Rodada:
    def __init__(self, cam, epi, eps, cus):
        self.caminho = cam
        self.rodada = epi
        self.epsilon = eps
        self.custo = cus

def check_inicio_dest_validos(labirinto):
    # Garante que as posições sejam válidas
    if labirinto[inicio_ix, inicio_iy, inicio_iz] == 1:
        labirinto[inicio_ix, inicio_iy, inicio_iz] = 1
    else:
        log("Ponto inicial não está em espaço livre.")

    if labirinto[dest_ix, dest_iy, dest_iz] == 1:
        labirinto[dest_ix, dest_iy, dest_iz] = 2
    else:
        log("Ponto objetivo não está em espaço livre.")

# Função para identificar edificações no arquivo .osm:
def parse_edificacoes_osm(arquivo_osm, coord_limites=None):
    tree = ET.parse(arquivo_osm)
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

    # Coleta edificações
    edificacoes = []
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
                edificacoes.append(building)

    return edificacoes


def obter_wsg84():
    # vamos criar um objeto de referência de sistema de coordenadas (CRS, ou Coordinate
    # Reference System). Especificamente, definimos o sistema de coordenadas WGS 84
    # (World Geodetic System 1984) com base no código EPSG 4326

    # WGS 84 (World Geodetic System 1984): sistema de referência global usado para representar
    # coordenadas georeferenciadas (latitude e longitude), baseado em um elipsoide que aproxima a
    # forma da Terra. É amplamente utilizado em GPS, mapas digitais e sistemas de navegação.

    # EPSG:4326: identificador oficial para o sistema WGS 84 na base de dados EPSG
    # (European Petroleum Survey Group). Representa coordenadas georeferenciadas em graus decimais,
    # com latitude e longitude.

    # usamos este objeto para configurar o sistema de referência de coordenadas WGS 84,
    # essencial para trabalhar com dados geográficos em latitude/longitude e realizar
    # projeções geoespaciais
    wgs84 = pyproj.CRS("EPSG:4326")
    log(" " * 80)
    log(wgs84)
    log(" " * 80)
    return wgs84

def obter_zona_global():
    # Definimos o sistema de referência do mapa web (EPSG:3857). Para maior precisão
    # podemos usar o sistema de referência específico da zona UTM, que depende da
    # região geográfica. Exemplo: região de Nova York (USA), usa-se 'EPSG:32618'
    # (identificador oficial do sistema UTM para a Zona 18 Norte, baseado no elipsoide WGS 84.
    # O código EPSG 32618 especifica que o sistema está em metros e abrange a região do
    # hemisfério norte na zona 18). Para a região de São Paulo (Brasil), usa-se
    # 'EPSG:32723', pois São Paulo está na zona 23S (sul do equador), que corresponde ao
    # código EPSG 32723 para o sistema UTM baseado no WGS 84 no hemisfério sul.
    zona_global = pyproj.CRS("EPSG:3857")  # Sistema global usado em mapas na web
    log(" " * 80)
    log(zona_global)
    log(" " * 80)
    return zona_global


def obtem_edificacoes_projetadas(edificacoes, transformer):
    edificacoes_projetadas = []
    for ed in edificacoes:
        nodes = ed["nodes"]
        coords_lonlat = [(lon, lat) for lat, lon in nodes]
        # Converte para coordenadas projetadas
        coords_proj = [transformer.transform(lon, lat) for lon, lat in coords_lonlat]
        poly = Polygon(coords_proj)
        height = ed["height"]
        levels = ed["levels"]
        if height is None and levels is not None:
            height = levels * 3
        elif height is None:
            height = 10
        edificacoes_projetadas.append({"polygon": poly, "height": height})
    return edificacoes_projetadas


def obtem_limites_e_resolucoes_grade(edificacoes, resolucao_grade=resolucao_grade):
    min_x = min(ed["polygon"].bounds[0] for ed in edificacoes)
    min_y = min(ed["polygon"].bounds[1] for ed in edificacoes)
    max_x = max(ed["polygon"].bounds[2] for ed in edificacoes)
    max_y = max(ed["polygon"].bounds[3] for ed in edificacoes)
    max_height = max(ed["height"] for ed in edificacoes)
    x_range = np.arange(min_x, max_x, resolucao_grade)
    y_range = np.arange(min_y, max_y, resolucao_grade)
    z_range = np.arange(0, max_height + resolucao_grade, resolucao_grade)
    return x_range, y_range, z_range, min_x, min_y, max_height


# Função para discretizar coordenadas reais em índices da grade:
def coord_para_index(coord, min_coord, resolucao):
    return int((coord - min_coord) // resolucao)


# Função que converte uma posição na grade (índices x, y, z) para
# coordenadas georeferenciadas.
def index_para_coord_geografica(index, min_x, min_y, resolucao, transformer):
    x_index, y_index, z_index = index

    # Converte índices para coordenadas contínuas (projetadas)
    x_proj = min_x + x_index * resolucao
    y_proj = min_y + y_index * resolucao
    altitude = z_index * resolucao  # resolução no z é em metros

    # Converte de coordenadas projetadas para geográficas (lon, lat)
    lon, lat = transformer.transform(x_proj, y_proj, direction="INVERSE")
    return lat, lon, altitude


def gera_matriz_labirinto(
    edificacoes, x_range, y_range, z_range, min_x, min_y, resolucao_grade
):
    nx = len(x_range)
    ny = len(y_range)
    nz = len(z_range)
    labirinto = np.ones((nx, ny, nz), dtype=np.int8)
    for ed in edificacoes:
        poly = ed["polygon"]
        height = ed["height"]
        minx, miny, maxx, maxy = poly.bounds
        ix_min = coord_para_index(minx, min_x, resolucao_grade)
        iy_min = coord_para_index(miny, min_y, resolucao_grade)
        ix_max = coord_para_index(maxx, min_x, resolucao_grade)
        iy_max = coord_para_index(maxy, min_y, resolucao_grade)
        iz_max = coord_para_index(height, 0, resolucao_grade)
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                x = min_x + ix * resolucao_grade + resolucao_grade / 2
                y = min_y + iy * resolucao_grade + resolucao_grade / 2
                point = Point(x, y)
                if poly.contains(point):
                    for iz in range(iz_max + 1):
                        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                            labirinto[ix, iy, iz] = 0
    return labirinto


# Função de recompensa
def obter_recompensa(pos, pos_anterior, labirinto):
    if labirinto[pos] == 2:  # chegou no destino
        return 10000000
    elif labirinto[pos] == 0:  # bateu em obstáculo
        return -100
    else:
        # Calcula a distância euclidiana até o destino
        distancia_anterior = np.linalg.norm(np.array(pos_anterior) - np.array(dest_pos))
        distancia_atual = np.linalg.norm(np.array(pos) - np.array(dest_pos))

        if distancia_atual < distancia_anterior:
            return recompensa_aproximar  # Recompensa por se aproximar
        else:
            return recompensa_aproximar * (-1)  # Penalização por se afastar

        # return (distancia_atual*(-1))


# Verificar se a posição é válida
# (posição não é válida caso passe dos limites
# do ambiente ou encontre um obstáculo)
def posicao_valida(pos, labirinto):
    x, y, z = pos
    return (
        0 <= x < labirinto.shape[0]
        and 0 <= y < labirinto.shape[1]
        and 0 <= z < labirinto.shape[2]
        and labirinto[pos] != 0
    )


# Atualiza posição com base na ação
def atualizar_posicao(pos, comando, labirinto, discretizado=False):
    x, y, z = pos
    if discretizado:
        dx, dy, dz = comando
    else:
        dx, dy, dz = comandos[comando]
    nova_pos = (x + dx, y + dy, z + dz)
    return nova_pos if posicao_valida(nova_pos, labirinto) else pos


def obtem_rodada_menor_custo(rodadas):
    menor = 100000000
    for rodada in rodadas:
        if rodada.custo < menor:
            menor = rodada.custo
            rodada_menor_custo = rodada
    return rodada_menor_custo


def obtem_comandos(cam):
    coms = [comandos[c[1]] for c in cam]
    comandos_discretizados = discretizar_comandos(coms)
    return comandos_discretizados


def loga_comandos(coms):
    for c in coms:
        log(c)


# Função para plotar o labirinto e o drone
# (apenas quando a matriz simbólica é pequena e possui poucos pontos,
# caso contrário, o sistema não consegue fazer a plotagem. Quando isso
# ocorre, é necessário enviar os comandos para outro software)
def plot_labirinto(rodada_voo, labirinto):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plotando obstáculos e objetivo
    for x in range(labirinto.shape[0]):
        for y in range(labirinto.shape[1]):
            for z in range(labirinto.shape[2]):
                if labirinto[x, y, z] == 0:
                    ax.scatter(
                        x,
                        y,
                        z,
                        color="black",
                        s=300,
                        marker="s",
                        label="Obstáculo" if (x, y, z) == (0, 0, 0) else "",
                    )
                elif labirinto[x, y, z] == 2:
                    ax.scatter(x, y, z, color="yellow", s=100, label="Objetivo")

    # Plotar o caminho do drone
    x_vals, y_vals, z_vals = zip(*[pos for pos, _ in rodada_voo.caminho])
    ax.plot(x_vals, y_vals, z_vals, color="blue", linewidth=2, label="Caminho do Drone")
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color="blue", s=100, label="Drone")

    # Configurações do gráfico
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Rodada {rodada_voo.rodada} | Custo {rodada_voo.custo:.2f}")
    ax.legend()
    plt.show()




# Treinamento Q-Learning (Reinforcement Learning)
# Executa múltiplas rodadas de treinamento, atualizando a tabela Q
# com base nas recompensas e penalizações recebidas. O parâmetro epsilon
# é reduzido ao longo do tempo para equilibrar exploração e exploração.
def treinar_ql(rodadas, posicao_inicial, labirinto, tabela_q):
    global epsilon
    global custo
    for rodada in range(1, rodadas + 1):
        log("=" * 80)
        log(f"RODADA: {rodada}, hora: {datetime.datetime.now()}")
        pos = posicao_inicial
        caminho = []
        custo = 0

        while True:
            # Escolhe ação
            if random.uniform(0, 1) < epsilon:
                # escolhe uma ação qualquer dentro das possibilidades
                comando = random.choice(range(len(comandos)))
            else:
                comando = np.argmax(tabela_q[pos])

            # Obter próxima posição e recompensa
            nova_pos = atualizar_posicao(pos, comando, labirinto)
            recompensa = obter_recompensa(
                nova_pos, pos, labirinto
            )  # Passa a posição atual e anterior
            custo = custo + 1

            # Atualizar Q-valor
            tabela_q[pos][comando] = tabela_q[pos][comando] + alpha * (
                recompensa + gamma * np.max(tabela_q[nova_pos]) - tabela_q[pos][comando]
            )

            caminho.append((pos, comando))  # Guarda o caminho

            # Atualiza posição
            pos = nova_pos
            # log(pos)

            # Termina a rodada se chegou ao destino ou se custo está mto alto
            # (para evitar que o agente fique preso)
            if recompensa == 10000000 or recompensa == -100 or custo >= custo_maximo:
                break

        esta_rodada = Rodada(caminho, rodada, epsilon, custo)

        # Plota o caminho após cada rodada (apenas c/ peq núm de pontos)
        # plot_labirinto(esta_rodada, labirinto)

        # Atualiza epsilon
        epsilon = max(epsilon_min, epsilon * taxa_decaimento)

        todas_rodadas.append(esta_rodada)
        log(f"Rodada: {rodada}, Epsilon: {epsilon:.2f}, Custo: {custo}")


# Função que faz a escrita em um arquivo de log específico:
def log(*args, sep=" ", end="\n", mode="a", encoding="utf-8"):
    try:
        print(*args)
        with open(arquivo_log, mode, encoding=encoding) as f:
            print(*args, sep=sep, end=end, file=f)
    except Exception as e:
        print(f"Erro ao escrever no arquivo: {e}")


# Função que identifica em quais eixos ocorreram variação:
def eixos_variacao(comando):
    return tuple((comando[i] > 0) for i in range(3))


# Função que elimina da lista de comandos, comandos que são reduntantes:
def elimina_comand_redundantes(comandos):
    resultado = []
    acumulador = None

    for comando in comandos:
        if acumulador is None:
            acumulador = comando
        elif tuple(-x for x in comando) == acumulador:
            # Remove movimentos redundantes de ida e volta
            acumulador = None
        else:
            # Adiciona o comando acumulado e inicia um novo
            if acumulador:
                resultado.append(acumulador)
            acumulador = comando
    # Adiciona o último comando acumulado, se houver
    if acumulador:
        resultado.append(acumulador)
    return resultado


def discretizar_comandos(comandos):
    # Lista para armazenar comandos simplificados
    resultado = []
    acumulador = None
    # faz uma primeira passagem de eliminação de redundantes:
    comandos = elimina_comand_redundantes(comandos)

    for comando in comandos:
        if acumulador is None:
            acumulador = comando
        elif tuple(-x for x in comando) == acumulador:
            # Remove movimentos redundantes de ida e volta (+1 vez)
            acumulador = None
        elif (comando == acumulador) or (
            eixos_variacao(acumulador) == eixos_variacao(comando)
        ):
            # Soma comandos repetidos
            acumulador = tuple(acumulador[i] + comando[i] for i in range(3))
        else:
            # Adiciona o comando acumulado e inicia um novo
            if acumulador:
                resultado.append(acumulador)
            acumulador = comando
        # comando_ant = comando

    # Adiciona o último comando acumulado, se houver
    if acumulador:
        resultado.append(acumulador)
    return resultado


def gera_lista_posicoes(posicao_inicial, comandos_discr, labirinto):
    posicoes = [posicao_inicial]
    for comando in comandos_discr:
        posicao = atualizar_posicao(posicoes[-1], comando, labirinto, discretizado=True)
        posicoes.append(posicao)
    return posicoes


def posicoes_grade_para_geograficas(
    posicoes, min_x, min_y, resolucao_grade, transformer
):
    posicoes_geograficas = []
    for pos in posicoes:
        lat, long, alt = index_para_coord_geografica(
            pos, min_x, min_y, resolucao_grade, transformer
        )
        posicoes_geograficas.append((lat, long, alt))
    return posicoes_geograficas




# Função que adiciona uma rota de drone ao arquivo .osm.
def adicionar_rota_no_osm(arquivo_osm, rota, arquivo_saida):
    # Carrega o arquivo .osm
    tree = ET.parse(arquivo_osm)
    root = tree.getroot()

    # Define o namespace, se necessário
    osm_namespace = root.tag.split("}")[0][1:] if "}" in root.tag else ""

    # Adiciona nós ao arquivo .osm
    nodes = []
    for i, (lat, lon, alt) in enumerate(rota):
        node_id = f"-{i + 1}"  # IDs negativos são usados para elementos temporários
        node = ET.Element(
            "node",
            {
                "id": node_id,
                "lat": str(lat),  # Converte lat para string
                "lon": str(lon),  # Converte lon para string
                "version": "1",
                "visible": "true",
            },
        )
        # Adiciona uma tag opcional para a altitude
        tag = ET.Element(
            "tag", {"k": "altitude", "v": str(alt)}
        )  # Converte alt para string
        node.append(tag)
        root.append(node)
        nodes.append(node_id)

    # Cria um caminho (way) conectando os nós
    way_id = "-1000"  # ID temporário para o caminho
    way = ET.Element("way", {"id": way_id, "version": "1", "visible": "true"})
    for node_id in nodes:
        nd = ET.Element("nd", {"ref": node_id})
        way.append(nd)
    # Adiciona uma tag opcional para identificar a rota
    tag = ET.Element("tag", {"k": "drone_route", "v": "yes"})
    way.append(tag)
    root.append(way)

    # Salva o arquivo .osm atualizado
    tree.write(arquivo_saida, encoding="utf-8", xml_declaration=True)


def adicionar_rota_no_osm_ajustada(arquivo_osm, rota, arquivo_saida):
    # Carrega o arquivo .osm
    tree = ET.parse(arquivo_osm)
    root = tree.getroot()

    # Adiciona nós ao arquivo .osm
    nodes = []
    for i, (lat, lon, alt) in enumerate(rota):
        node_id = f"-{i + 1}"  # IDs negativos são usados para elementos temporários
        node = ET.Element("node", {
            "id": node_id,
            "lat": str(lat),
            "lon": str(lon),
            "version": "1",
            "visible": "true"
        })
        # Adiciona uma tag opcional para a altitude
        node.append(ET.Element("tag", {"k": "ele", "v": str(alt)}))  # Altitude como tag 'ele'
        root.append(node)
        nodes.append(node_id)

    # Cria um caminho (way) conectando os nós
    way_id = "-1000"  # ID temporário para o caminho
    way = ET.Element("way", {
        "id": way_id,
        "version": "1",
        "visible": "true"
    })
    for node_id in nodes:
        nd = ET.Element("nd", {"ref": node_id})
        way.append(nd)
    # Adiciona as tags que o OSM2World reconhece
    way.append(ET.Element("tag", {"k": "highway", "v": "path"}))
    way.append(ET.Element("tag", {"k": "drone_route", "v": "yes"}))
    root.append(way)

    # Salva o arquivo .osm atualizado
    tree.write(arquivo_saida, encoding="utf-8", xml_declaration=True)


def converter_para_qgc_plan(matriz_posicoes, nome_arquivo):
    # Estrutura básica do arquivo .plan
    plano = {
        "fileType": "Plan",
        "version": 1,
        "groundStation": "QGroundControl",
        "mission": {
            "version": 2,
            "firmwareType": 0,
            "vehicleType": 2,
            "cruiseSpeed": 15,
            "hoverSpeed": 5,
            "items": [],
            "plannedHomePosition": [
                matriz_posicoes[0][0],
                matriz_posicoes[0][1],
                matriz_posicoes[0][2]
            ]
        },
        "geoFence": {
            "circles": [],
            "polygons": [],
            "version": 2
        },
        "rallyPoints": {
            "points": [],
            "version": 2
        }
    }

    # Adiciona cada posição como um item de missão
    for idx, (lat, lon, alt) in enumerate(matriz_posicoes):
        item = {
            "AMSLAltAboveTerrain": None,
            "Altitude": alt,
            "AltitudeMode": 1,
            "autoContinue": True,
            "command": 16 if idx > 0 else 22,  # 16: WAYPOINT, 22: TAKEOFF
            "doJumpId": idx + 1,
            "frame": 3,
            "params": [0, 0, 0, None, lat, lon, alt],
            "type": "SimpleItem"
        }
        plano["mission"]["items"].append(item)

    # Salva o plano em um arquivo .plan
    with open(nome_arquivo, 'w') as arquivo:
        json.dump(plano, arquivo, indent=4)
