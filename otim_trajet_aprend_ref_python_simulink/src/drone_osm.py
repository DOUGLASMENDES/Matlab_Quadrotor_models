import numpy as np
import xml.etree.ElementTree as ET
import pyproj
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import matplotlib

matplotlib.use("TkAgg")  # para geração não-interativa: usar 'Agg'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime

from utils import *


log("=" * 80)
log("Iniciando busca de melhor trajetória")
log("Convertendo o arquivo .osm para uma matriz simbólica...")
edificacoes = parse_edificacoes_osm(arquivo_osm, coord_limites=coord_limites)


wgs84 = obter_wsg84()
zona_global = obter_zona_global()

# criamos um objeto transformer para poder reusar transformações entre coordenadas
# sem a necessidade de recriá-las
transformer = pyproj.Transformer.from_crs(wgs84, zona_global, always_xy=True)

edificacoes_projetadas = obtem_edificacoes_projetadas(edificacoes, transformer)

x_range, y_range, z_range, min_x, min_y, max_height = obtem_limites_e_resolucoes_grade(
    edificacoes_projetadas, resolucao_grade=resolucao_grade
)

# Define o ambiente do labirinto (grade 3D)
# 1: caminho livre, 0: obstáculo, 2: objetivo
# Usa como origem o arquivo .osm para criar a matriz 3d:
labirinto = gera_matriz_labirinto(
    edificacoes_projetadas, x_range, y_range, z_range, min_x, min_y, resolucao_grade=1.0
)
log("Matriz simbólica obtida a partir do arquivo .osm!")

check_inicio_dest_validos(labirinto)

# Obter coordenadas georeferenciadas da posição inicial:
lat_ini, long_ini, alt_ini = index_para_coord_geografica(
    posicao_inicial, min_x, min_y, resolucao_grade, transformer
)

# Coordenadas da posição inicial do drone:
texto_pos = "Posição geográfica inicial do drone : "
log(f"{texto_pos}Latitude={lat_ini}, Longitude={long_ini}, Altitude={alt_ini}")

# Define a tabela Q
# Configuração do Q-Learning: Inicializa-se a tabela Q com zeros,
# define-se as ações e parâmetros de aprendizado
tabela_q = np.zeros((*labirinto.shape, len(comandos)))


log("Iniciando treino do drone (agente)...")
treinar_ql(max_rodadas, posicao_inicial, labirinto, tabela_q)

log(f"FIM DO TREINO, hora: {datetime.datetime.now()}")
log("Treino do drone (agente) concluído...")

rodada_menor_custo = obtem_rodada_menor_custo(todas_rodadas)

log(f"Rodada com menor custo: {rodada_menor_custo.rodada}")
log("Comandos:")
coms = obtem_comandos(rodada_menor_custo.caminho)
loga_comandos(coms)

log("Caminho:")
log(rodada_menor_custo.caminho)

# como o número de comandos é muito grande, não vamos
# plotar o labirinto (comentado)
# plot_labirinto(rodada_menor_custo)

log("Melhor trejetória obtida!")
