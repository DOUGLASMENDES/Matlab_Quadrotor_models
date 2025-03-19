import numpy as np
import xml.etree.ElementTree as ET
import pyproj
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import random
import matplotlib

matplotlib.use("TkAgg")  # para geração não-interativa: usar 'Agg'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime

from utils import *

log("=" * 80)
log("Iniciando conversão de rota para arquivo QGroundControl")
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


# abre o arquivo log_comandos.txt:
with open(arquivo_log_comandos, 'r') as arquivo:
    linhas = arquivo.readlines()
    p_line = lambda s: s.replace('\n','').replace('(','').replace(')','')
    comandos_relatorio = [tuple(p_line(c).split(',')) for c in linhas]
    comandos_relatorio = [(int(x), int(y), int(z)) for x, y, z in comandos_relatorio]     

comandos_discr = discretizar_comandos(comandos_relatorio)

log("="*80)
log("Discretizando comandos:")
log(f"Número de comandos: {len(comandos_discr)}")
log(comandos_discr)
log("="*80)

log("Listando coordenadas das posições geográficas da melhor rota:")



posicoes = gera_lista_posicoes(posicao_inicial, comandos_discr, labirinto)
log("Posições na grade:")
log(posicoes)
log(' ')

posicoes_geog = posicoes_grade_para_geograficas(posicoes, 
                                                min_x, 
                                                min_y, 
                                                resolucao_grade, 
                                                transformer)

log("Posições georeferenciadas:")
log(posicoes_geog)
log(' ')


# Arquivo OSM a gravar a rota:
#arquivo_saida = "map_with_route2.osm"

#adicionar_rota_no_osm_ajustada(arquivo_osm, posicoes_geog, arquivo_saida)
#log(f"Rota adicionada e salva em {arquivo_saida}")

print("Gerando arquivo do QGroundControl...")
converter_para_qgc_plan(posicoes_geog, 'melhor_rota.plan')
print("Arquivo mission QGroundControl gerado!")


log("="*80)



