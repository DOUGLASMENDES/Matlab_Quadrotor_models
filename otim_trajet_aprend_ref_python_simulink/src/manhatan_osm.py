import osmnx as ox
import matplotlib.pyplot as plt

# Carregar os dados de Manhattan
G = ox.graph_from_place("Manhattan, New York, USA", network_type="all")

# Plotar o mapa para visualização inicial
fig, ax = ox.plot_graph(G, figsize=(10, 10), node_size=0, edge_linewidth=0.5)

print('fim')