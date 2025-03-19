import numpy as np
import xml.etree.ElementTree as ET
import pyproj
from shapely.geometry import Polygon, Point
from shapely.ops import transform


def parse_osm_buildings(osm_file, bounding_box=None):
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # Coleta nós (nodes)
    nodes = {}
    for node in root.findall("node"):
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        if bounding_box:
            min_lon, min_lat, max_lon, max_lat = bounding_box
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


# Código principal
osm_file = "seu_arquivo.osm"

bounding_box = (-73.9920, 40.7500, -73.9900, 40.7520)
buildings = parse_osm_buildings(osm_file, bounding_box=bounding_box)

wgs84 = pyproj.CRS("EPSG:4326")
utm_zone = pyproj.CRS("EPSG:32618")
transformer = pyproj.Transformer.from_crs(wgs84, utm_zone, always_xy=True)
projected_buildings = get_projected_buildings(buildings, transformer)
x_range, y_range, z_range, min_x, min_y, max_height = get_grid_bounds_and_resolution(
    projected_buildings, grid_resolution=1.0
)


labirinto = create_labirinto_array(
    projected_buildings, x_range, y_range, z_range, min_x, min_y, grid_resolution=1.0
)

# Define posições inicial e objetivo
start_ix, start_iy, start_iz = 0, 0, 0
goal_ix, goal_iy, goal_iz = (
    labirinto.shape[0] - 1,
    labirinto.shape[1] - 1,
    labirinto.shape[2] - 1,
)

# Garante que as posições são válidas
if labirinto[start_ix, start_iy, start_iz] == 1:
    labirinto[start_ix, start_iy, start_iz] = 1
else:
    print("Ponto inicial não está em espaço livre.")

if labirinto[goal_ix, goal_iy, goal_iz] == 1:
    labirinto[goal_ix, goal_iy, goal_iz] = 2
else:
    print("Ponto objetivo não está em espaço livre.")

posicao_inicial = (start_ix, start_iy, start_iz)
