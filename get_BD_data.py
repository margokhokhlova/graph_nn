# this script is created to load the data from BD topo to display them alltogether with the graphs
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, MultiPoint, Polygon, shape
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.linestring import LineString
from shapely.geometry.collection import  GeometryCollection
from shapely.ops import triangulate
from shapely.affinity import translate
import pandas as pd
import numpy as np
from shapely import ops
import pickle
from earthpy import clip as cl


def load_3_departments_for_category(data_folder, category="/A_RESEAU_ROUTIER/ROUTE.shp"):
    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
    data = gpd.read_file('/' + subfolders[0] + category)
    for dep in subfolders[1:]:
        fp_category = data_folder + '/' + dep + category
        data.append(gpd.read_file(fp_category))
    return data


def load_bd(year, data_folder):
    if year == 2008:
    path = data_folder + '/' + str(year)
    roads = load_3_departments_for_category(path, "/A_RESEAU_ROUTIER/ROUTE.shp")
    buildings_i = load_3_departments_for_category(path, "E_BATI/BATI_INDIFFERENCIE.shp")
    buildings_n = load_3_departments_for_category(path, "E_BATI/BATI_INDUSTRIEL.shp")
    all_buildings = pd.concat([buildings_i, buildings_n], ignore_index=True)
    rivers = load_3_departments_for_category(path, "D_HYDROGRAPHIE/TRONCON_COURS_EAU.shp")
    build_remk = load_3_departments(path, "E_BATI/BATI_REMARQUABLE.shp")
    railroads = load_3_departments(path, "B_VOIES_FERREES_ET_AUTRES/TRONCON_VOIE_FERREE.shp")
    return Bunch(roads=roads, buildings=all_buildings, rivers=rivers, build_reml=build_remk, rail=railroads);

    if year== 2010:
    path = data_folder + '/' + str(year)
    roads = load_3_departments_for_category(path, "/A_RESEAU_ROUTIER/ROUTE.shp")
    buildings_i = load_3_departments_for_category(path, "E_BATI/BATI_INDIFFERENCIE.shp")
    buildings_n = load_3_departments_for_category(path, "E_BATI/BATI_INDUSTRIEL.shp")
    all_buildings = pd.concat([buildings_i, buildings_n], ignore_index=True)
    rivers = load_3_departments_for_category(path, "D_HYDROGRAPHIE/TRONCON_COURS_EAU.shp")
    build_remk = load_3_departments(path, "E_BATI/BATI_REMARQUABLE.shp")
    railroads = load_3_departments(path, "B_VOIES_FERREES_ET_AUTRES/TRONCON_VOIE_FERREE.shp")
    return Bunch(roads=roads, buildings=all_buildings, rivers=rivers, build_reml=build_remk, rail=railroads);

    if year==2014:
        path = data_folder + '/' + str(year)
        roads = load_3_departments_for_category(path, "/A_RESEAU_ROUTIER/ROUTE.shp")
        buildings_i = load_3_departments_for_category(path, "E_BATI/BATI_INDIFFERENCIE.shp")
        buildings_n = load_3_departments_for_category(path, "E_BATI/BATI_INDUSTRIEL.shp")
        all_buildings = pd.concat([buildings_i, buildings_n], ignore_index=True)
        rivers = load_3_departments_for_category(path, "D_HYDROGRAPHIE/TRONCON_COURS_EAU.shp")
        build_remk = load_3_departments(path, "E_BATI/BATI_REMARQUABLE.shp")
        railroads = load_3_departments(path, "B_VOIES_FERREES_ET_AUTRES/TRONCON_VOIE_FERREE.shp")
        return Bunch(roads=roads, buildings=all_buildings, rivers=rivers, build_reml=build_remk, rail=railroads);
    if year == 2019:
        path = data_folder + '/' + str(year)
        roads = load_3_departments_for_category(path, "/TRANSPORT/TRONCON_DE_ROUTE.shp")
        all_buildings = load_3_departments(path, "BATI/BATI.shp")
        rivers = load_3_departments_for_category(path, "HYDROGRAPHIE/COURS_D_EAU.shp")
        build_remk = all_buildings.loc[(all_buildings['NATURE'] == "Eglise") | (all_buildings['NATURE'] == "Chapelle") |
                                       (all_buildings['NATURE'] == "Tour, donjon") | (all_buildings['NATURE'] == "Monument") |
                                       (all_buildings['NATURE'] == 'Fort, blockhaus, casemate') | (
                                                   all_buildings['NATURE'] == 'ChÃ¢teau') |
                                       (all_buildings['NATURE'] == 'Arc de triomphe')]
        buildings = all_buildings.loc[
            (all_buildings['NATURE'] == 'Indifférenciée') | (all_buildings['NATURE'] == "Industriel, agricole ou commercial") |
            (all_buildings['NATURE'] == 'Serre') | (all_buildings['NATURE'] == 'Serre')]
        railroads = load_3_departments(path, "TRANSPORT/TRONCON_DE_VOIE_FERREE.shp")
        return Bunch(roads=roads, buildings=buildings, rivers=rivers, build_reml=build_remk, rail=railroads);
    default:
    return "get the correct year";
    };
