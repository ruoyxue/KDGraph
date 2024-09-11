import pickle 
import sys 
import math
import json 

lat_top_left = 41.0 
lon_top_left = -71.0 
min_lat = 41.0 
max_lon = -71.0 

def rowcol2latlon(row, col):
	lat = lat_top_left - row * 1.0 / 111111.0
	lon = lon_top_left + (col * 1.0 / 111111.0) / math.cos(math.radians(lat_top_left))
	return lat, lon 

f_in = sys.argv[1]
f_out = sys.argv[2] 

try:
	neighbors = pickle.load(open(f_in, "r"))
except:
	neighbors = pickle.load(open(f_in, "rb"))

nodes = []
edges = []

cc=0

nodemap = {}
edge_map = {}

for k, v in neighbors.items():
	nodemap[k] = len(nodes)
	n1 = k 
	lat,lon = rowcol2latlon(n1[1], n1[0])

	nodes.append([lat,lon])

for k, v in neighbors.items():
	n1 = k 
	for n2 in v:
		if (n1,n2) in edge_map or (n2,n1) in edge_map:
			continue
		else:
			edge_map[(n1,n2)] = True 
		edges.append([nodemap[n1], nodemap[n2]])

json.dump([nodes,edges], open(f_out, "w"), indent=2)
