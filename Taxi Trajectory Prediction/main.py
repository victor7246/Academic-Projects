## this code is based on a single day. predicting taxi travel time for each road segment at any point of time of day and then predicting the shortest path between any two points at any time of day. in this code, we only use mean prediction and regression based prediction. In the project we also implemented prediction based on decision tree and randomforest.

import pandas
import matplotlib as ma
import gmplot
from colour import Color
import pygmaps 
import matplotlib.pyplot as plt
import numpy as np
import scipy
import random
import time
import networkx as nx
import plotly
import datetime
import json
import itertools
import sklearn
import sklearn.cluster
import scipy.spatial
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from decimal import *
import sys

# to calculate haversine distance between two points
def distance1(a,b):
	lon_diff = np.abs(a[0]-b[0])*np.pi/360
	lat_diff = np.abs(a[1]-b[1])*np.pi/360
	a1 = np.sin(lat_diff)**2 + np.cos(a[1]*np.pi/180.0) * np.cos(b[1]*np.pi/180.0) * np.sin(lon_diff)**2
	d = 2*6371*np.arctan2(np.sqrt(a1),np.sqrt(1-a1)) 
	return round(d,3) 

# minimum distance between two list of points
def distance2(a,b):
	return min([distance1(i,b) for i in a])

# to calculate minimum distance index
def nearest(a,b):
	return [distance1(i,b) for i in a].index(distance2(a,b))

def getkey(i):
		return i[0]


if __name__ == "__main__":


	current_time = input('Enter the current time in second : ')
	point1 = input('Enter the initial point : ')
	point2 = input('Enter the destination : ')
	coeff1 = input('Enter the weight of time in the route prediction : ')
	coeff2 = input('Enter the weight of distance in the route prediction : ')
	
	main_data = pandas.read_csv("../data/main_data_day.csv")
	columns = ['POLYLINE','DATE','MONTH','DAY','TIME','MAP']
	main_data = main_data[columns]
	main_data['POLYLINE'] = main_data['POLYLINE'].apply(json.loads)
	main_data['MAP'] = main_data['MAP'].apply(json.loads)

	# we created the standard points by taking centres of 100 meter X 100 meter squares over the whole map of the city
	standard_points = pandas.read_csv("../data/standard_points.csv")
	standard_points = [[round(standard_points["0"][i],6),round(standard_points["1"][i],6)] for i in range(len(standard_points))]

	all_location = []
	for i in range(len(main_data)):
		all_location += main_data.iloc[i].POLYLINE
	all_location = [list(x) for x in set(tuple(x) for x in all_location)]

	lat = [i[0] for i in all_location]
	lon = [i[1] for i in all_location]

	int1 = int(distance1([max(lat),0],[min(lat),0])/.1)+1
	int2 = int(distance1([0,max(lon)],[0,min(lon)])/.1)+1

	array1 = np.linspace(min(lat),max(lat),int1)
	array2 = np.linspace(min(lon),max(lon),int2)	

	def position(i):
		num1,num2 = 0,0
		for j in range(len(array1)-1):
			if array1[j] <= i[0] and array1[j+1] >= i[0]:
				num1 = j
		for j in range(len(array2)-1):
			if array2[j] <= i[1] and array2[j+1] >= i[1]:
				num2 = j
		return len(array1)*num2 + num1

	standard_points_dict = {i:standard_points[i] for i in range(len(array1)*len(array2))}

	if standard_points_dict[position(point1)] == [0,0] or standard_points_dict[position(point2)] == [0,0]:
		print "hard luck bro"
		sys.exit()

	#v = cKDTree(standard_points)
	


	# after getting the standard points we create road segments between standard points if the distance between them is less than 250 meters and also keep the time taken information at the time of day to travel that road segment - from the main dataset
	roads = []
	for i in range(0,len(main_data)):
		xx = main_data.MAP[i]
		for j in range(0,len(xx)-1):
			if xx[j] != xx[j+1]:
				if distance1(standard_points_dict[xx[j]],standard_points_dict[xx[j+1]]) <= .25:
					time_taken = (xx.index(xx [j+1]) - xx.index(xx[j]))*15
					start_time = main_data.TIME[i] + xx.index(xx[j])*15
					roads.append([(xx[j],xx[j+1]),main_data.DAY.iloc[i],main_data.MONTH.iloc[i],start_time,abs(time_taken)])

	

	roads = [list(x) for x in set(tuple(x) for x in roads)]
	roads = sorted(roads,key=getkey)

	groups_roads = []
	uniquekeys = []

	for k, g in itertools.groupby(roads, getkey):
		groups_roads.append(list(g))      
		uniquekeys.append(k)	

	def getkey2(i):
		return len(i)

	#freq_road = sorted(groups_roads,key=getkey2)[len(groups_roads)-1]
	#print freq_road[0][0]	
     	

	# simpler model to predict time taken for each road segment by just taking the mean and create graph based on the prediction
	weights1 = []
	for i in groups_roads:
		wt = np.mean([j[4] for j in i])
		weights1 += [[i[0][0],wt]]
		weights1 += [[(i[0][0][1],i[0][0][0]),wt]]

	G = nx.Graph()
	for i in weights1:
		if i[0][0] != i[0][1]:
			G.add_edge(i[0][0],i[0][1],weight=coeff1*i[1]+coeff2*distance1(standard_points_dict[i[0][0]],standard_points_dict[i[0][1]]))



	# regression with decision tree to predict time taken for each road segment by just taking the mean and create graph based on the prediction
	weights3 = []
	from sklearn import tree
	for i in groups_roads:
		x = [[j[3]] for j in i]
		y = [j[4] for j in i]
		clf = tree.DecisionTreeRegressor()
		clf = clf.fit(x,y)
		weights3 += [[i[0][0],clf]]
		weights3 += [[(i[0][0][1],i[0][0][0]),clf]]




	# predictions to go from one node to another using simple mean time

	'''
	dist,node1 = v.query(point1,k=1)
	dist,node2 = v.query(point2,k=1)
	'''
	# now to get the shortest route we use dijstra on the graph
	node1 = position(point1)
	node2 = position(point2)

	path1 = nx.dijkstra_path(G,node1,node2)		
	temp = 0
	for i in range(0,len(path1)-1):
		temp += distance1(standard_points_dict[path1[i]],standard_points_dict[path1[i+1]])
	path1_length = (nx.dijkstra_path_length(G,node1,node2)-coeff2*temp)/coeff1
	print path1
	print path1_length

	map1 = pygmaps.maps(41.1496100,-8.6109900,12)
	for i in path1:
		map1.addpoint(standard_points_dict[i][0],standard_points_dict[i][1],"#FF0000")
	map1.draw("../map1.html")		

	
	# prediction using decision tree
	
	weights3_prediction = [[i[0],float(i[1].predict([current_time]))] for i in weights3]
	G3 = nx.Graph()
	for i in weights3_prediction:
		G3.add_edge(i[0][0],i[0][1],weight=coeff1*i[1]+coeff2*distance1(standard_points_dict[i[0][0]],standard_points_dict[i[0][1]]))

	path3 = nx.dijkstra_path(G3,node1,node2)		
	temp = 0
	for i in range(0,len(path3)-1):
		temp += distance1(standard_points_dict[path3[i]],standard_points_dict[path3[i+1]])
	path3_length = (nx.dijkstra_path_length(G3,node1,node2)-coeff2*temp)/coeff1
	print path3
	print path3_length

	map3 = pygmaps.maps(41.1496100,-8.6109900,12)
	for i in path3:
		map3.addpoint(standard_points_dict[i][0],standard_points_dict[i][1],"#FF0000")
	map3.draw("../map3.html")
'''
	d13 = nx.betweenness_centrality(G3)
	d23 = nx.edge_betweenness_centrality(G3)
	for i in d13.keys():
		if d13[i] == max(d13.values()):
			print standard_points_dict[i]
			print 'with centrality ' + str(d13[i])
			break
	for i in d23.keys():
		if d23[i] == max(d23.values()):
			print (standard_points_dict[i[0]],standard_points_dict[i[1]])
			print 'with centrality ' + str(d23[i])
			break		

	groups_roads_simulated = list(groups_roads)
	for i in range(len(groups_roads_simulated)):
    x = []
    for j in groups_roads_simulated[i]:
        for k in range(1,31):
            x += [[j[0],j[1],j[2],j[3]+k*60,j[4]]]
    groups_roads_simulated[i] += x   

    # regression model with normal equation
		weights2 = []
		for i in groups_roads_simulated:
			x1 = tuple([1 for j in i])
			x2 = tuple([j[3] for j in i])
			x3 = tuple([j[3]*j[3] for j in i])
			x4 = tuple([j[3]*j[3]*j[3] for j in i])
			#x5 = tuple([j[3]*j[3]*j[3]*j[3] for j in i])
			y1 = tuple([j[4] for j in i])
			x = np.matrix((x1,x2,x3,x4))
			y = np.matrix((y1))
			wt = ((x * x.transpose()) ** (-1))*(x * y.transpose())
			weights2 += [[i[0][0],wt]]
			weights2 += [[(i[0][0][1],i[0][0][0]),wt]]

	# prediction using 2nd method if have lots of road data
	
	weights2_prediction = [[i[0],np.matrix([1,pow(current_time,1),pow(current_time,2),pow(current_time,3)])*i[1]] for i in weights2]

	G2 = nx.Graph()
	for i in weights2_prediction:
		G2.add_edge(i[0][0],i[0][1],weight=coeff1*i[1]+coeff2*distance1(standard_points_dict[i[0][0]],standard_points_dict[i[0][1]]))

	path2 = nx.dijkstra_path(G2,node1,node2)		
	
	temp = 0
	for i in range(0,len(path2)-1):
		temp += distance1(standard_points_dict[path2[i]],standard_points_dict[path2[i+1]])
	path2_length = round((nx.dijkstra_path_length(G2,node1,node2)-coeff2*temp)/coeff1,2)
	print path2
	print path2_length

	map2 = pygmaps.maps(41.1496100,-8.6109900,12)
	for i in path2:
		map2.addpoint(standard_points_dict[i][0],standard_points_dict[i][1],"#FF0000")
	map2.draw("../map2.html")		
	
	'''		


