#hands on!!
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2.aruco as aruco
import requests
video=cv2.VideoCapture(1)


destX=100
destY=100
cdestX=100
cdestY=100
gridX=20
gridY=20
alongheight=True
startBot=0
notDecided=True
traversing=True
pushing=False
node_count=0

def traverse(vector_aruco,path,node_count,id):
	node_tbt=path[node_count]
	x=path[node_count][0]
	y=path[node_count][1]
	node_tbt[0]=((x+1/2)*630)/gridX
	node_tbt[1]=((y+1/2)*630)/gridY
	vector_node=(node_tbt[0]-centroid[0],node_tbt[1]-centroid[1])
	angle=angle_between(vector_node,vector_aruco)
	if  distance(centroid,node_tbt)<min_dist:
		requests.get(str(id)+'/s')
		node_count=node_count+1
		if node_count==len(path):
			pushing=True
			traversing=False
			node_count=0
		else node_count++
		print('stop')


	if angle<min_angle and distance(centroid,node_tbt)>min_dist:
		requests.get(str(id)+'/F')
		print('forward')


	if angle>min_angle and distance(corners[0][0][0],node_tbt)>distance(corners[0][0][1],node_tbt) and distance(centroid,node_tbt)>min_dist:
		requests.get(str(id)+'/r')
		print('right')


	if angle>min_angle and distance(corners[0][0][0],node_tbt)<=distance(corners[0][0][1],node_tbt) and distance(centroid,node_tbt)>min_dist:
		requests.get(str(id)+'/l')
		print('left')

def push_box(vector_aruco,id):
	node_tbt=[destX,destY]
	vector_node=(node_tbt[0]-centroid[0],node_tbt[1]-centroid[1])
	angle=angle_between(vector_node,vector_aruco)
	if  distance(centroid,node_tbt)<min_dist:
		if node_count==len(path):
			return True
		print('stop')


	if angle<min_angle and distance(centroid,node_tbt)>min_dist:
		requests.get(str(id)+'/F')
		print('forward')


	if angle>min_angle and distance(corners[0][0][0],node_tbt)>distance(corners[0][0][1],node_tbt) and distance(centroid,node_tbt)>min_dist:
		requests.get(str(id)+'/r')
		print('right')


	if angle>min_angle and distance(corners[0][0][0],node_tbt)<=distance(corners[0][0][1],node_tbt) and distance(centroid,node_tbt)>min_dist:
		requests.get(str(id)+'/l')
		print('left')

def shapedet(c):
	rect = cv2.minAreaRect(c)
	w=rect[1][0]
	h=rect[1][1]
	area=cv2.contourArea(c)
	peri=2*(w+h)
	ratio=(peri**2)/area
	if ratio>18:
		return 'circle'
	else:
		return 'square'



def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def distance(a,b):
	dist=(a[0]-b[0])**2+(a[1]-b[1])**2
	return dist


def Reverse(lst):
	return [ele for ele in reversed(lst)] 






import sys  
class Graph(): 
  
	def __init__(self, vertices): 
		self.V = vertices 
		self.graph = [[0 for column in range(vertices)]  
					for row in range(vertices)] 
  
	def printSolution(self, dist): 
		print ('Vertex \tDistance from Source')
		for node in range(self.V): 
			print (node, "\t", dist[node] )
  
	# A utility function to find the vertex with  
	# minimum distance value, from the set of vertices  
	# not yet included in shortest path tree 
	def minDistance(self, dist, sptSet): 
  
		# Initilaize minimum distance for next node 
		min = sys.maxsize 
  
		# Search not nearest vertex not in the  
		# shortest path tree 
	   
		for v in range(self.V): 

			if dist[v] < min and sptSet[v] == False: 
				min = dist[v]
				global min_index
				min_index=v 
  
		return min_index 
  
	# Funtion that implements Dijkstra's single source  
	# shortest path algorithm for a graph represented  
	# using adjacency_matrix representation 
	def parent(self,a):
		for i in parent:
			if i[1]==a:
				return i[0]
		
	def dijkstra(self, src,dest): 
  
		global dist
		dist = [sys.maxsize] * self.V 
		dist[src] = 0
		sptSet = [False] * self.V 
		global parent
		parent=[]
  
		for cout in range(self.V):
  
			# Pick the minimum distance vertex from  
			# the set of vertices not yet processed.  
			# u is always equal to src in first iteration 
			u = self.minDistance(dist, sptSet) 
  
			# Put the minimum distance vertex in the  
			# shotest path tree 
			sptSet[u] = True

  
			# Update dist value of the adjacent vertices  
			# of the picked vertex only if the current  
			# distance is greater than new distance and 
			# the vertex in not in the shotest path tree 
			for v in range(self.V):
				if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
					dist[v] = dist[u] + self.graph[u][v]
					parent.append((u,v))
		global path
		path=[]
		pqr=True
		end=dest
		path.append(dest)
		while(pqr):
			path.append(self.parent(end))
			end=self.parent(end)
			if end==src:
				pqr=False
		path.reverse()
		# self.printSolution(dist)

		def dfs(ti,tj,centroid,grid,pi,pj):
			if ti<0 or tj<0 or ti>=gridX or tj>=gridY or grid[ti][tj]==0:
				pass
			parent[ti][tj]=[pi,pj]
			if ti==centroid[0] and tj==centroid[1]:
				pass
			dfs(ti+1,tj,centroid,grid,ti,tj)
			dfs(ti-1,tj,centroid,grid,ti,tj)
			dfs(ti,tj+1,centroid,grid,ti,tj)
			dfs(ti,tj-1,centroid,grid,ti,tj)

while(1):
	_,ooimg=video.read()
	# oimg=ooimg[10:452,130:583]
	img=cv2.resize(oimg,(630,630))
	img_1=img.copy()
	img_2=img.copy()
	aruco_dict=aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters=aruco.DetectorParameters_create()


	min_angle=0.19
	# slow_dist=11500
	min_dist=200

	

	if(nDetection):

		hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


		lower_yellow=np.array([20,37,165])
		upper_yellow=np.array([35,255,255])


		lower_black = np.array([85,47,155])
		higher_black = np.array([179,255,255])
		

		# lower_red = np.array([0,120,70])
		# upper_red = np.array([10,255,255])


		# lower_white=np.array([0,0,149])
		# higher_white=np.array([179,51,255])


		# lower_green = np.array([72,74,4])
		# upper_green = np.array([90,174,213])


		#mask_1->for red
		#mask->for yellow
		#mask3->for blue
		#mask4->for white



		kernel=np.ones((5,5),np.uint8)

		mask=cv2.inRange(hsv,lower_yellow,upper_yellow)
		mask=cv2.GaussianBlur(mask,(15,15),0)

		maskB=cv2.inRange(hsv,lower_black,upper_black)
		maskB=cv2.GaussianBlur(mask,(15,15),0)

		box=[]
		
		contours,_=cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(img_1,contours,-1,(130,255,0),3)
		#cv2.imshow('modified',mask)
		cv2.imshow('mask',mask)

		for c in contours:
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			approx=cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
			if cv2.contourArea(c)>150:
				box=[cX,cY]
				x,y,w,h = cv2.boundingRect(c)

				m1=[x+w/2,y]
				m2=[x+w,y+h/2]
				m3=[x+w/2,y+h]
				m4=[x,y+h/2]

		a1=angle_between([destX-m1[0],destY-m1[1]],[destX-m3[0],destY-m3[1]])
		a2=angle_between([destX-m2[0],destY-m2[1]],[destX-m4[0],destY-m4[1]])

		if(abs(a1)<abs(a2)):
			alongheight=False
			if(distance(m1,[destX,destY])<distance(m3,[destX,destY])):
				intital_t=m3
			else:
				initial_t=m1;
		else:
			if(distance(m2,[destX,destY])<distance(m4,[destX,destY])):
				intital_t=m4
			else:
				initial_t=m2;
	gr=[]
	for i in range(gridX):
		gr.append(0)
	grid=[]
	for i in range(gridY):
		grid.append(gr)

	visitable=[]
	for i in range(gridX):
		for j in range(gridY):
			mask2n=mask2[j*630//gridY:(j+1)*630//gridY,i*630//gridX:(i+1)*630//gridX]
			nwp = np.sum(img == 255)
			nbp = np.sum(img == 0)
			if(nwp>nbp*10):
				visitable.append([i,j])
				grid[i][j]=1
	parent=[]
	ti=(initial_t[0]//630)*gridX
	tj=(initial_t[1]//630)*gridY
	for i in range(gridY):
		parent.append(gr)

	corners,ids,_=aruco.detectMarkers(img,aruco_dict,parameters=parameters)
	bot_cent=[]

	for (c,i) in zip(corners,ids):
		centroid=[(c[0][0][0]+c[0][1][0]+c[0][2][0]+c[0][3][0])/4,(c[0][0][1]+c[0][1][1]+c[0][2][1]+c[0][3][1])/4]
		centroid=[(centroid[0]//630)*gridX,(centroid[1]//630)*gridY]
		bot_cent.append(centroid,i)

	if not_decided:
		path=[]
		dfs(ti,tj,centroid,grid,-1,-1)
		x=centroid[0]
		y=centroid[1]
		path.append([x,y])
		while not(x==ti and y==tj):
			xx=parent[x][y][0]
			yy=parent[x][y][1]
			x=xx
			y=yy
			path.append([x,y])
		not_decided=False

	if traversing:
		traverse(bot_cent[startBot][0],path,node_count,bot_cent[startBot][1])
	time=0
	if pushing:
		pushing(bot_cent[startBot][0],bot_cent[startBot][1])
		time++
		if time==5000 and distance([destX,destY],[cdestX,cdestY])<min_dist:
			pushing=False
			not_decided=True

