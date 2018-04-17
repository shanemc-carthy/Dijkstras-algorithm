#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Name:           MIS40550_Assgnmnt_01_V1
Author:         Shane McCarthy (14200512)
Created:        12/3/2017
Email:          shane.mc-carthy@ucdconnect.ie
Purpose:        This script forms the MIS40550 Assignment 01 submission 
                

Sections:       (1) Declare functions 
                (2) Test created functions against networkx
                (3) Run-time behaviour of dijkstra vs bidirectional dijkstra
                (4) Create explanatory plots for write up
    
"""

#import libs in order of use
import os 
import numpy as np
import networkx as nx
import math
from collections import Counter
import itertools
from heapq import heapify, heappush, heappop
import time
import random
from random import choice
import matplotlib.pyplot as plt   


"""
Priority queue implementation by James McDermott
Sourced from https://elearning.ucd.ie/bbcswebdav/pid-1361101-dt-content-rid-6058286_1/xid-6058286_1 

"""
class PriorityQueue:

    # A priority queue implementation based on standard library heapq
    # module. Taken from https://docs.python.org/2/library/heapq.html, but
    # encapsulated in a class. Also iterable, printable, and len-able.

    # TODO some extra capabilities that would be nice: check for empty, peek.


    REMOVED = '<removed-task>' # placeholder for a removed task

    def __init__(self, tasks_prios=None):
        self.pq = []
        self.entry_finder = {} # mapping of tasks to entries
        self.counter = itertools.count() # unique sequence count -- tie-breaker when prios equal
        if tasks_prios:
            for task, prio in tasks_prios:
                self.add_task(task, prio) # would be nice to use heapify here instead

    def __iter__(self):
        return ((task, prio) for (prio, count, task) in self.pq if task is not self.REMOVED)

    def __len__(self):
        return len(list(self.__iter__()))

    def __str__(self):
        return str(list(self.__iter__()))

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority # NB a change from the original: we return prio as well
        raise KeyError('pop from an empty priority queue')


"""
Dijkstra priority queue implementation 
This was adapted from the version posted by James McDermott here: 
https://elearning.ucd.ie/bbcswebdav/pid-1361101-dt-content-rid-6091483_1/xid-6091483_1

"""
        
        
def dijkstra_predecessors_and_distances(G, r):
    A, p, D = dijkstra(G, r)
    return p, D

    
def dijkstra_path_source_target(G,source,target):
    #Retun the path for soure and target, use this for benchmarking against 
    #the bidirectional version
    def get_sp(pointer,source,target):
                path=[source]
                x=source
                while target not in path :
                    x =pointer[x]
                    path.append(x)
                return path
    A, p, D = dijkstra(G, target)
    SP = get_sp(p,source,target)
    return SP

def dijkstra_all_paths(G,target):
    #Retun all shortest paths
    def get_all_sp(pointer,target):
      all_paths=[]
      for v in pointer:
               path=[v]
               x=v
               while target not in path :
                   x =pointer[x]
                   path.append(x)
               all_paths.append(path)
      return all_paths
    A, p, D = dijkstra(G, target)
    paths=get_all_sp(p,target)
      
    return D, paths

def dijkstra(G, r):

    if not G.has_node(r):
        raise ValueError("Source node " + str(r) + " not present")

    for e1, e2, d in G.edges(data=True):
        if d["weight"] < 0:
            raise ValueError("Negative weight on edge " + str(e1) + "-" + str(e2))

    P = {r} # permanent set
    S = PriorityQueue() # V-P. This is the crucial data structure.
    D = {} # estimates of SPST lengths
    p = {} # parent-pointers
    A = set() # our result, an SPST

    for n in G.nodes():
        if n == r:
            D[n] = 0
        else:
            if G.has_edge(r, n):
                D[n] = G[r][n]["weight"]
            else:
                D[n] = math.inf

            p[n] = r
            S.add_task(n, D[n])

    while len(S):
        u, Du = S.pop_task()
        if u in P: continue

        P.add(u) # move one item to the permanent set P and to SPST A
        A.add((p[u], u))

        for v, Dv in S:
            if v in P: continue
            if G.has_edge(u, v):
                if D[v] > D[u] + G[u][v]["weight"]:
                    D[v] = D[u] + G[u][v]["weight"]
                    p[v] = u
                    S.add_task(v, D[v]) # add v, or update its prio

    return A, p, D # let's return the SPST, predecessors and distances: user can decide which to use
    
    
    
    
    
"""
Bidirectional Dijkstra implementation 
Follows a divide-and-conquer frontier search approach, 
alternating between the forward frontier (traversing from the source s) 
and the backward frontier (traversing backwards from the target t) iteratively. 

Parts adapted from the version posted by James McDermott here: 
https://elearning.ucd.ie/bbcswebdav/pid-1361101-dt-content-rid-6091483_1/xid-6091483_1
"""
def bidirectional_dijkstra(G, source, target):
        
        #used to get link node w
        def intersect(a,b):
            return tuple(set(a) & set(b))   
            
        #used to find path using w and pointers
        def get_path(pointer,link,v):
            path=[link]
            x=link
            while v not in path :
                x =pointer[x]
                path.append(x)
            return path
            
        #used to remove duplicate w from path
        def dedupe_link_node(path):
            seen = set()
            seen_add = seen.add
            return [x for x in path if not (x in seen or seen_add(x))]
             
        #error handling 
        if not G.has_node(source): 
            raise ValueError("Source node " + str(source) + " not present in graph")
            
        if not G.has_node(target): 
            raise ValueError("Target node " + str(target) + " not present")
            
        if source==target: 
            return "Source and Target are equal, Route not Possible"

        for e1, e2, d in G.edges(data=True):
            if d["weight"] < 0:
                raise ValueError("Negative weight on edge " + str(e1) + "-" + str(e2))

        Vf = {source} # permanent set
        Vb = {target} # permanent set
        Df = {} # estimates of SPST lengths
        Db = {} # estimates of SPST lengths
        pf = {} # parent-pointers
        pb= {}
        SPf = set() # our result, an SPST
        SPb = set() # our result, an SPST
        Sbar = PriorityQueue() #Sbar is the priority queue
        Tbar = PriorityQueue() #Tbar is the version of Sbar in reverse dijkstra
        
        for n in G.nodes():
                if n == source:
                    Df[n] = 0
                else:
                    if G.has_edge(source, n):
                        Df[n] = G[source][n]["weight"]
                    else:
                        Df[n] = math.inf
        
                    pf[n] = source
                    Sbar.add_task(n, Df[n])
    
         
        for n in G.nodes():
                if n == target:
                    Db[n] = 0
                else:
                    if G.has_edge(target, n):
                        Db[n] = G[target][n]["weight"]
                    else:
                        Db[n] = math.inf
        
                    pb[n] = target
                    Tbar.add_task(n, Db[n])
  
        
        while len(intersect(Vf,Vb))==0:
            s, Ds = Sbar.pop_task()
            if s in Vf: continue
            Vf.add(s) # move one item to the permanent set P and to SPST A
            #SPf.add((pf[s], s))
            SPf.add(s)
            for v, Dv in Sbar:
                if v in Vf: continue
                if G.has_edge(s, v):
                    if Df[v] > Df[s] + G[s][v]["weight"]:
                        Df[v] = Df[s] + G[s][v]["weight"]
                        Sbar.add_task(v, Df[v]) # add v, or update its prio
                        pf[v] = s
     
            # find the min d[i] in Sbar and get i
            t, dt = Tbar.pop_task()
            if t in Vb: continue
        
            Vb.add(t) # move one item to the permanent set P and to SPST A
            #SPb.add((pb[t], t))
            SPb.add(t)
            
            for w, Dw in Tbar:
                if w in Vb: continue
                if G.has_edge(t, w):
                    if Db[w] > Db[t] + G[t][w]["weight"]:
                        Db[w] = Db[t] + G[t][w]["weight"]
                        Tbar.add_task(w, Db[w]) # add v, or update its prio
                        pb[w] = t
            
        link =intersect(Vf,Vb)
        final_link=link[0]
        final_distance =Df[link[0]]+Db[link[0]]
        
        if final_distance==float('Inf'):
            return "Route not Possible"
            
        if w in Df and w in Db:
        #see if this path is better than than the already
        #discovered shortest path
            totaldist = Counter(Df) + Counter(Db)
        if totaldist[min(totaldist)] < final_distance:# OR totaldist=[]
           link_sp =min(totaldist) 
           final_distance=totaldist[min(totaldist)] 
           final_link=link_sp
        
        path=dedupe_link_node(list(reversed(get_path(pf,final_link,source)))+get_path(pb,final_link,target))
        
        return final_distance,path
            
        

        
"""
Section 2: Testing, compare functions againts network x implemenattion
    
"""       

#Function for building random graphs
def ER_random_graph(n,p):    
#random number generator
    def rand(): 
    	random_int = random.randint(0,1000)
    	f_rand = float(random_int)
    	d_rand = f_rand * 0.001
    	return d_rand

# build graph
    g = nx.Graph()
    
    for x in range (0, n):
    	g.add_node(x)
    
    for x in range (0, n):
    	for y in range (x, n):
    		if (rand() < p or p == 1.0) & (x != y):
                     g.add_edge(x, y, weight=rand())
    return g
    
  
netx=[]
diy=[]
#create random ER graphs ranging from 100 nodes to 1000 nodes 
for i in range(10,1000,100):
    G=ER_random_graph(i,.2)
    #random pick two nodes from the random graph as the source/target
    source=choice(G.nodes())
    target=choice(G.nodes())
    netx.append(bidirectional_dijkstra(G, source,target)) 
    diy.append(nx.bidirectional_dijkstra(G, source,target))  
    compare=[netx,diy]
    
# Results match networkx, move on... 

"""
Section 3: Scaling compare bidirectional dijkstra vs orginal dijkstra

""" 

t_avg_dij=[]
t_avg_bidij=[]
index=[]
#create random ER graphs ranging from 100 nodes to 1000 nodes 
for i in range(100,1500,100):
    #Get the average of each graph 
    for r in range(1,5):
        print(i)
        #random pick two nodes from the random graph as the source/target
        G=ER_random_graph(i,.2)
        source=choice(G.nodes())
        target=choice(G.nodes())
        
        bidij=[]
        start1 = time.time()
        bidirectional_dijkstra(G, source,target)      
        end1 = time.time()
        bidij.append(end1-start1)
    
        dij=[]
        start2 = time.time()
        dijkstra_path_source_target(G, source,target)      
        end2 = time.time()
        dij.append(end2-start2)
    index.append(i)
    t_avg_bidij.append(sum(bidij)/len(bidij))
    t_avg_dij.append(sum(dij)/len(dij))
    comp=np.column_stack((index,t_avg_bidij,t_avg_dij))


 fig, ax = plt.subplots(figsize=(12, 4))
 #ax.plot(range(0,3,.5),index, '.-')
 ax.plot(comp[:,0],comp[:,2],linewidth=2, label='Orginal Dijkstra')
 ax.plot(comp[:,0],comp[:,1],linewidth=2, label='Bidirectional Dijkstra')
 ax.grid(True)
 ax.legend(loc='left')
 ax.set_title('Time Complextity: Bidirectional Dijkstra vs Orginal Dijkstra')
 ax.set_xlabel('Number of Nodes')
 ax.set_ylabel('Time (Seconds)')
 ax.set_xticks(np.arange(0, 1500, 100))
 ax.grid(which='major', alpha=0.5) 
 plt.show()
    
  
    
"""
Section 4: Plot Graphs
    
""" 
 
from random import choice
 
X=ER_random_graph(40,.2)
source=choice(X.nodes())
target=choice(X.nodes())
final_distance,path,link,final_link,Vf,Vb =bidirectional_dijkstra(X,target,source)

for e in X.edges():
    X[e[0]][e[1]]['color'] = 'black'
    X[e[0]][e[1]]['alpha'] = .2
    
# Set color of edges of the shortest path to red

for i in range(len(path)-1):
    X[path[i]][path[i+1]]['color'] = 'red'
    X[path[i]][path[i+1]]['alpha'] = 2
    

# Store in a list to use for drawing
edge_color_list = [ X[e[0]][e[1]]['color'] for e in X.edges() ]
edge_weight_list = [ X[e[0]][e[1]]['alpha'] for e in X.edges() ]

plt.figure(3,figsize=(14,8)) 
position = nx.spring_layout(X)

nx.draw_networkx_labels(X, position, labels=None, font_size=10, font_color='w', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
nx.draw(X,position,node_color='k',edge_color = edge_color_list,width=edge_weight_list)

nx.draw_networkx_nodes(X,position, nodelist=Vf, node_color="b",with_labels = True)
nx.draw_networkx_nodes(X,position, nodelist=Vb, node_color="g",with_labels = True)
nx.draw_networkx_nodes(X,position, nodelist=link, node_color="y",with_labels = True)
nx.draw_networkx_nodes(X,position, nodelist=[final_link], node_color="r",with_labels = True)
nx.draw_networkx_nodes(X,position, nodelist=[source], node_color="grey",with_labels = True)
nx.draw_networkx_nodes(X,position, nodelist=[target], node_color="grey",with_labels = True)
plt.savefig("graph4.png", dpi=500, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
plt.show()
print(link,final_link)







