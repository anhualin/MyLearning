# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:21:51 2017

@author: alin
"""

import copy
import random
a = set([1,2,3,4,5])
b = random.sample(a,1)[0]

V = {}
V[0] = ([1, 6, 7], [])
V[1] = ([0, 2, 6, 7], [])
V[2] = ([1, 3, 4, 5], [])
V[3] = ([2, 4, 5], [])
V[4] = ([2, 3, 5], [])
V[5] = ([2, 3, 4, 6], [])
V[6] = ([0, 1, 5, 7], [])
V[7] = ([0, 1, 6], [])

E = []
for v in V:
    neighbors = [(v, w) for w in V[v][0] if w > v]
    E += neighbors

def graph_contraction(V, E):
    E_backup = copy.copy(E)
    V_backup = copy.copy(V)
    for i in range(len(V) - 2):
        e = E[0]
        contract_edge(V, E, e)
        
def contract_edge(V, E, e):
    u = e[0]
    v = e[1]
    E = [edge for edge in E if edge != e]
    V[u][0] = [x for x in V[u][0] if x != v]
    V[u][1].append(v)
    V[u][1] += V[v][1]
    for w in V[v][0]:
        if w != u:
            if v < w:
                E.remove((v, w))
            else:
                E.remove((w, v))
            V[u][0].append(w)
            if w < u:
                E.append((w, u))
            else:
                E.append((u, w))
                
            
        