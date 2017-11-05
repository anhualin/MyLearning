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
V[0] = [[1, 6, 7], []]
V[1] = [[0, 2, 6, 7], []]
V[2] = [[1, 3, 4, 5], []]
V[3] = [[2, 4, 5], []]
V[4] = [[2, 3, 5], []]
V[5] = [[2, 3, 4, 6], []]
V[6] = [[0, 1, 5, 7], []]
V[7] = [[0, 1, 6], []]

E = []
for v in V:
    neighbors = [(v, w) for w in V[v][0] if w > v]
    E += neighbors

def graph_contraction(V, E):
    print('here')
    E_backup = copy.copy(E)
    V_backup = copy.copy(V)
    for i in range(len(V) - 2):
        e = E[0]
        V, E = contract_edge(V, E, e)
    return V, E
        
def contract_edge(V, E, e):
    u = e[0]
    v = e[1]
    E = [edge for edge in E if edge != e]
    V[u][0] = [x for x in V[u][0] if x != v] + [x for x in V[v][0] if x != u]
    V[u][1].append(v)
    V[u][1] += V[v][1]
    for w in V[v][0]:
        if w != u:
            V[w][0].remove(v)
            V[w][0].append(u)
            if v < w:
                E.remove((v, w))
            else:
                E.remove((w, v))
            if w < u:
                E.append((w, u))
            else:
                E.append((u, w))
                
    del V[v]
    return V, E

V1, E1 = contract_edge(V, E, (1, 6))
V2, E2 = contract_edge(V1, E1, (1, 7))
V3, E3 = contract_edge(V2, E2, (2, 5))

Vx, Ex = graph_contraction(V, E)