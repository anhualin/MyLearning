# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 08:42:07 2017

@author: alin
"""

from math import sqrt, fabs
#import time
from time import time
import sys
from heapq import heappush, heappop 
import resource
#import numpy as np

def Manhattan(state):
    # Manhattn distance from state to the target
    # state is a string
    dis = 0
    state = [int(e) for e in state.split('*')]
    N = int(sqrt(len(state)))
    for i in range(len(state)):
        if state[i] != 0:
            s = int(state[i])
            x0 = int(i/N)
            y0 = i - x0 * N
          
            x1 = int(s/N)
            y1 = s - x1 * N
#            delta = fabs(x0 - x1) + fabs(y0 - y1)
#            print "i=%s, x0 = %s, y0 = %s, s = %s, x1 = %s, y1=%s, delta = %s" % (i, x0, y0, s, x1, y1, delta)
            dis  = dis + fabs(x0 - x1) + fabs(y0 - y1)
    return dis


class Npuzzle(object):
    def __init__(self, state, parent, x0, y0, move, depth, g):
        self.state = state
        self.parent = parent
        self.x0 = x0
        self.y0 = y0
        self.move = move # the move it takes to get here from the parent
        self.depth = depth
        self.N = int(sqrt(len(state.split('*'))))
        self.g = g
        self.f = g + Manhattan(state)
#    def display(self):
#        print np.array([int(e) for e in self.state.split('*')]).reshape(self.N, self.N)
#        print self.depth
#        print self.x0
#        print self.y0
#        print self.state
#    def setParent(self, parent):
#        self.parent = parent
    def getParent(self):
        return self.parent
    def getState(self):
        return self.state
    def getMove(self):
        return self.move
#    def setMove(self, move):
#        self.move = move
    def getDepth(self):
        return self.depth
    def setDepth(self, depth):
        self.depth = depth
    def getF(self): 
        return self.f
    def setF(self, f):
        self.f = f
    def getG(self):
        return self.g
    def setG(self, g):
        self.g = g
    def swap(self, x1, y1):
        state_lst = self.state.split('*')
        state_lst[self.x0 * self.N + self.y0] = state_lst[x1 * self.N + y1]
        state_lst[x1 * self.N + y1] = '0'
        return '*'.join(state_lst)
    def getNeighbors(self):
        neighbors = []
        if self.x0 > 0:
            state1 = self.swap(self.x0 - 1, self.y0)
            neighbors.append(Npuzzle(state1, self, self.x0 - 1, self.y0, 'Up', self.depth + 1, self.g + 1 ))
        if self.x0 < self.N - 1:
            state1 = self.swap(self.x0 + 1, self.y0)
            neighbors.append(Npuzzle(state1, self, self.x0 + 1, self.y0, 'Down', self.depth + 1, self.g + 1 ))
        if self.y0 > 0:
            state1 = self.swap(self.x0, self.y0 - 1)
            neighbors.append(Npuzzle(state1, self, self.x0, self.y0 - 1, 'Left', self.depth + 1, self.g + 1 ))
        if self.y0 < self.N - 1:
            state1 = self.swap(self.x0, self.y0 + 1)            
            neighbors.append(Npuzzle(state1, self, self.x0, self.y0 + 1, 'Right', self.depth + 1, self.g + 1 ))
        return neighbors
    def __lt__(self, other):
        return self.f < other.getF()
    
        
def retracePath(node):
    path = [node]
    while node.getParent():
        node = node.getParent()
        path.insert(0,node)
    path.pop(0)
    path_to_goal = []
    for node in path:
        path_to_goal.append(node.getMove())
    return path, path_to_goal, len(path_to_goal)


    
def AStar(start):
    start_time = time()
    target = '*'.join([str(e) for e in range(len(start))])
    start_state = '*'.join(start)
    pos = start.index('0')
    N = int(sqrt(len(start)))
    x0 = int(pos/N)
    y0 = pos - x0 * N
    root = Npuzzle(start_state, None, x0, y0, None, 0, 0) 
    iteration = 0
    Frontier = []
    heappush(Frontier, root)
    Explored = set()
    FrontierSet = {root.getState(): root.getF()}
    done = False
   
    success = False
    nodes_expanded = 0
    max_fringe_size = 1
    max_depth = root.getDepth()
    output = []
    while(not done):
        
        if Frontier:
            node = heappop(Frontier)
            empty_Frontier = False
            while not empty_Frontier and node.getState() in Explored:
                if Frontier:
                    node = heappop(Frontier)
                else:
                    empty_Frontier = True
            if empty_Frontier:
                done = True
            else:
                
                del FrontierSet[node.getState()] 
                
                if node.getState() == target:
                    done = True
                    success = True
                else:
                    nodes_expanded += 1
                  
                    Explored.add(node.getState())
                  
                    neighbors = node.getNeighbors()
                   
                    for neighbor in neighbors:
                
                        if neighbor.getState() in Explored:
                            pass
                        elif neighbor.getState() in FrontierSet:
                            # the neighbor is already in Frontier
                            # check to see whether we have a smaller f value 
                            if neighbor.getF() < FrontierSet[neighbor.getState()]:
                                heappush(Frontier, neighbor)
                                FrontierSet[neighbor.getState()] = neighbor.getF()
                                if max_depth < neighbor.getDepth():
                                    max_depth = neighbor.getDepth()
                        else:
                            #not in Frontier, add it
                            heappush(Frontier, neighbor)
                            FrontierSet[neighbor.getState()] = neighbor.getF() 
                            
                            if max_depth < neighbor.getDepth():
                                max_depth = neighbor.getDepth()
                    if max_fringe_size < len(Frontier):
                        max_fringe_size = len(Frontier)
                iteration += 1
        else:
            done = True
    if success:
        path, path_to_goal, cost_of_path = retracePath(node)
        running_time = time() - start_time
        output.append('path_to_goal: ' + str(path_to_goal))
        output.append('cost_of_path: ' + str(cost_of_path))
        output.append('nodes_expanded: ' + str(nodes_expanded))
        output.append('fringe_size: ' + str(len(Frontier))) 
        output.append('max_fringe_size: ' + str(max_fringe_size))
        output.append('search_depth: ' + str(node.getDepth()))
        output.append('max_depth: ' +  str(max_depth))
        output.append('running_time: ' + "{0:.8f}".format(running_time))
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        output.append('max_ram_usage: ' + "{0:.8f}".format(mem))
       
    return output

    

    
#state = ['7', '2', '4', '5', '0', '6', '8', '3', '1']
#state = ['1','2','5','3','4','0','6','7','8']
#output = AStar(state)
#output1 = BFS(state)

def BFS(start):
    start_time = time()
    target = '*'.join([str(e) for e in range(len(start))])
    start_state = '*'.join(start)
    pos = start.index('0')
    N = int(sqrt(len(start)))
    x0 = int(pos/N)
    y0 = pos - x0 * N
    root = Npuzzle(start_state, None, x0, y0, None, 0, 0) 
    iteration = 0
    Frontier = [root]
    Explored = set()
    FrontierSet = set([start_state])
    done = False
    node = root
    success = False
    nodes_expanded = 0
    max_fringe_size = 1
    max_depth = root.getDepth()
    output = []
    while(not done):
        if Frontier:
            node = Frontier.pop(0) 
#            if max_depth < node.getDepth():
#                max_depth = node.getDepth()
            FrontierSet.remove(node.getState())
            if node.getState() == target:
                done = True
                success = True
            else:
                nodes_expanded += 1
                Explored.add(node.getState())
                neighbors = node.getNeighbors()
                for neighbor in neighbors:
                    if neighbor.getState() in Explored or neighbor.getState() in FrontierSet:
                        pass
                    else:
                        Frontier.append(neighbor)
                        FrontierSet.add(neighbor.getState())
                        if max_depth < neighbor.getDepth():
                            max_depth = neighbor.getDepth()
                if max_fringe_size < len(Frontier):
                    max_fringe_size = len(Frontier)
            iteration += 1
        else:
            done = True
    if success:
        path, path_to_goal, cost_of_path = retracePath(node)
        running_time = time() - start_time
        output.append('path_to_goal: ' + str(path_to_goal))
        output.append('cost_of_path: ' + str(cost_of_path))
        output.append('nodes_expanded: ' + str(nodes_expanded))
        output.append('fringe_size: ' + str(len(Frontier))) 
        output.append('max_fringe_size: ' + str(max_fringe_size))
        output.append('search_depth: ' + str(node.getDepth()))
        output.append('max_depth: ' +  str(max_depth))
        output.append('running_time: ' + "{0:.8f}".format(running_time))
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        output.append('max_ram_usage: ' + "{0:.8f}".format(mem))
       
    return output


#test = ['1','2','5','3','4','0','6','7','8']
#out = BFS(test)
#test2 = ['3', '2', '1','0']
#out2 = BFS(test2)
    
def DFS(start):
    start_time = time()
    target = '*'.join([str(e) for e in range(len(start))])
    start_state = '*'.join(start)
    pos = start.index('0')
    N = int(sqrt(len(start)))
    x0 = int(pos/N)
    y0 = pos - x0 * N
    root = Npuzzle(start_state, None, x0, y0, None, 0, 0) 
    Frontier = [root]
    Explored = set()
    FrontierSet = set([start_state])
    done = False
    success = False
    nodes_expanded = 0
    max_fringe_size = 1
    max_depth = root.getDepth()
    output = []
    while(not done):
        if Frontier:
            node = Frontier.pop(0) 
            FrontierSet.remove(node.getState())
            if node.getState() == target:
                done = True
                success = True
            else:
                nodes_expanded += 1
                Explored.add(node.getState())
                neighbors = node.getNeighbors()
                for i in range(len(neighbors) - 1, -1, -1):
                    neighbor = neighbors[i]
                    if neighbor.getState() in Explored or neighbor.getState() in FrontierSet:
                        pass
                    else:
                        Frontier.insert(0, neighbor)
                        FrontierSet.add(neighbor.getState())
                        if max_depth < neighbor.getDepth():
                            max_depth = neighbor.getDepth()
                if max_fringe_size < len(Frontier):
                    max_fringe_size = len(Frontier)
        else:
            done = True
    if success:
        path, path_to_goal, cost_of_path = retracePath(node)
        running_time = time() - start_time
        output.append('path_to_goal: ' + str(path_to_goal))
        output.append('cost_of_path: ' + str(cost_of_path))
        output.append('nodes_expanded: ' + str(nodes_expanded))
        output.append('fringe_size: ' + str(len(Frontier))) 
        output.append('max_fringe_size: ' + str(max_fringe_size))
        output.append('search_depth: ' + str(node.getDepth()))
        output.append('max_depth: ' +  str(max_depth))
        output.append('running_time: ' + "{0:.8f}".format(running_time))
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        output.append('max_ram_usage: ' + "{0:.8f}".format(mem))
    return output

def IDAStar(start):
    start_time = time()
#    target = '*'.join([str(e) for e in range(len(start))])
    start_state = '*'.join(start)
    pos = start.index('0')
    N = int(sqrt(len(start)))
    x0 = int(pos/N)
    y0 = pos - x0 * N
    root = Npuzzle(start_state, None, x0, y0, None, 0, 0)
    bound = root.getF()
    done = False
    output = []
    nodes_expanded = 0
    max_fringe_size = 0
    max_depth = 0
    found = False
    while(not done):
        minBound, result, found = DFLStar(root, bound)
        if minBound < 0 or found:
            done = True
        else:
            bound = minBound
        nodes_expanded += result['nexp']
        max_fringe_size = max(max_fringe_size, result['mfsize'])
        max_depth = max(max_depth, result['mdp'])
        
    if found:
        running_time = time() - start_time
        output.append('path_to_goal: ' + str(result['ptg']))
        output.append('cost_of_path: ' + str(result['cop']))
        output.append('nodes_expanded: ' + str(nodes_expanded))
        output.append('fringe_size: ' + str(result['fsize'])) 
        output.append('max_fringe_size: ' + str(max_fringe_size))
        output.append('search_depth: ' + str(result['sdp']))
        output.append('max_depth: ' +  str(max_depth))
        output.append('running_time: ' + "{0:.8f}".format(running_time))
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        output.append('max_ram_usage: ' + "{0:.8f}".format(mem))
    return output
    
       
def DFLStar(root, bound):
    target = '*'.join([str(e) for e in range(len(root.getState().split('*')))])
    Frontier = [root]
    FrontierSet = set([root.getState()])
    Explored = {}
    done = False
    success = False
    nodes_expanded = 0
    max_fringe_size = 1
    max_depth = root.getDepth()
    result = {}
    
    minBound = -1
    while(not done):
        if Frontier:
            node = Frontier.pop(0) 
            FrontierSet.remove(node.getState())
            if node.getState() in Explored and node.getF() >= Explored[node.getState()]:
                pass 
            else:
                Explored[node.getState()] = node.getF()
           
                if node.getF() > bound:
                    if minBound < 0:
                        minBound = node.getF()
                    else:
                        minBound = min(minBound, node.getF())
                elif node.getState() == target:
                    done = True
                    success = True
                else:
                    nodes_expanded += 1
                    neighbors = node.getNeighbors()
                    for i in range(len(neighbors) - 1, -1, -1):
                        neighbor = neighbors[i]
                        if neighbor.getState() in Explored and neighbor.getF() >= Explored[neighbor.getState()]:
                            pass
                        elif neighbor.getState() in FrontierSet:
                            pass
                        else:
                            Frontier.insert(0, neighbor)
                            FrontierSet.add(neighbor.getState())
                            if max_depth < neighbor.getDepth():
                                max_depth = neighbor.getDepth()
                    if max_fringe_size < len(Frontier):
                        max_fringe_size = len(Frontier)
        else:
            done = True
    result['nexp'] = nodes_expanded
    result['fsize'] = len(Frontier)
    result['mfsize'] = max_fringe_size
    result['sdp'] = node.getDepth()
    result['mdp'] = max_depth
    if success:
        path, path_to_goal, cost_of_path = retracePath(node)
        #running_time = time() - start_time
        result['ptg'] = path_to_goal
        result['cop'] = cost_of_path
       
    return minBound, result, success 
#    
#test = ['1','2','5','3','4','0','6','7','8']
#out1 = IDAStar(test)
#root = Npuzzle('1*2*5*3*4*0*6*7*8', None, 1, 2, None, 0, 0) 
#minBound, output, nexp, fronsize, maxfron, depth, maxdep = DFLStar(root, root.getF())
#test2 = ['3', '2', '1','0']
#out2 = DFS(test)
#test3 = ['8', '7', '6','5','4', '3', '2', '1', '0']
#out3star = IDAStar(test3)
#out3b = BFS(test3)
#out3d = DFS(test3)
#out3a = AStar(test3)



  
def main(argv):
   
    method = argv[0]
    start = argv[1].split(',')
    if method == 'bfs':
        output = BFS(start)
    elif method == 'dfs':
        output = DFS(start)
    elif method == 'ast':
        output = AStar(start)
    else:
        output = IDAStar(start)
    f = open('output.txt','w')
    for e in output:
        f.write(e+'\n')
    f.close()
if __name__ == "__main__":
    main(sys.argv[1:])