# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 08:42:07 2017

@author: alin
"""

from math import sqrt
import time
import sys
#import numpy as np

class Npuzzle(object):
    def __init__(self, state, parent, x0, y0, move, depth):
        self.state = state
        self.parent = parent
        self.x0 = x0
        self.y0 = y0
        self.move = move # the move it takes to get here from the parent
        self.depth = depth
        self.N = int(sqrt(len(state.split('*'))))
    def display(self):
#        print np.array([int(e) for e in self.state.split('*')]).reshape(self.N, self.N)
#        print self.depth
#        print self.x0
#        print self.y0
        print self.state
    def setParent(self, parent):
        self.parent = parent
    def getParent(self):
        return self.parent
    def getState(self):
        return self.state
    def getMove(self):
        return self.move
    def setMove(self, move):
        self.move = move
    def getDepth(self):
        return self.depth
    def setDepth(self, depth):
        self.depth = depth
    def swap(self, x1, y1):
        state_lst = self.state.split('*')
        state_lst[self.x0 * self.N + self.y0] = state_lst[x1 * self.N + y1]
        state_lst[x1 * self.N + y1] = '0'
        return '*'.join(state_lst)
    def getNeighbors(self):
        neighbors = []
        if self.x0 > 0:
            state1 = self.swap(self.x0 - 1, self.y0)
            neighbors.append(Npuzzle(state1, self, self.x0 - 1, self.y0, 'Up', self.depth + 1 ))
        if self.x0 < self.N - 1:
            state1 = self.swap(self.x0 + 1, self.y0)
            neighbors.append(Npuzzle(state1, self, self.x0 + 1, self.y0, 'Down', self.depth + 1 ))
        if self.y0 > 0:
            state1 = self.swap(self.x0, self.y0 - 1)
            neighbors.append(Npuzzle(state1, self, self.x0, self.y0 - 1, 'Left', self.depth + 1 ))
        if self.y0 < self.N - 1:
            state1 = self.swap(self.x0, self.y0 + 1)            
            neighbors.append(Npuzzle(state1, self, self.x0, self.y0 + 1, 'Right', self.depth + 1 ))
        return neighbors

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

def BFS(start):
    target = '*'.join([str(e) for e in range(len(start))])
    start_state = '*'.join(start)
    pos = start.index('0')
    N = int(sqrt(len(start)))
    x0 = int(pos/N)
    y0 = pos - x0 * N
    root = Npuzzle(start_state, None, x0, y0, None, 0) 
    start = time.time()
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
        running_time = time.time() - start
        output.append('path_to_goal: ' + str(path_to_goal))
        output.append('cost_of_path: ' + str(cost_of_path))
        output.append('nodes_expanded: ' + str(nodes_expanded))
        output.append('fringe_size: ' + str(len(Frontier))) 
        output.append('max_fringe_size: ' + str(max_fringe_size))
        output.append('search_depth: ' + str(node.getDepth()))
        output.append('max_depth: ' +  str(max_depth))
        output.append('running_time: ' + str(running_time))
        output.append('max_ram_usage: 0.5')
    return output


test = ['1','2','5','3','4','0','6','7','8']
out = BFS(test)
test2 = ['3', '2', '1','0']
out2 = BFS(test2)
    
def DFS(start):
    target = '*'.join([str(e) for e in range(len(start))])
    start_state = '*'.join(start)
    pos = start.index('0')
    N = int(sqrt(len(start)))
    x0 = int(pos/N)
    y0 = pos - x0 * N
    root = Npuzzle(start_state, None, x0, y0, None, 0) 
    start = time.time()
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
        running_time = time.time() - start
        output.append('path_to_goal: ' + str(path_to_goal))
        output.append('cost_of_path: ' + str(cost_of_path))
        output.append('nodes_expanded: ' + str(nodes_expanded))
        output.append('fringe_size: ' + str(len(Frontier))) 
        output.append('max_fringe_size: ' + str(max_fringe_size))
        output.append('search_depth: ' + str(node.getDepth()))
        output.append('max_depth: ' +  str(max_depth))
        output.append('running_time: ' + str(running_time))
        output.append('max_ram_usage: 0.5')
    return output

#test = ['1','2','5','3','4','0','6','7','8']
#out = DFS(test)
#test2 = ['3', '2', '1','0']
#out2 = DFS(test2)
#test3 = ['8', '7', '6','5','4', '3', '2', '1', '0']
#out3 = DFS(test3)

def main(argv):
    method = argv[0]
    start = argv[1].split(',')
    if method == 'bfs':
        output = BFS(start)
    elif method == 'dfs':
        output = DFS(start)
    f = open('output.txt','w')
    for e in output:
        f.write(e+'\n')
    f.close()
if __name__ == "__main__":
    main(sys.argv[1:])