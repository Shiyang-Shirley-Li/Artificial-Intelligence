# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

#
# University of Utah, CS 4300
# Shirley(Shiyang) Li, u1160160
# Lin Pan, u1213321
#

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def general_graph_search(problem, fringe, type, heuristic):
    """
    A generic graph search. Fringe is the data structure chosen for a specific search
    algorithm. Type is to differentiate different search algorithms, since they may have
    different methods for different data structures. Heuristic is for A* search heuristic
    function.
    Returns a sequence of moves that solves a specific maze.
    """
    closed = set([])  # an empty set
    path = []
    action = None
    preNode = None
    accumulateCost = 0
    #A node is four tuples with current state(position), action, previous node and accumulated costs
    node = (problem.getStartState(), action, preNode, accumulateCost)

    if type is 'UCS':
        fringe.push(node, 0)
    elif type is 'aStar':
        fringe.push(node, 0+heuristic(node[0], problem))
    else:
        fringe.push(node)

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            while node is not None:
                if node[1] is not None:
                    path.append(node[1])
                    node = node[2]
                else:
                    path.reverse()
                    return path
        if node[0] not in closed:
            closed.add(node[0])
            childNodes = problem.getSuccessors(node[0])
            for successor in childNodes:
                if type is 'UCS':
                    fringe.push((successor[0], successor[1], node, node[3]+successor[2]), node[3]+successor[2])
                elif type is 'aStar':
                    fringe.push((successor[0], successor[1], node, node[3]+successor[2]), node[3]+successor[2]+heuristic(successor[0], problem))
                else:
                    fringe.push((successor[0], successor[1], node, 1))
    return path


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    stack = util.Stack()
    return general_graph_search(problem, stack, 'stack', nullHeuristic)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    return general_graph_search(problem, queue, 'queue', nullHeuristic)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priorityQ = util.PriorityQueue()
    return general_graph_search(problem, priorityQ, 'UCS', nullHeuristic)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priorityQ = util.PriorityQueue()
    return general_graph_search(problem, priorityQ, 'aStar', heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
