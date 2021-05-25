# multiAgents.py
# --------------
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

import operator

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        if action == "Stop":
            return float("-inf")
        minGhostScore = float("inf")
        for ghostState in newGhostStates:
            X = ghostState.getPosition()[0]
            Y = ghostState.getPosition()[1]
            minGhostScore = min(minGhostScore,manhattanDistance((X,Y), newPos))
        scared = min(newScaredTimes) > 0
        if not scared and minGhostScore < 2:
            return float("-inf")

        if successorGameState.isWin():
            return float("inf")

        minFoodPacDist = float("inf")
        for foodPos in newFood.asList():
            if foodPos == newPos:
                minFoodPacDist=0
            else:
                minFoodPacDist = min(minFoodPacDist, (manhattanDistance(foodPos, newPos)))

        a =successorGameState.getScore()
        if minFoodPacDist == 0:
            return float("inf")
        if scared:
            return 1000000-minFoodPacDist+a

        return minGhostScore/2-minFoodPacDist+a

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.
       """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()-1
        depth = 0
        result = []
        for action in gameState.getLegalActions(0):
            result.append((action, self.value(gameState.generateSuccessor(0, action), depth, 1)))
        #max value for root
        result.sort(key = operator.itemgetter(1))
        return result[-1][0]

    def value(self, state, depth, agent):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        if agent == 0:
            return self.max_value(state, agent, depth)
        if agent != 0:
            return self.min_value(state, agent, depth)

    def max_value(self, state, agent, depth):
        v = float("-inf")
        legalMoves = state.getLegalActions(agent)
        for action in legalMoves:
            v = max(v, self.value(state.generateSuccessor(agent, action), depth, agent + 1))
        return v

    def min_value(self, state, agent, depth):
        v = float("inf")
        legalMoves = state.getLegalActions(agent)
        for action in legalMoves:
            if agent < self.numAgents:
                v = min(v, self.value(state.generateSuccessor(agent, action), depth, agent + 1))
            else:
                v = min(v, self.value(state.generateSuccessor(agent, action), depth+1, 0))#at the last of ghost, the next will be max
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents() - 1
        a = float("-inf")
        b = float("inf")
        depth = 1
        maximum = float("-inf")
        maxAction = None
        for action in gameState.getLegalActions(0):
            result = self.value(gameState.generateSuccessor(0, action), depth, 1, a, b)
            if result > maximum:
                maximum = result
                maxAction = action
            # if result > self.beta:
            #     return action
            a = max(a, result)
        return maxAction

    def value(self, state, depth, agent, a, b):
        if agent == 0 or agent > self.numAgents:
            depth += 1
        if state.isWin() or state.isLose() or depth > self.depth:
            return self.evaluationFunction(state)
        if agent == 0 or agent > self.numAgents:
            return self.max_value(state, 0, depth, a, b)
        if agent != 0 and agent <= self.numAgents:
            return self.min_value(state, agent, depth, a, b)

    def max_value(self, state, agent, depth, a, b):
        v = float("-inf")
        legalMoves = state.getLegalActions(agent)
        for action in legalMoves:
            v = max(v, self.value(state.generateSuccessor(agent, action), depth, agent + 1, a, b))
            if v > b:
                return v
            a = max(a, v)
        return v

    def min_value(self, state, agent, depth, a, b):
        v = float("inf")
        legalMoves = state.getLegalActions(agent)
        for action in legalMoves:
            v = min(v, self.value(state.generateSuccessor(agent, action), depth, agent + 1, a, b))
            if v < a:
                return v
            b = min(b, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents() - 1
        depth = 0
        result = []
        for action in gameState.getLegalActions(0):
            result.append((action, self.value(gameState.generateSuccessor(0, action), depth, 1)))
        # max value for root
        result.sort(key=operator.itemgetter(1))
        return result[-1][0]

    def value(self, state, depth, agent):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        if agent == 0:
            return self.max_value(state, agent, depth)
        if agent != 0:
            return self.exp_value(state, agent, depth)

    def max_value(self, state, agent, depth):
        v = float("-inf")
        legalMoves = state.getLegalActions(agent)
        for action in legalMoves:
            v = max(v, self.value(state.generateSuccessor(agent, action), depth, agent + 1))
        return v

    def exp_value(self, state, agent, depth):
        v = 0
        legalMoves = state.getLegalActions(agent)
        prob = 1 / len(legalMoves)

        for action in legalMoves:
            if agent < self.numAgents:
                v += prob * self.value(state.generateSuccessor(agent, action), depth, agent + 1)
            else:
                v += prob * self.value(state.generateSuccessor(agent, action), depth + 1,
                                      0)  # at the last of ghost, the next will be max
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: The evaluation function takes in the current GameState and
                 returns a number, where higher numbers are better.
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsule = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    score = currentGameState.getScore() * 2

    if currentGameState.isLose():
        return float("-inf")
    if currentGameState.isWin():
        return float("inf")

    minGhostScore = float("inf")
    for ghostState in ghostStates:
        X = ghostState.getPosition()[0]
        Y = ghostState.getPosition()[1]
        minGhostScore = min(minGhostScore, manhattanDistance((X, Y), pos))

    scared = min(scaredTimes) > 0
    if not scared and minGhostScore < 4:
        return float("-inf")

    if not scared and minGhostScore < 2:
        score -= 800

    if not scared and minGhostScore < 1:
        score -= 1600

    minFoodPacDist = float("inf")
    for foodPos in food.asList():
        if foodPos == pos:
            minFoodPacDist = 0
        else:
            minFoodPacDist = min(minFoodPacDist, (manhattanDistance(foodPos, pos)))

    if len(capsule) == 2:
        score += 300
    if len(capsule) == 1:
        score += 600
    if len(capsule) == 0:
        score += 1000

    if minFoodPacDist == 0:
        return float("inf")
    if scared:
        return score + 1000000 - minFoodPacDist

    return score - minFoodPacDist

# Abbreviation
better = betterEvaluationFunction
