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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        import math
        score = successorGameState.getScore()
        score -= 128 * newFood.count() # I don't really understand why 128 and the following 4 play well here
        food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        score -= 4 * min(food_distances) if food_distances else 0
        ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        score -= min(ghost_distances) if ghost_distances else 0
        if ghost_distances and 0 <= min(ghost_distances) and min(ghost_distances) < 2:
            return -math.inf
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        import math
        def minimax(state, agent_index, depth):
            if depth == 0 or state.isWin() or state.isLose():
                return None, self.evaluationFunction(state)
            if agent_index == state.getNumAgents() - 1:
                depth -= 1
            final_value = -math.inf if agent_index == 0 else math.inf
            final_action = None
            next_agent_index = (agent_index + 1) % state.getNumAgents()
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                _, next_state_value = minimax(next_state, next_agent_index, depth)
                if (agent_index == 0) == (next_state_value > final_value):
                    final_value = next_state_value
                    final_action = action
            return final_action, final_value
        return minimax(gameState, 0, self.depth)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import math
        def alpha_beta_prune(state, agent_index, depth, alpha, beta):
            do_max = agent_index == 0
            do_min = not do_max
            if depth == 0 or state.isWin() or state.isLose():
                return None, self.evaluationFunction(state)
            if agent_index == state.getNumAgents() - 1:
                depth -= 1
            final_value = -math.inf if agent_index == 0 else math.inf
            final_action = None
            next_agent_index = (agent_index + 1) % state.getNumAgents()
            value = None
            alpha_beta = [alpha, beta]
            for action in state.getLegalActions(agent_index):
                if value is not None and [value < alpha_beta[do_max], alpha_beta[do_max] < value][do_max]:
                    break
                next_state = state.generateSuccessor(agent_index, action)
                _, value = alpha_beta_prune(next_state, next_agent_index, depth, *alpha_beta)
                if [alpha_beta[do_min] < value, value < alpha_beta[do_min]][do_min]:
                    alpha_beta[do_min] = value
                if do_max == (value > final_value):
                    final_value = value
                    final_action = action
            return final_action, final_value
        return alpha_beta_prune(gameState, 0, self.depth, -math.inf, math.inf)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        import math
        def expectimax(state, agent_index, depth):
            if depth == 0 or state.isWin() or state.isLose():
                return None, self.evaluationFunction(state)
            if agent_index == state.getNumAgents() - 1:
                depth -= 1
            final_value = -math.inf if agent_index == 0 else math.inf
            final_action = None
            next_agent_index = (agent_index + 1) % state.getNumAgents()
            values = []
            for action in state.getLegalActions(agent_index):
                next_state = state.generateSuccessor(agent_index, action)
                _, next_state_value = expectimax(next_state, next_agent_index, depth)
                if (agent_index == 0):
                    if next_state_value > final_value:
                        final_value = next_state_value
                        final_action = action
                else:
                    values.append(next_state_value)
            return final_action, final_value if agent_index == 0 else sum(values) / len(values)
        return expectimax(gameState, 0, self.depth)[0]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    score -= currentGameState.getFood().count()
    pos = currentGameState.getPacmanPosition()
    ghost_distances = [manhattanDistance(pos, ghost.getPosition()) for ghost in currentGameState.getGhostStates()]
    score -= min(ghost_distances) if ghost_distances else 0
    return score

# Abbreviation
better = betterEvaluationFunction
