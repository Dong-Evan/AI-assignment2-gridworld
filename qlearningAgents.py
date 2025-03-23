# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalue = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Q2.1

        # util.Counter() initializes dict with default value 0
        # we will be updating self.qvalue dict, so just return
        # value for key (state, action) from self.qvalue dict
        return self.qvalue[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # Q2.2

        # get legal actions for state
        legalActions = self.getLegalActions(state)
        # if no legal actions, return 0.0
        if not legalActions:
            return 0.0
        
        # initialize value to negative infinity
        bestValue = float("-inf")
        # loop through all legal actions
        for action in legalActions:
            # get Q value for state, action
            qValue = self.getQValue(state, action)
            # update best value if q_value is greater
            bestValue = max(bestValue, qValue)
            
        return bestValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Q2.3
        
        # get legal actions for state
        legalActions = self.getLegalActions(state)
        # if no leval actions, return None
        if not legalActions:
            return None
        
        # initialize best action and value
        bestAction = None
        bestValue = float("-inf")
        
        # loop through all legal actions
        for action in legalActions:
            # get the Q value for state, action
            qValue = self.getQValue(state, action)
            # update action and value if q_value is greater
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action
            # break ties with random.choice()
            elif qValue == bestValue: 
                # technically not best (most equal) tie breaker 
                # because it chooses from last best action found 
                # and new best action found
                bestAction = random.choice([bestAction, action])
                
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        
        legalActions = self.getLegalActions(state)
        action = None
        
        # Q3

        # if no legal actions, return None
        if not legalActions:
            return None

        # pick random action with probability epsilon
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        # else, pick the best policy action
        else:
            action = self.computeActionFromQValues(state)
        
        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Q2.4

        # get Q value for state, action; old estimate: Q(s, a)
        qValue = self.getQValue(state, action)
        # get max (best) Q value for next state, max_aQ(s', a')
        bestQValue = self.computeValueFromQValues(nextState)
            # computeValueFromQValues(nextState) returns the max Q
            # value of the legal actions for nextState
        # calculate sample estimate: R(s, a, s') + gamma*max_aQ(s', a')
        sample = reward + self.discount * bestQValue
        # update Q value: Q(s, a) = (1 - alpha)*Q(s, a) + alpha*sample
        self.qvalue[(state, action)] = (1 - self.alpha) * qValue + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

