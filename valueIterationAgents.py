# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """

        '''
        pseudocode from lec:
        V_0(s) = 0 for all s
        k = 0
        while k < N:
            for all s in S:
                do Bellman: V(s) = max_a sum_s' T(s, a, s')*[R(s, a, s') + gamma*V_k(s')]
                # i.e., value = max sum of prob of transitioning from s to s' given a * (reward for transitioning from s to s' given a + gamma * value of s')
            k = k + 1 // k++
        '''

        # Q1
        # V_0(s) = 0 for all s
        # all values (self.values) are initialized to 0 with util.Counter()
        
        # Get all states
        states = self.mdp.getStates()

        # Loop through the given number of iterations (while k < N)
        for i in range(self.iterations): # does k = 0; k < N; k++
            # Create a copy of the current values 
            newValues = self.values.copy()

            # Loop through all states
            for state in states: # for all s in S
                # Check if state is terminal
                if self.mdp.isTerminal(state):
                    newValues[state] = 0

                # steps for Bellman equation
                # Get all possible actions (A(s))
                actions = self.mdp.getPossibleActions(state)

                # Initialize the value for the state
                value = float("-inf")

                # Loop through all actions (a in A(s))
                for action in actions:
                    # Get the Q value
                    # Q(s, a) = sum_s' T(s, a, s')*[R(s, a, s') + gamma*V_k(s')]
                    qValue = self.computeQValueFromValues(state, action)

                    # Update the value of the state 
                    value = max(value, qValue) # V(s) = max_a sum_s' (...)

                # Update the value in copied dict to max (best) value
                newValues[state] = value

            # Update the values dict
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        # Q1.2
        # this is basically just the Bellman equation
        # Q(s, a) = sum_s' T(s, a, s')*[R(s, a, s') + gamma*V_k(s')]
        
        # get transition states and probs, i.e., T(s, a, s'):
            # get all successors (nextState), s', for the 
            # state, s, given action, a, and get the prob for 
            # each successor (prob of transitioning from s to 
            # s' given a)
        successors = self.mdp.getTransitionStatesAndProbs(state, action)
            # this function gets all (s', prob) pairs for
            # s given a # i.e., all (nextState) for 
            # currentState, all (prob from currentState to 
            # nextState given action) for action in currentState
            # e.g., (nextState1, prob1), (nextState2, prob2), ...
            
        qValue = 0

        # loop through all successors and calculate the Q value
        for nextState, prob in successors:
            # get R(s, a, s'):
                # the reward for transitioning from s (state)
                # to s' (nextState) given a (action)
            reward = self.mdp.getReward(state, action, nextState)
            # get gamma*V_k(s'): 
            # get value of next state, discounted by gamma
            nextStateValue = self.discount * self.getValue[nextState]
            
            # add to sum:
            # sum_s' (T(s, a, s')*[R(s, a, s') + gamma*V_k(s')])
            qValue += prob * (reward + nextStateValue)
        
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Q1.3
        # basically same as runValueIteration(...), but calculates 
        # for one state
        # get best action for state
        # policy(s) = arg_max_{a in actions} Q(s,a)
        # i.e., get the action that maximizes the Q value for the state
        actions = self.mdp.getPossibleActions(state)

        # Initialize the best action and value
        bestAction = None
        bestValue = float("-inf")

        # Loop through all actions
        for action in actions:
            # Get the Q value
            qValue = self.computeQValueFromValues(state, action)

            # Update the best action and value
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
