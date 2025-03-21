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
        start wtih V_0(s) = 0
        Given vector of V_k(s) values, do one ply of expectimax 
        from each state:
        <Bellman equation>
        (compute vector of values V_{k+1}(s))
        Repeat until convergence

        V_0(s) = 0 for all s
        k = 0
        while k < N:
            for all s in S:
                do Bellman
            k = k + 1 // k++
        '''
        '''
        V_0(s) = 0 for all s?
        states = self.mdp.getStates()
        for i in range(self.iterations): # k = 0; k < N; k++
            for state in states: # for all s in S
                # Bellman: V(s) = max_a sum_s' T(s, a, s')*[R(s, a, s') + gamma*V_k(s')]
                # i.e., value = max sum of prob of transitioning from s to s' given a * (reward for transitioning from s to s' given a + gamma * value of s')
                actions = self.mdp.getPossibleActions(state) # get all a in A(s)
                for action in actions: # for a in A(s)
                    value = 0
                    # get T(s, a, s')
                    # get all successors (nextState), s', for the state, s, given action, a
                    # and get the prob for each successor (prob of transitioning from s to s' given a)
                    successors = self.mdp.getTransitionStatesAndProbs(state, action)
                    # this function gets all (s', prob) pairs for s given a 
                    # i.e., all (nextState) for currentState, all (prob from currentState to nextState given action) for action in currentState
                    # e.g., (nextState1, prob1), (nextState2, prob2), ...
                    for nextState, prob in successors:
                        # get R(s, a, s')
                        # get the reward for transitioning from s to s' given a
                        reward = self.mdp.getReward(state, action, nextState)
                        # get gamma*V_k(s')
                        # get the value of the nextState, V_k(s'), and discount it by gamma
                        gamma = self.discount
                        #value += prob * (reward + gamma * self.values[nextState])
                    # update max value for action
                #update best value for state
            #update values
        '''
        # Q1*
        # Get all states
        states = self.mdp.getStates()

        # Loop through all iterations
        for i in range(self.iterations):
            # Create a copy of the values
            new_values = self.values.copy()

            # Loop through all states
            for state in states:
                # Check if state is terminal
                if self.mdp.isTerminal(state):
                    continue

                # Get all possible actions
                actions = self.mdp.getPossibleActions(state)

                # Initialize the value
                value = float("-inf")

                # Loop through all actions
                for action in actions:
                    # Get the Q value
                    # Q(s, a) = sum_s' T(s, a, s')*[R(s, a, s') + gamma*V_k(s')]
                    q_value = self.computeQValueFromValues(state, action)

                    # Update the value
                    value = max(value, q_value)

                # Update the value
                new_values[state] = value

            # Update the values
            self.values = new_values

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
        
        # Get the transition states and probs
        successors = self.mdp.getTransitionStatesAndProbs(state, action)

        q_value = 0

        # loop through all successors and calculate the Q value
        for next_state, prob in successors:
            # get reward, R(s, a, s')
            reward = self.mdp.getReward(state, action, next_state)
            # get value of next state, discounted by gamma, gamma * V_k(s')
            gamma = self.discount
            nextStateValue = gamma * self.values[next_state]
            # get q value for current state and action, 
            # sum_s' (T(s, a, s')*[R(s, a, s') + gamma*V_k(s')])
            q_value += prob * (reward + nextStateValue)
        
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Q1.3*
        # get best action for state
        # policy(s) = arg_max_{a in actions} Q(s,a)
        # i.e., get the action that maximizes the Q value for the state
        actions = self.mdp.getPossibleActions(state)

        # Initialize the best action
        best_action = None
        best_value = float("-inf")

        # Loop through all actions
        for action in actions:
            # Get the Q value
            q_value = self.computeQValueFromValues(state, action)

            # Update the best action
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
