import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.1
        self.gamma = 1
        self.epsilon = 0.1
        self.decay_rate = 1e-4
        self.i_episode = 0
        print("alpha: ", self.alpha, " | ", \
              "epsilon: ", self.epsilon, " | ", \
              "decay: ", self.decay_rate) 

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.epsilon_greedy_probs(self.Q[state])
        return np.random.choice(np.arange(len(probs)), p = probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        probs = self.epsilon_greedy_probs(self.Q[next_state])
        
        # Expected Sarsa
        self.Q[state][action] = self.update_Q(self.Q[state][action], \
                                              np.dot(probs, self.Q[next_state]), \
                                              reward)

        if done:
            self.epsilon = self.epsilon * np.exp(-self.decay_rate * self.i_episode)
            self.i_episode += 1
        
    def epsilon_greedy_probs(self, q_s):
        """ Get epsilon greedy probs for given state
            Take a Q table entry for a single state and returns chosen/drawn action from pmf
        Params
        ======
        - q_s (list): action values for a particular state
        - epsilon (float): level of "greed" betwee 0 and 1 

        Returns
        =======
        - pmf (probability mass function) of possible actions
        """
        nA = len(q_s)
        idx_max = np.argmax(q_s)
        eps = self.epsilon
        return np.array([(1 - eps) + eps/nA if a == idx_max else eps/nA for a in range(nA)])
    
    def update_Q(self, Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))
  
    