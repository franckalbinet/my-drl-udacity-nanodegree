## Project 1: Navigation report

## Table of content
* [I. Introduction](#introduction)
* [II. Learning algorithms](#learning-algo)
  * [II.1 Vanilla DQN](#vanilla-dqn)

## I. Introduction
<a id="introduction"></a>

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


## II. Learning algorithms
<a id="learning-algo"></a>

The goal of this project is to implement value-based Deep Reinforcement Learning algorithms in order to solve Unity Bananas' environment. We have at our disposal a series of algorithms from Deep Q-network (DQN), double DQN, Prioritized Replay DQN, Dueling DQN, ...

Our approach consists in:
1. re-use Vanilla DQN with hyperparameters used to solve OpenAI Gym Lunar Landing environment;
2. based on visual observation of agent's behaviour during learning, guessing more optimal hyperparameters;
3. explore more systematically hyperparameters space to find out "best parameters";
4. implement one of the improved version of DQN (Double DQN in our case).

### II.1 Vanilla DQN with default parameters
<a id="vanilla-dqn"></a>



We first implemented a Vanilla DQN approach with the followin agent and DQN hyperparameters:

```
n_episodes=2000 # maximum number of training episodes
max_t=1000 # maximum number of timesteps per episode
eps_start=1.0 # starting value of epsilon, for epsilon-greedy action selection
eps_end=0.01 # minimum value of epsilon
eps_decay=0.995 # multiplicative factor (per episode) for decreasing epsilon


```

<img src="img/dqn-default.png" width="400" />

## Plot of rewards

## Ideas for future work

