# Breakout_game
This project will demonstrate how deep reinforcement learning can be implemented and applied to play a CartPole game using Keras and Gym.

The state in Breakout is the image of the game


#### Actions : 
- Left
- Right


#### Reward :
- +2 for every 
- -1 everytime it loses the ball
- -2 in the end of a game

#### The neural network:
##### 1 st case (without squeezenet):
- The state is a picture of size (210,160,3), we turn it into grayscale and resize it before giving it to the neural network (the preprocessing), then we put togeter the last 4 states so we can have more information. 
- The input of the neural network is the result of size (84,84,4) 
- ###### Hidden layer:
          - Conv1: 32 filters, size 8X8, stride 4, activation function ReLU
          - Conv1: 32 filters, size 8X8, stride 4, activation function ReLU

#### 2 nd case (with squeezenet):
- Input layer: 4 nodes receiving the state (4 observations)
- N hidden layers with M nodes 
- Output layer: 2 nodes (2 actions possible)


### Reinforcement learning
Q-function: used to approximate the reward based on a state

Q(s,a) :Calculates the expected future value from state "s" and action "a"

 Problem in reinforcement learning = unstable / diverge when neural networks are used to represent the action-value function.
 
#### Causes:
- Correlation present in the sequence of observations
- small updates to Q may change the data distribution
- correlations between the action-values (Q) and the target value (r+ gamma * Q(s,a))

#### Solution:
- use a replay function that randomizes over the data => removing correlation in the observation sequence and smoothing changes
- use an iterative update that adjusts the action-values (Q) toward target values that are only periodically adjusted => reducing correlations with the target

#### How to use theses solutions?
- We need to store the agent's experiences (memory) 
- During learning, we apply Q-learning updates on minibatches
- The target at iteration "i" only updates with the Q-network parameters of iteration "i-1"
- Calculate average score-per-episode + average predicted 




### References
Human-level control through deep reinforcement learning,


### Installation Dependences
Python
TensorFlow
Keras
Gym
Gym [atari]
