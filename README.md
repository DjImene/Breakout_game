# Breakout_game
This project will demonstrate how deep reinforcement learning can be implemented and applied to play a CartPole game using Keras and Gym.

The state in Breakout is the image of the game

<p align="center">
  <img src="https://github.com/DjImene/Breakout_game/blob/master/state0.jpg" width="350"/>
</p>

For this game we started by making a python code in order to resolve it then we used the transfer learning with squeezenet to fasten it.

#### Actions : 
- Left
- Right
- Fire
- Noop


#### Reward :
- +2 for every 
- -1 for the firt ball it loses
- -2 for the second ball it loses
- -3 for the third ball it loses
- -4 for the fourth ball it loses
- -6 in the end of a game


#### The neural network:
##### 1 st case (without squeezenet):
- The state is a picture of size (210,160,3), we turn it into grayscale and resize it before giving it to the neural network (the preprocessing), then we put togeter the last 4 states so we can have more information. 

<p align="center">
  <img src="https://github.com/DjImene/Breakout_game/blob/master/4frames.png" width="500"/>
</p>


- The input of the neural network is the result of size (84,84,4) 
- ###### Hidden layer:
          - Conv1: 32 filters, size 8X8, stride 4, activation function ReLU
          - Conv2: 64 filters, size 4X4, stride 2, activation function ReLU
          - Conv3: 64 filters, size 3X3, stride 1, activation function ReLU
          - Flatten
          - Fully connected layer (512 units), activation function ReLU
       
- The output layer: nb_actions neurones => 4 neurones

#### 2 nd case (with squeezenet):
- Squeezenet 
- Input layer: fully connected layer of 512 neurones, input size (4,1000), 1000 features for the last 4 states
- Output layer: 4 nodes (4 actions possible)


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
Human-level control through deep reinforcement learning,  Nature, Volodymyr Mnih,	Koray Kavukcuoglu,	David Silver, Andrei A. Rusu,	Joel Veness,	Marc G. Bellemare,	Alex Graves,	Martin Riedmiller,	Andreas K. Fidjeland,	Georg Ostrovski,	Stig Petersen,	Charles Beattie,	Amir Sadik,	Ioannis Antonoglou,	Helen King,	Dharshan Kumaran, Daan Wierstra,	Shane Legg	& Demis Hassabis.


### Dependencies
- Python
- TensorFlow
- Keras
- Gym
- Gym [atari]

### Content of the project
- In the document "Breakout.py", we have a first version of the code without squeezenet. In this code, we did the preprocessing of the images and all the processing part using Keras.
- We then wanted to use the transfer learning to handle the image processing part, we have chosen squeezenet for that because it's small and fast.

### Problems
- For the python code without squeezenet, when we get to approximatly 95 games played by the agent, it suddenly stop playing and start displaying different colors on the emulator. 


### To do
- Resolve the problems
- Improve the results
