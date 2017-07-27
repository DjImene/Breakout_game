#Code pour atari "breakout"
import warnings
warnings.filterwarnings('ignore')
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D,Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from skimage.color import rgb2gray
from skimage.transform import resize
import copy
import matplotlib.pyplot as plt
from skimage import color



#The number of games must be high enough to observe the variation and see if there is a permanent state
nb_games=100000
learning_rate=0.0002
#I've tried diffrent batch sizes as 32, 64, 128,... and I've found out that 64 is giving the best results
batch_size=32
rate=1.0
#rate=0.7
gamma=0.95
state_length=4
frame_width=84
frame_height=84

#We define an agent that learn from the environment by interacting with it
#we get observations after each of it's actions on the environment
#we store those experiences so our agent could learn from its mistakes
class my_agent:
    #The init funcion is an initialization function that calls the "myNN" function which builds our neural network
    #it also defines two constant values, the input and output sizes of our neural network
    def __init__(self,output_size):
        self.output_size=output_size
        self.model=self.myNN()

    #Defining the neural network, fixed number of neurones for the input and output layers
    #For different number of neurons and different number of hidden layer, the results are not as good as the ones
    #provided with this specific modal, others activations functions as well as optimizers and losses have been tested
    #better results have been found with the modal bellow
    def myNN(self):     
        model = Sequential()
        model.add(Convolution2D(32,(8,8),subsample=(4,4),activation='relu', input_shape=(frame_width, frame_height,state_length)))     
        model.add(Convolution2D(64,4,4,subsample=(2,2),activation='relu'))
        model.add(Convolution2D(64,3,3,subsample=(1,1),activation='relu'))
        model.add(Flatten())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.output_size, activation='relu'))
        model.compile(loss='mse',optimizer=RMSprop(lr=learning_rate))
        return model
    
   
    def agent_action(self, state):
        predict_val=self.model.predict(state)
        return np.argmax(predict_val[0])



    #This function is used to train the neural network bases on the memory(the previous experiencies or actions it had)
    def replay(self):
        minibatch = random.sample(my_memory,batch_size)
        
        for state,futur_state, action, reward,tot_reward,info, done in minibatch:
            target = reward
            if not done:
                #target=best_reward
                target = reward + gamma * np.amax(self.model.predict(futur_state)[0])   
            target_f = self.model.predict(state)
            target_f[0][action] = target
            #the neural network will predict the reward given a certain state
            # approximate the output based on the input
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
            
    def preprocess(self,state):
        #turn the image into gray then resize it to (84,84) then compress the data to 8bits
        #state=np.uint8(resize(rgb2gray(state),(frame_width,frame_height))*255)
        state=resize(rgb2gray(state),(frame_width,frame_height))
        state=np.uint8(np.reshape(state,(1,frame_width,frame_height))*255)
        return state
            
    def initial_state(self,state0):
        
        state=agent.preprocess(state0)
        
        #save the last 4 states into state
        state=[state for _ in range(4)]
        state=np.stack(state,axis=0)
        state=state.reshape(1,84,84,4)
        return state
    
    def new_state(self,futur_state0,state0,state_1,state_2):     
        state=agent.preprocess(state0)
        futur_state=agent.preprocess(futur_state0)
        futur_state0=np.maximum(state0,futur_state0)
        state_1=agent.preprocess(state_1)
        state_2=agent.preprocess(state_2)
        
      #save the last 4 states into state
        state=[futur_state,state,state_1,state_2]
        
        state=np.stack(state,axis=0)
        state=state.reshape(1,84,84,4)

        return state
    
        
if __name__=="__main__":
    
    #my_memory is the memory in which we store the previous experiences
    #the size of the memory is fixed to 1000 in this case so the first experiencies are removed sequentially
    #we get better results when we remove some of the previous experiences and keep the most recent ones
    my_memory=deque(maxlen=1000)
    #the decision memory is the memory in which we store the last 30 results so we can estimate the average results 
    decision_memory=deque(maxlen=30)
    #data memories are memories in which we store the data to be used for our plots
    data=[]
    data2=[]
    #we decide which environment is to be used, in this case it's Breakout
    env = gym.make('Breakout-v0')
    #the output size is the number of actions we could make to interact with the environment,
    # in this case we have 4 actions
    output_size = env.action_space.n
    #we initialize our agent with the  output sizes
    agent=my_agent(output_size)
    #we initialize done with false, it becomes true when the game ends
    done=False
    #we initialize enough_data with 0, it becomes 1 when we store enough data (data>batch_size) to begin the training
    enough_data=False
    tot_reward=0


    for i in range(nb_games):
        # Obtain an initial observation of the environment
        state0=env.reset()
        state_1=state0
        state_2=state0
        state=agent.initial_state(state0)
        reward_sum=0
        reward_avg=0
        tot_reward=0
        ancien_info={'ale.lives': 5}
        info={'ale.lives': 5}
        #reward=0
        #rate = rate * 0.8
        
        if i>30:
            rate = rate * 0.9
        t=0
        while(1):
            
            reward=0
            env.render()
            #at first our agent act randomly then when it start to learn it acts randomly occasionaly
            if (i<30) or (t==0) or (info!=ancien_info) or (np.random.rand()<rate) :
                action=random.randrange(output_size)
            else:
                action=agent.agent_action(state)
            
            t=t+1 
            
            #reward=t
            
            #action represent either 0 or 1, when we pass it to the env which represents the game environment, 
            #it emits the following 4 outputs
            ancien_info=info
            futur_state0,reward,_,info=env.step(action)

            if info=={'ale.lives': 0}:
                done=True            

            futur_state=agent.new_state(futur_state0,state0,state_1,state_2)
            #tot_reward=tot_reward+reward+1
            if info!=ancien_info:
                if info=={'ale.lives': 4}:
                    tot_reward=tot_reward-1
                
                if info=={'ale.lives': 3}:
                    tot_reward=tot_reward-1

                if info=={'ale.lives': 2}:
                    tot_reward=tot_reward-1

                if info=={'ale.lives': 1}:
                    tot_reward=tot_reward-1
                     
           
            tot_reward=tot_reward+2*reward
 


            if t>batch_size:
                enough_data=1 
            

            #All the actions are stored into a memory, to be used afterwise for the training step
            #my_memory.append((state,action,reward,tot_reward,futur_state,t,done))

            my_memory.append((state,futur_state,action,reward,tot_reward,info,done))
            state_2=state_1
            state_1=state0
            state0=futur_state0
            state=futur_state
            
            if t>batch_size:
                enough_data=True           
            

            
            if done:
                done=False
 
                tot_reward=tot_reward-2

                #tot_reward=round(tot_reward,0)
                print("Game number : {}/{},time: {},total reward:{}," .format(i,nb_games,t,tot_reward),info)
                decision_memory.append(tot_reward)
                data.append(tot_reward)
                if i>100: 
                    copy_mem=copy.deepcopy(data_memory)
                    for j in range(29):
                        reward_sum = reward_sum + copy_mem.pop()
                        reward_avg=reward_sum/30
                        data2.append(reward_avg)
                        #print("average reward:",reward_avg)
                break

        
        
        #If we have enough data we can start the training
        
        if enough_data:
            agent.replay()
        
