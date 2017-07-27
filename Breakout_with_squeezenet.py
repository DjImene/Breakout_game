import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import time
import warnings
warnings.filterwarnings('ignore')
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D,Flatten, MaxPooling2D
from keras.optimizers import Adam
from skimage.color import rgb2gray
from skimage.transform import resize
import copy
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc

#The number of games must be high enough to observe the variation and see if there is a permanent state
nb_games=100000
learning_rate=0.002
#I've tried diffrent batch sizes as 32, 64, 128,... and I've found out that 64 is giving the best results
batch_size=16
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
    def __init__(self, input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.model=self.myNN()

    #Defining the neural network, fixed number of neurones for the input and output layers
    #For different number of neurons and different number of hidden layer, the results are not as good as the ones
    #provided with this specific modal, others activations functions as well as optimizers and losses have been tested
    #better results have been found with the modal bellow
    def myNN(self):

        model = Sequential()
        #model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dense(512, activation='relu',input_shape=(4,1000)))
        model.add(Dense(self.output_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=learning_rate))
        return model
    
    #The action can be either 0 or 1
    def agent_action(self, features):
        predict_val=self.model.predict(features)
        return np.argmax(predict_val[0])



    #This function is used to train the neural network bases on the memory(the previous experiencies or actions it had)
    def replay(self):
        minibatch = random.sample(my_memory,batch_size)
        
        for features,futur_features, action, reward,tot_reward, done in minibatch:
            target = reward
            if not done:
                #target=best_reward
                target = reward + gamma * np.amax(self.model.predict(futur_features)[0])   
            target_f = self.model.predict(features)
            target_f[0][action] = target
            #the neural network will predict the reward given a certain state
            # approximate the output based on the input
            self.model.fit(features, target_f, epochs=1, verbose=0)
            
    def initial_state(self,state0):
        #turn the image into gray then resize it to (84,84) then compress the data to 8bits
        #state0=np.uint8(resize(rgb2gray(state0),(frame_width,frame_height))*255)
        
        
        sq_model = SqueezeNet()
        start = time.time()
        #img = state0
        img = image.load_img('/home/i16djell/Bureau/state0.jpg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = sq_model.predict(x)
        
        features=[feature ,feature ,feature ,feature ]
        features=np.stack(features,axis=0)
        features=features.reshape(1,4,1000)
        return features
    
    def new_state(self,futur_state0,state0,state_1,state_2):
        #state0=np.uint8(resize(rgb2gray(state0),(frame_width,frame_height))*255)
        #futur_state0=np.uint8(resize(rgb2gray(futur_state0),(frame_width,frame_height))*255)
        #state0=np.maximum(state0,futur_state0)
        #state_1=np.uint8(resize(rgb2gray(state_1),(frame_width,frame_height))*255)
        #state_2=np.uint8(resize(rgb2gray(state_2),(frame_width,frame_height))*255)
        
        sq_model = SqueezeNet()
        start = time.time()
        #img = state_2
        img = image.load_img('/home/i16djell/Bureau/state_2.jpg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature1 = sq_model.predict(x)
        
        #img = state_1
        img = image.load_img('/home/i16djell/Bureau/state_1.jpg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature2 = sq_model.predict(x)
        
        #img = state0
        img = image.load_img('/home/i16djell/Bureau/state0.jpg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature3 = sq_model.predict(x)  
        
        #img = futur_state0
        img = image.load_img('/home/i16djell/Bureau/futur_state0.jpg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature4 = sq_model.predict(x)
        
        features=[feature4,feature3,feature2,feature1]
        features=np.stack(features,axis=0)
        features=features.reshape(1,4,1000)
        
        return features
        

if __name__ == '__main__':
    
    
        #my_memory is the memory in which we store the previous experiences
    #the size of the memory is fixed to 1000 in this case so the first experiencies are removed sequentially
    #we get better results when we remove some of the previous experiences and keep the most recent ones
    my_memory=deque(maxlen=1000)
    #the decision memory is the memory in which we store the last 100 results so we can decide if the average of those 100
    #exepriences is good enough to consider that our agent has learned
    decision_memory=deque(maxlen=30)
    #data memories are memories in which we store the data to be used for our plots
    data=[]
    data2=[]
    #we decide which environment is to be used, in this case it's CartPole
    env = gym.make('Breakout-v0')
    #the input_size is the number of observations we get from our environment, in this case it's 4
    #cart position, cart velocity, pole angle, pole velocity
    input_size = env.observation_space.shape[0]
    #the output size is the number of actions we could make to interact with the environment,
    # in this case we have 4 actions, moving right or moving left
    output_size = env.action_space.n
    #we initialize our agent with the input and output sizes
    agent=my_agent(input_size,output_size)
    #we initialize done with false, it becomes true when the game ends
    done=False
    #we initialize enough_data with 0, it becomes 1 when we store enough data (data>batch_size) to begin the training
    enough_data=0
    tot_reward=0


    for i in range(nb_games):
        # Obtain an initial observation of the environment
        state0=env.reset()
        state_1=state0
        state_2=state0
        scipy.misc.imsave('/home/i16djell/Bureau/state_1.jpg',state_1)
        scipy.misc.imsave('/home/i16djell/Bureau/state_2.jpg',state_2)
        scipy.misc.imsave('/home/i16djell/Bureau/state0.jpg',state0)
        features=agent.initial_state(state0)
        reward_sum=0
        reward_avg=0
        tot_reward=0
        ancien_info={'ale.lives': 5}
        info={'ale.lives': 5}
        #reward=0
        
        if i>100:
            rate = rate * 0.8
        
        for t in range(1000000000000):
            reward=0
            env.render()
            #at first our agent act randomly then when it start to learn it acts randomly occasionaly
            if(i<30) or (np.random.rand()<rate) or (t==0) or (info!=ancien_info):
                action=random.randrange(output_size)
            else:
                action=agent.agent_action(features)
            
              
            
            #reward=t
            
            #action represent either 0 or 1, when we pass it to the env which represents the game environment, 
            #it emits the following 4 outputs
            ancien_info=info
            futur_state0,reward,done,info=env.step(action)
            scipy.misc.imsave('/home/i16djell/Bureau/futur_state0.jpg',futur_state0)
            futur_features=agent.new_state(futur_state0,state0,state_1,state_2)
            #tot_reward=tot_reward+reward+1
            if info!=ancien_info:
                if info=={'ale.lives': 4}:
                    tot_reward=tot_reward-1
                if info=={'ale.lives': 3}:
                    tot_reward=tot_reward-2
                if info=={'ale.lives': 2}:
                    tot_reward=tot_reward-3
                if info=={'ale.lives': 1}:
                    tot_reward=tot_reward-4
                     
                        
            tot_reward=tot_reward+reward
            
            #futur_state=np.uint8(resize(rgb2gray(futur_state),(frame_width,frame_height))*255)
            #futur_state=np.reshape(futur_state,(1,frame_width,frame_height))
            #futur_state=[futur_state for _ in range(4)]
            #futur_state=np.stack(futur_state,axis=0)
            #futur_state=futur_state.reshape(1,84,84,4)

            if t>batch_size:
                enough_data=1 
            

            #All the actions are stored into a memory, to be used afterwise for the training step
            #my_memory.append((state,action,reward,tot_reward,futur_state,t,done))
            my_memory.append((features,futur_features,action,reward,tot_reward,done))
            state_2=state_1
            state_1=state0
            state0=futur_state0
            scipy.misc.imsave('/home/i16djell/Bureau/state_1.jpg',state_1)
            scipy.misc.imsave('/home/i16djell/Bureau/state_2.jpg',state_2)
            scipy.misc.imsave('/home/i16djell/Bureau/state0.jpg',state0)
            features=futur_features
            if t>batch_size:
                enough_data=1             
            
            if done:
                tot_reward=tot_reward-6
                print("Game number : {}/{},time: {},total reward:{}" .format(i,nb_games,t,tot_reward))
                decision_memory.append(tot_reward)
                data.append(t)
                if i>100: 
                    copy_mem=copy.deepcopy(decision_memory)
                    for j in range(29):
                        reward_sum = reward_sum + copy_mem.pop()
                        reward_avg=reward_sum/30
                        data2.append(reward_avg)
                        #print("average reward:",reward_avg)
                break

        
        

            
            #if reward_avg >= 195.0:
                #print("\n Problem solved, average reward :", reward_avg)
                #break
                

        
        
        #If we have enough data we can start the training
        
        if enough_data==1:
            agent.replay()
        
    env.monitor.close()
    plt.plot(data2,'g')
    plt.xlabel('Game number +100')
    plt.ylabel('Average game score')
    plt.title('score variation')
    plt.show()





