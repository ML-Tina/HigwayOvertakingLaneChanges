# import numpy as np
# import gym
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam
# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory
# import Data
# import Environment.environment
# import Environment.environment.envs
#
# ENV_NAME = 'highway-v0'
# env = gym.make(ENV_NAME)
# np.random.seed(24)
# env.seed(42)
# nb_actions = env.action_space.n
# print('Actions possible :', nb_actions)
# model= Sequential()
# model.add(Dense(24, input_dim=env.observation_space.n,activation='relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
# print(model.summary())
#
# policy = EpsGreedyQPolicy()
# memory = SequentialMemory(limit=50000, window_length=1)
# dqn= DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#
# dqn.fit(env, nb_steps=500, visualize=True, verbose=2)
#
# dqn.test(env,nb_episodes=5,visualize=True)

import numpy as np
import random as rn
import tensorflow as tf
import math
import os
import scipy.stats
# Setting the seeds to get reproducible results
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
np.random.seed(42)
rn.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
from keras import backend as keras
keras.set_session(sess)

import gym
from collections import deque
from keras.models import Sequential
from keras import backend
from keras.backend import eval
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import initializers
import matplotlib.pyplot as plt
import pickle
import Environment.environment
import Environment.environment.envs
from Data.RouteCreator import RouteCreator

""""
File is based on the tutorial of 
@url{https://keon.io/deep-q-learning/}
"""

# constant values
EPISODES    = 10000
BATCH_SIZE  = 32
MAX_STEPS   = 100
FilePathLog = os.path.dirname(os.path.realpath(__file__)) + '/OutputLog.out'

def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    # convert to percentage
    sma = [i * 100 for i in sma]
    return sma

class DQNAgent:

    def __init__(self, alpha):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9 #0.9  # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999#0.99992 # so that the exploration reaches minimum by 27466 episodes
        self.learning_rate = alpha#0.001 # as per ADAM optimization technique
        self.learning_rate_decay= 0.5
        self.learning_rate_const = 0.01
        self.model = self._build_model()

    def relu_advanced(self, x):
        return backend.relu(x, alpha=0.2)

    # Building neural Net for Deep-Q learning Model
    def _build_model(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size,
                        activation='relu',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0.1)))
        model.add(Dense(100,
                        activation='relu',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0.1)))
        # model.add(Dense(50,
        #                 activation=self.'relu',
        #                 kernel_initializer=initializers.glorot_normal(seed=1337),
        #                 bias_initializer=initializers.Constant(value=0.1)))
        model.add(Dense(self.action_size,
                        activation='linear',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0.1)))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, episode, state, action, reward, next_state, done):
        self.memory.append((episode, state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if np.random.rand() <= self.epsilon and use_epsilon:
           # print('action taken on random not based on Q values')
            return rn.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, episodeNumber):
        minibatch = rn.sample(self.memory, batch_size)
        squareError = 0
        for _, state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            # networkQValue = target_f[0][action]
            # actualQValue = target
            # squareError = squareError + np.square(networkQValue-actualQValue)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            squareError += history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # #Change Learning rate
        # self.learning_rate = self.learning_rate_const/(1+(self.learning_rate_decay*episodeNumber))
        # keras.set_value(self.model.optimizer.lr,self.learning_rate)
        # print("New Learning rate {}" .format(eval(self.model.optimizer.lr)))
        # Return RMS error
        rmsError = squareError/batch_size
        # rmsError = np.sqrt(squareError)/batch_size
        return rmsError

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def trainOrTest(batch_size, episodes, training):
    # List to store the completion status
    TestCompletionStatus = []
    vehPosArray = []
    NonCollisionCountAvg = []
    NonCollisionCount = 0
    CollisionCountAvg = []
    CollisionCountList= []
    CollisionCount = 0
    VehSpeedArray = []
    rewardarray = []
    collisionindexarray = []
    avgSpeed = 0
    unsuitableepisodes = 0  # episodes that cannot be taken into account for performance testing. Ex when an accident occurs at the start itself
    RMSErrorList = []
    NumberOfLaneChangelist=[]
    NumberOfOvertakesList =[]
    FilePathSaveData = os.path.dirname(os.path.realpath(__file__)) + '/Result/SavedResults'
    print("File Path for the saved results" + FilePathSaveData)
    for e in range(episodes):
        tempNumberofOvertakes = 0
        tempNumberofLaneChanges = 0
        # Set If the episodes are for training or not
        env.TrainingStatus(training)
        # reset the env for a new episode
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        VehSpeedArray.clear()
        #Clear the reward array
        # Step through the episode until MAX_STEPS is reached
        for steps in range(MAX_STEPS):
            action = agent.act(state, use_epsilon=training)
            next_state, reward, done, vehpos, tempNumberofLaneChanges, tempNumberofOvertakes, _ = env.step(action)
            dummydone = done  # keep a copy of whether collision has occured as it is also used for finding the last step for replay
            if steps == MAX_STEPS -1:
                done = True
            # if steps == 0:
            #     rewardarray.append(reward)
            # else:
            #     rewardarray.append(reward + rewardarray[steps-1])
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(e, state, action, reward, next_state, done)
            state = next_state
            done = dummydone # as done is also used to find if collision has occured
            # save the vehicle velocity to plot
            VehSpeedArray.append(next_state[0][0])
            if done:
                avgSpeed = -1
                if steps > 3: # check if the accident has occured as soon as the vehicle is inserted
                    TestCompletionStatus.append(0) # 0 means ended because of collision
                    #Update RewardArray
                    totalReward = sum([x[3] for x in agent.memory if x[0] == e])
                    rewardarray.append(totalReward)
                    collisionindexarray.append(e - unsuitableepisodes)
                    CollisionCount += 1
                    #Append the number of lane changes
                    NumberOfLaneChangelist.append(tempNumberofLaneChanges)
                    # append the distance the vehicle travelled
                    vehPosArray.append(vehpos)
                    # Find average number of non collisions with increase in episodes
                    if (e + 1 - unsuitableepisodes != 0):
                        NonCollisionCountAvg.append((NonCollisionCount / (e + 1 - unsuitableepisodes)) * 100)
                        CollisionCountAvg.append((CollisionCount / (e + 1 - unsuitableepisodes)) * 100)
                        CollisionCountList.append(CollisionCount)
                else:
                    #vehPosArray.append(vehpos)
                    unsuitableepisodes += 1
                break
            if steps == MAX_STEPS-1:
                TestCompletionStatus.append(1) # ended because no of steps have been executed
                totalReward = sum([x[3] for x in agent.memory if x[0] == e])
                rewardarray.append(totalReward)
                NonCollisionCount += 1
                # Append number of lane changes
                NumberOfLaneChangelist.append(tempNumberofLaneChanges)
                # append the distance the vehicle travelled
                vehPosArray.append(vehpos)
                avgSpeed = vehpos/steps
                # Find average number of non collisions with increase in episodes
                if (e + 1 - unsuitableepisodes != 0):
                    NonCollisionCountAvg.append((NonCollisionCount / (e + 1 - unsuitableepisodes)) * 100)
                    CollisionCountAvg.append((CollisionCount / (e + 1 - unsuitableepisodes)) * 100)
                    CollisionCountList.append(CollisionCount)
        # if e % 1000 == 0:
        #     ax = plt.axes()
        #     ax.xaxis.set_major_locator(plt.MultipleLocator(10.0))
        #     ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        #     ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        #     ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        #     ax.plot(VehSpeedArray, 'r')
        #     ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
        #     ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
        #     ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
        #     ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
        #     ax.set_xlabel('time steps')
        #     ax.set_ylabel('Speed in m/s')
        #     plt.show()
        #Save the Total Overtakes in each episode
        NumberOfOvertakesList.append(tempNumberofOvertakes)

        # # Find average number of non collisions with increase in episodes
        # if (e + 1 - unsuitableepisodes != 0):
        #     NonCollisionCountAvg.append((NonCollisionCount / (e + 1 - unsuitableepisodes)) * 100)
        #     CollisionCountAvg.append((CollisionCount / (e + 1 - unsuitableepisodes)) * 100)

        # plot the reward updation for the final simulation
        # if training == False:
        #     plt.plot(rewardarray, 'r')
        #     plt.xlabel('Steps')
        #     plt.ylabel('Accumulated Reward')
        #     plt.show()
        # if training == False:
        #     plt.plot(VehSpeedArray, 'r')
        #     plt.xlabel('time steps')
        #     plt.ylabel('Speed in m/s')
        #     plt.show()
        # Start experience replay if the agent.memory > batch_size
        rmstemp = -1 # to avoid error while printing the output if the memory is less than batch size
        if len(agent.memory) > batch_size and training:
           rmstemp = agent.replay(batch_size,e)
           RMSErrorList.append(rmstemp)
        # print statistics of this episode
        total_reward = sum([x[3] for x in agent.memory if x[0] == e])
        with open(FilePathLog, "a") as fp:
            print("episode: {}/{}, total reward:: {} Current Exploration Rate: {} Avg Speed: {}m/s Collision Count: {} Unsuitable Episodes {} MinTotalOverakes: {} MeanSquaredError: {} Learning Rate: {}".format(e+1, episodes, total_reward, agent.epsilon,avgSpeed, CollisionCount, unsuitableepisodes, tempNumberofOvertakes, rmstemp, agent.learning_rate), file=fp)
        print("episode: {}/{}, total reward:: {} Current Exploration Rate: {} Avg Speed: {}m/s Collision Count: {} Unsuitable Episodes {} MinTotalOverakes: {} MeanSquaredError: {} Number of Lane Changes {} Learning rate {}"
              .format(e+1, episodes, total_reward, agent.epsilon,avgSpeed, CollisionCount, unsuitableepisodes, tempNumberofOvertakes, rmstemp, tempNumberofLaneChanges, agent.learning_rate))
        #Close the SUMO application
        #env.close()
    # Plot the number of overtakes
    ax = plt.axes()
    # x = list(range(1, len(TrainRewards) + 1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(500.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.plot(NumberOfOvertakesList)
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Minimum Number of overtakes')
    plt.show()
    #Plot RMS Error
    ax = plt.axes()
    # x = list(range(1, len(TrainRewards) + 1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(200.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.plot(RMSErrorList)
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('RMS Error')
    plt.show()
    #Plot number of lane changes
    ax = plt.axes()
    # x = list(range(1, len(TrainRewards) + 1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(200.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    # ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.plot(NumberOfLaneChangelist)
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Total Number of lane changes')
    plt.show()
    # Find the Propability of collision based on Clopper Pearson Binomial confidence method
    alpha = 0.05  # Percent chance of making type 1 error
    plo = scipy.stats.beta.ppf(alpha / 2, CollisionCount, (episodes - unsuitableepisodes - CollisionCount + 1))
    phi = scipy.stats.beta.ppf(1 -  alpha / 2, CollisionCount + 1, (episodes - unsuitableepisodes - CollisionCount))
    print("Propability of collision : \n LowerBound : {}\n UpperBound : {}" .format(plo, phi))
    # if training == True:
    #     weightFile.close()
    # Plot results
    plt.plot(TestCompletionStatus, 'ro')
    plt.xlabel('Episodes')
    plt.show()
    tempcollsisonarray = [0] * len(collisionindexarray)
    plt.plot(collisionindexarray, tempcollsisonarray, 'ro')
    plt.xlabel('Episodes')
    plt.ylabel('Simulation Collision Status 1- No collision 0- collision')
    plt.plot(rewardarray,'b')
    plt.ylabel('Total Reward')
    plt.show()
    # # Plot the average results
    # plt.plot(NonCollisionCountAvg, 'r')
    # plt.xlabel('Episodes')
    # plt.ylabel('Average Non collisions')
    # plt.show()

    #FilePathSaveData = os.path.dirname(__file__)+'/Result/SavedResults'
    if training:
        FilePathSaveDataTraining = FilePathSaveData + '/Train3laneinputacc1'+agent.learning_rate
        FilePathSaveDataTrainingDist = FilePathSaveData + '/TrainDist3laneinputacc1'+agent.learning_rate
        FilePathSaveDataTrainingRewards = FilePathSaveData + '/TrainRewards3laneinputacc1'+agent.learning_rate
        FilePathSaveDataTrainingError = FilePathSaveData + '/TrainRMSError'+agent.learning_rate
        with open(FilePathSaveDataTraining, "wb") as fp:
            pickle.dump(CollisionCountAvg, fp)
        with open(FilePathSaveDataTrainingDist, "wb") as fp:
            pickle.dump(vehPosArray, fp)
        with open(FilePathSaveDataTrainingRewards, "wb") as fp:
            pickle.dump(rewardarray, fp)
        with open(FilePathSaveDataTrainingError, 'wb') as fp:
            pickle.dump(RMSErrorList, fp)
    # else:
    #     FilePathSaveDataTest = FilePathSaveData + '\Test3laneinputacc1'
    #     FilePathSaveDataTestDist = FilePathSaveData + '\TestDist3laneinputacc1'
    #     FilePathSaveDataTestRewards = FilePathSaveData + '\TestRewards3laneinputacc1'
    #     with open(FilePathSaveDataTest, "wb") as fp:
    #         pickle.dump(CollisionCountAvg, fp)
    #     with open(FilePathSaveDataTestDist, "wb") as fp:
    #         pickle.dump(vehPosArray, fp)
    #     with open(FilePathSaveDataTestRewards, "wb") as fp:
    #         pickle.dump(rewardarray, fp)

    plt.plot(CollisionCountAvg, 'b', label='Average number of Collisions')
    plt.plot(CollisionCountList,'r', label='Collision Count')
    plt.xlabel('Episodes')
    plt.ylabel('Average collisions and Number of collisions')
    plt.legend()
    plt.show()

    # plot the distance travelled by the vehicle in each episode
    plt.plot(vehPosArray, 'b')
    plt.xlabel('Episodes')
    plt.ylabel('Distance travelled in meters')
    plt.show()

    #Plot a histogram based on distance travelled
    validPosArray = list(filter(lambda a: a != 0, vehPosArray))
    mean = np.mean(validPosArray)
    sd = np.std(validPosArray)
    plt.hist(validPosArray)
    plt.xlabel('Distance (Meters)')
    plt.ylabel('Frequency')
    plt.title('Histogram Distance Travelled, Mean %g, SD %g' % (mean, sd))
    plt.show()
    # plot the speed if training is false

    #Plot Moving Average for training data
    if training == True:
        movingAverage = movingaverage(TestCompletionStatus, 100)
        plt.plot(movingAverage, 'r')
        plt.xlabel('Episodes')
        plt.ylabel('Average Non collisions')
        plt.show()

if __name__ == "__main__":
    env = gym.make('highway-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    for i in range(20):
        r = np.random.rand()
        alpha = np.power(10,r)
        agent = DQNAgent(alpha)
        #agent.load(os.path.dirname(os.path.realpath(__file__))+'/KearasModels/Model3LinAcc0.510543374318792')
        # Create a log file to print the outputs
        with open(FilePathLog, "w") as fp:
            print("Outputs", file=fp)
        env.test = True
        env.log = False
        env.test = False
        env.start(gui=False)
        trainOrTest(BATCH_SIZE, EPISODES, training=True)
        env.close()
        env.log = True
        agent.save(os.path.dirname(os.path.realpath(__file__))+'/KearasModels/Model3LinAcc'+agent.learning_rate)
        env.test = True
        env.start(gui=False)
        trainOrTest(BATCH_SIZE, episodes=2000, training=False)

        agent.save('model')
       # plot_model(agent.model, show_shapes=True)

        env.close()
