from __future__ import absolute_import
from __future__ import print_function

import logging

import gym
import numpy as np
from gym import spaces
import random as rn

import os
import sys

# Setting the seeds to get reproducible results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
import traci.constants as tc

logger = logging.getLogger(__name__)

gui = True
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')
config_path = os.path.dirname(__file__)+"/../../../Data/StraightRoad.sumocfg"

class HighwayEnv(gym.Env):
    metadata = {'render.modes':['human']}

    # return -1 if there is no vehicle behind in lane 0. Currently all other vehicles will be in lane 0
    # also return the distance to the vehicle in front in lane 0
    # Values required from the function are:
    # d1 -> distance from veh in front in same lane
    # v1 -> velocity of veh in front in same lane
    # d2 -> distance from veh behind in same lane
    # v2 -> velocity of veh behind in same lane
    # d3 -> distance from veh in front in left lane
    # v3 -> velocity of veh in front in left lane
    # d4 -> distance from veh behind in left lane
    # v4 -> velocity of veh behind in left lane
    # v5 -> velocity of veh in front in right lane
    # d5 -> distance from vehicle in front in right lane
    # v6 -> velocity of vehicle behind in right lane
    # d6 -> velocity of vehicle behind in right lane
    def _findRearVehDistance(self, vehicleparameters):
        parameters = [[0 for x in range(3)] for x in range(len(vehicleparameters))]
        i = 0
        d1 = -1
        d2 = -1
        d3 = -1
        d4 = -1
        d5 = -1
        d6 = -1
        v1 = -1
        v2 = -1
        v3 = -1
        v4 = -1
        v5 = -1
        v6 = -1
        for VehID in self.VehicleIds:
            parameters[i][0] = VehID
            parameters[i][1] = vehicleparameters[VehID][tc.VAR_POSITION][0]  # X position
            parameters[i][2] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # lane Index
            i = i + 1
        parameters = sorted(parameters, key=lambda x: x[1])  # Sorted in ascending order based on x distance
        # Find Row with Auto Car
        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        # if there are no vehicles in front
        if RowIDAuto == len(self.VehicleIds)-1:
            d1 = -1
            v1 = -1
            d3 = -1
            v3 = -1
            d5 = -1
            v5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 100
            #Check if an overtake has happend
            if(self.currentTrackingVehId !='None' and (vehicleparameters[self.currentTrackingVehId][tc.VAR_POSITION][0] < vehicleparameters[self.AutoCarID][tc.VAR_POSITION][0])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = 'None'
        else:
            # If vehicle is in the lowest lane, then d5,d6,v5,v6 do not exist
            if parameters[RowIDAuto][2] == 0:
                d5 = -1
                v5 = -1
                d6 = -1
                v6 = -1
            # if the vehicle is in the maximum lane index, then d3.d4.v3.v4 do not exist
            elif parameters[RowIDAuto][2] == (self.maxLaneNumber - 1):
                d3 = -1
                v3 = -1
                d4 = -1
                v4 = -1
            # find d1 and v1
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d1 = parameters[index][1] - parameters[RowIDAuto][1]
                    v1 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d1 = -1
                v1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 100
            # find d3 and v3
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d3 = parameters[index][1] - parameters[RowIDAuto][1]
                    v3 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d3 = -1
                v3 = -1
            # find d5 and v5
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d5 = parameters[index][1] - parameters[RowIDAuto][1]
                    v5 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d5 = -1
                v5 = -1
            # find d2 and v2
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d2 = parameters[RowIDAuto][1] - parameters[index][1]
                    v2 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d2 = -1
                v2 = -1
            # find d4 and v4
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d4 = parameters[RowIDAuto][1] - parameters[index][1]
                    v4 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d4 = -1
                v4 = -1
            # find d6 and v6
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d6 = parameters[RowIDAuto][1] - parameters[index][1]
                    v6 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d6 = -1
                v6 = -1
            # Find if any overtakes has happend
            if (self.currentTrackingVehId != 'None' and (vehicleparameters[self.currentTrackingVehId][tc.VAR_POSITION][0] <vehicleparameters[self.AutoCarID][tc.VAR_POSITION][0])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0: #This means that there is no car behind
            RearDist = -1
        else: # There is a car behind return the distance between them
            RearDist =  (parameters[RowIDAuto][1] - parameters[RowIDAuto-1][1])
        # Return car in front distance
        if RowIDAuto == len(self.VehicleIds)-1:
            FrontDist = -1
            # Save the current front vehicle Features
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 100
        else:
            FrontDist = (parameters[RowIDAuto+1][1] - parameters[RowIDAuto][1])
            # Save the current front vehicle Features
            self.CurrFrontVehID = parameters[RowIDAuto+1][0]
            self.CurrFrontVehDistance = FrontDist
        #return RearDist, FrontDist
        return d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6


    def _findstate(self):
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        # find d1,v1,d2,v2,d3,v3,d4,v4, d5, v5, d6, v6
        d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6 = self._findRearVehDistance(VehicleParameters)
        # For Fault Simulation use random generation to generate if a communication fault should occur
        commErrorStatus = False
        if np.random.rand() < self.ErrorPropability:
            commErrorStatus = True
        #check if they are between the limits if they are not then give the maximum possible
        if ((d1 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d1 <= self.CommRange))):
            d1 = self.maxDistanceFrontVeh
        elif d1 < 0: # if there is no vehicle ahead in L0
            d1 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v1 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d1 <= self.CommRange))) : # there is no vehicle ahead in L0 or there is a communication error: # there is no vehicle ahead in L0
            v1 = 0

        if ((d2 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d2 <= self.CommRange))):
            d2 = self.maxDistanceRearVeh
        elif d2 < 0: #There is no vehicle behind in L0
            d2 = 0 # to avoid negetive reward
        if ((v2 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d2 <= self.CommRange))) : # there is no vehicle behind in L0 or there is a communication error
            v2 = 0

        if ((d3 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d3 <= self.CommRange))):
            d3 = self.maxDistanceFrontVeh
        elif d3 < 0: # no vehicle ahead in L1
            d3 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v3 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d3 <= self.CommRange))) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v3 = 0

        if ((d4 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d4 <= self.CommRange))):
            d4 = self.maxDistanceRearVeh
        elif d4 < 0: #There is no vehicle behind in L1
            d4 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v4 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d4 <= self.CommRange))) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v4 = 0

        if ((d5 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d5 <= self.CommRange))):
            d5 = self.maxDistanceFrontVeh
        elif d5 < 0: # no vehicle ahead in L1
            d5 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v5 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d5 <= self.CommRange))) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v5 = 0

        if ((d6 > self.CommRange) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d6 <= self.CommRange))):
            d6 = self.maxDistanceRearVeh
        elif d6 < 0: #There is no vehicle behind in L1
            d6 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v6 < 0) or ((commErrorStatus == True) and (self.DisableFaultSimulation == False) and (d6 <= self.CommRange))) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v6 = 0
        va = VehicleParameters[self.AutoCarID][tc.VAR_SPEED]
        #Vehicle acceleration rate
        vacc = va - self.PrevSpeed # as the time step is 1sec long
        # Distance Travelled
        DistanceCovered = VehicleParameters[self.AutoCarID][tc.VAR_DISTANCE]
        return va, v1, d1, v2, d2, v3, d3, v4, d4, v5, d5, v6, d6, VehicleParameters[self.AutoCarID][tc.VAR_LANE_INDEX],vacc, DistanceCovered



    def __init__(self):
        #Observation states. This includes Distance with vehicle in front, with vehicle in back, velocity of vehicle
        # in front, velocity of self driving vehicle
        self.minAutoVelocity = 0
        self.maxAutoVelocity = 30

        self.minOtherVehVelocity = 0
        self.maxOtherVehVelocity = 20

        self.minDistanceFrontVeh = 0
        self.maxDistanceFrontVeh = 100

        self.minDistanceRearVeh = 0
        self.maxDistanceRearVeh = 100

        self.minLaneNumber = 0
        self.maxLaneNumber = 3

        self.maxAcceleration = 30
        self.minAcceleration = -30

        self.maxTotalDistanceCovered = 40000
        self.minTotalDistanceCovered = -1

        # State array order, va,v1,d1,v2,d2,v3,d3,v4,d4,v5,d5,v6,d6, Lane number, Vacc, DistCovered
        high = np.array([self.maxAutoVelocity, self.maxOtherVehVelocity, self.maxDistanceFrontVeh, self.maxOtherVehVelocity, self.maxDistanceRearVeh, self.maxOtherVehVelocity, self.maxDistanceFrontVeh, self.maxOtherVehVelocity, self.maxDistanceRearVeh, self.maxOtherVehVelocity, self.maxDistanceFrontVeh, self.maxOtherVehVelocity, self.maxDistanceRearVeh, self.maxLaneNumber, self.maxAcceleration, self.maxTotalDistanceCovered])
        low = np.array([self.minAutoVelocity, self.minOtherVehVelocity, self.minDistanceFrontVeh, self.minOtherVehVelocity, self.minDistanceRearVeh, self.minOtherVehVelocity, self.minDistanceFrontVeh, self.minOtherVehVelocity, self.minDistanceRearVeh, self.minOtherVehVelocity, self.minDistanceFrontVeh, self.minOtherVehVelocity, self.minDistanceRearVeh, self.minLaneNumber, self.minAcceleration, self.minTotalDistanceCovered])
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(3)
        self.viewer = None
        self.state = None
        self.log = False
        self.result = []
        self.run = []
        self.test = False
        self.VehicleIds=[]
        self.AutoCarID='Auto'
        self.end = False
        # front vehicle characteristics previous step values
        self.PrevFrontVehID = 'None'
        self.PrevFrontVehDistance = 100
        self.PrevSpeed = 0
        # Current Vehicle characterisitic of vehicle in front
        self.CurrFrontVehID = 'None'
        self.CurrFrontVehDistance = 100
        self.StartTime = 0
        self.TotalReward = 0
        self.AutocarSpeed = 0
        self.AccRate = 2
        self.DecRate = 1.5
        # Fault Simulation
        self.CommRange = 100
        self.DisableFaultSimulation = True
        self.ErrorPropability = 0
        self.numberOfLaneChanges = 0
        self.numberOfOvertakes = 0
        self.currentTrackingVehId = 'None'
        self.prevDistanceCovered = 0
    def step(self,action):
        reward=0
        # position of autonomous vehicle
        posAutox = -1 # if the vehicle is not available
        # action 0 -> move to right lane
        if action == 0:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            if laneindex != 0:
                traci.vehicle.changeLane(self.AutoCarID, laneindex - 1, 100)
                self.numberOfLaneChanges += 1
            #traci.vehicle.setSpeed(self.AutoCarID, traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_SPEED]) # maintain the same speed
        # action 1 change to left Lane 1
        elif action == 1:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            if laneindex != self.maxLaneNumber:
                traci.vehicle.changeLane(self.AutoCarID, laneindex + 1, 100)
                self.numberOfLaneChanges += 1
            #traci.vehicle.setSpeed(self.AutoCarID, traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_SPEED])# maintain the same speed
        #Accelerate Use the slow down command. try to reach maximum speed and vary the time to get the desired acceleration
        # fomula (v2 - v1)/ t, t should be in msec
        # elif action == 2:
        #     maxSpeed = traci.vehicletype.getMaxSpeed(self.AutoCarID)
        #     time = int((maxSpeed - traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_SPEED])/ self.AccRate)
        #     #timems = int(time * 1000)
        #     traci.vehicle.slowDown(self.AutoCarID, maxSpeed, time)
        #     #self.AccRate = self.AccRate + 1.26
        #     #self.DecRate = 0.63
        # # Decrease Speed
        # elif action == 3:
        #     time = ((traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_SPEED]) - 0)/self.DecRate #((traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_SPEED]) - 5)) / self.DecRate
        #     #timems = int(time * 1000)
        #     traci.vehicle.slowDown(self.AutoCarID, 0, time)
        #     #self.DecRate = self.DecRate + 0.63
        #    # self.AccRate = 1.26
       # else:
            # traci.vehicle.setSpeed(self.AutoCarID, traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_SPEED])

        # if action is 4 then it needs to maintain the same state, so no need to change anything
        traci.simulationStep()
        self.VehicleIds = traci.vehicle.getIDList()
        # Get the vehicle parameters after the step
        # if Auto is not present, then it means a collision has occured and it has got teleported
        if self.AutoCarID in traci.vehicle.getIDList():
            # Subscribe for vehicle data as new vehicles may have come into existence
            for VehId in self.VehicleIds:
                traci.vehicle.subscribe(VehId, (tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_LANE_INDEX, tc.VAR_DISTANCE))
                traci.vehicle.subscribeLeader(self.AutoCarID,
                                              50)  # Subscribe the vehicle information of the car in front of Auto
                # traci.vehicle.setSpeedMode(VehId, 0)  # Disable Car Following
            if VehId == self.AutoCarID:
                speedMode = traci.vehicle.getSpeedMode(self.AutoCarID)
                speedMode = speedMode & int('11000', 2)
            Vehicle_Params = traci.vehicle.getAllSubscriptionResults()
            self.AutocarSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
            posAutox = Vehicle_Params[self.AutoCarID][tc.VAR_POSITION][0]
            # Find the new state we are in
            self.state = self._findstate()
            reward = 0
            # Calculate the Rewards
            allowedSpeed = traci.lane.getMaxSpeed('Lane_0') # allowed speedlimit in the lane
            # if (self.state[0] >= 0) and (self.state[0] <= 2):
            #     reward = -0.5#-75
            # # Give negetive reward for high deceleration rate
            # elif np.abs(self.state[0] - self.PrevSpeed) > self.DecRate:
            #     reward = -0.5#-20
            # # give negetive reward if the speed exceeds the speed limit in the lane
            # elif self.state[0] > allowedSpeed:
            #     reward = -0.1#-0.2 * (self.state[0] - allowedSpeed)
            # Condition 1 -> Vehicle is close to vehicle in front dist 20 and it is not in the max lane
            if (self.state[2] < 20) and (Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] != self.maxLaneNumber):
                reward = -0.25#-10#-5#-0.5
            elif (self.state[10] < 20) and (Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] != 0) and (self.state[0] >= self.PrevSpeed):
                reward = 0.2#60 - self.state[10]
            # Condition 2 -> Reward for being in overtaking lane and if the vehicle in front is more than 20m away
            elif (Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] != 0) and (self.state[10] > 20):
                reward = -0.2#-1.5 * (self.state[10])
                #reward = -1
            elif (self.state[2] < 20) and (Vehicle_Params[self.AutoCarID][tc.VAR_LANE_INDEX] == self.maxLaneNumber):
                if self.state[0] < self.PrevSpeed:
                    reward = 0.1#0.5
                else:
                    reward = -0.1#-0.5
            # Condition 4-> Reward based on Speeding until the speed limit
            else:
                reward = 0.1
                # if (self.state[0] == allowedSpeed): #or (self.state[0] > self.PrevSpeed):
                #     # reward = 75
                #     reward = 0.2#2
                # elif (self.state[0] > self.PrevSpeed):
                #     reward = 0.1#1
                # else: # give reward based on total distance travelled until now
                #     reward = 0#self.state[15]/10

            self.PrevFrontVehID = self.CurrFrontVehID
            self.PrevFrontVehDistance = self.CurrFrontVehDistance
            self.PrevSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
           # print('Action taken %d, Reward is %f' %(action, reward))
            DistanceTravelled = Vehicle_Params[self.AutoCarID][tc.VAR_DISTANCE]
        # Condition 4 -> Reward for Collision
        else:
            # change Distance covered in the input state to -1 to so that the learning algorithm can learn that a collision has occured
            self.state = [self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.state[8], self.state[9], self.state[10], self.state[11], self.state[12], self.state[13], self.state[14], -1 ]
            reward = -1#-101
            self.end = True
            print('Collision has occured')
            DistanceTravelled = 0
        self.TotalReward += reward
        # DistanceTravelled = Vehicle_Params[self.AutoCarID][tc.VAR_DISTANCE]
        return self.state, reward, self.end, DistanceTravelled,self.numberOfLaneChanges, self.numberOfOvertakes, {}

    def reset(self):
        self.end = False
        self.TotalReward = 0
        self.numberOfLaneChanges = 0
        self.numberOfOvertakes = 0
        self.currentTrackingVehId = 'None'
        if self.test and len(self.run) != 0:
            self.result.append(list(self.run))
            self.run.clear()
        traci.load(["-c", config_path])
        print('Resetting the layout')
        traci.simulationStep()
        AutoCarAvailable = False
        # lanes = traci.lane.getIDList()
        while AutoCarAvailable == False:
            traci.simulationStep()
            self.VehicleIds = traci.vehicle.getIDList()
            if self.AutoCarID in traci.vehicle.getIDList():
                AutoCarAvailable = True
                self.StartTime = traci.simulation.getCurrentTime()
        self.VehicleIds = traci.vehicle.getIDList()
        # Just check if the auto car still exisits and that there has not been any collision
        for VehId in self.VehicleIds:
            traci.vehicle.subscribe(VehId, (tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_LANE_INDEX, tc.VAR_DISTANCE))
            traci.vehicle.subscribeLeader(self.AutoCarID,
                                          50)  # Subscribe the vehicle information of the car in front of Auto
            # traci.vehicle.setSpeedMode(VehId,0) #Disable Car Following
            if VehId == self.AutoCarID:
                speedMode = traci.vehicle.getSpeedMode(VehId)
                speedMode = speedMode & int('11000', 2)
                # traci.vehicle.subscribeLeader(self.AutoCarID,50)  # Subscribe the vehicle information of the car in front of Auto
                traci.vehicle.setLaneChangeMode(VehId, 0)  # Disable automatic lane changing
                #Disable following SpeedLimit
                # traci.vehicle.setSpeedFactor(VehId,2)
                #self.AutocarSpeed = 15.00
                #traci.vehicle.setSpeed(VehId, 0)
        self.state = self._findstate()
        # traci.simulationStep()
        return self.state

    def close(self):
        traci.close()

    def start(self, gui=False):
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        traci.start([sumoBinary, "-c", config_path])
       # self.traci_data = traci.vehicle.getSubscriptionResults()

    def TrainingStatus(self, training):
        self.DisableFaultSimulation = training

#traci.start([sumoBinary, "-c", config_path])

