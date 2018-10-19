from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random

import traci
class CarFeatures:
    CarID = ''
    accel = 1
    decel = 1
    sigma = 0.5
    maxSpeed = 5
    length = 3
    color ="255,0,0"
    vClass = "passenger"
    minGap = 0
    tau = 0.1
class RouteCreator:
    # the Autonomous vehicle will always start at the end of other vehicles and will have the speed and acceleration more than the other ones
    AutoCarID = 'Auto'
    # Randomise the slow vehicles between 3 and 20
    NoOfOtherVehicles = random.randint(3, 20)
    # NoOfOtherVehicles = 4
    # Randomise the fast vehicles between 2 and 10
    NoOfOverSpeedVehicle = random.randint(2, 10)
    # NoOfOverSpeedVehicle = 5
    def CreateRoute(self):
        #random.seed(123)
        #Define 2 object 1 for normal car and one for autonomous vehicle
        AutoCarFeature = CarFeatures()
        # Features of the vehicles coming behind the auto car
        OverSpeedVeh = CarFeatures()
        OverSpeedVeh.accel  = 3
        OverSpeedVeh.decel = 3
        OverSpeedVeh.maxSpeed = 20
        OverSpeedVeh.color = "255,0,255"
        OverSpeedVeh.CarID = 'FastCar'
        # Change the features of the AutoCar
        AutoCarFeature.accel = 3
        AutoCarFeature.CarID = self.AutoCarID
        AutoCarFeature.decel = 3
        AutoCarFeature.maxSpeed = 40
        AutoCarFeature.color = "0,0,255"

        NormalCarFeature  = CarFeatures()
        NormalCarFeature.CarID = 'Car'
        with open("C:\Program Files (x86)\DLR\Sumo\Projects\HigwayOvertakingRandomVehicles\Data\StraightRoad.rou.xml","w") as routes:
            print("""<routes>
                <vType id= "%s" accel="%f" decel="%f" sigma="%f" maxSpeed = "%f" length="%f" color="%s" vClass="%s" minGap="0" tau="0.1"/>"""
                  % (AutoCarFeature.CarID, AutoCarFeature.accel, AutoCarFeature.decel, AutoCarFeature.sigma, AutoCarFeature.maxSpeed, AutoCarFeature.length,
                    AutoCarFeature.color, AutoCarFeature.vClass), file=routes)
            print("""<vType id= "%s" accel="%f" decel="%f" sigma="%f" maxSpeed = "%f" length="%f" color="%s" vClass="%s" minGap="0" tau="0.1"/>"""
                  % (NormalCarFeature.CarID, NormalCarFeature.accel, NormalCarFeature.decel, NormalCarFeature.sigma,
                     NormalCarFeature.maxSpeed, NormalCarFeature.length,
                     NormalCarFeature.color, NormalCarFeature.vClass), file=routes)
            print("""<vType id= "%s" accel="%f" decel="%f" sigma="%f" maxSpeed = "%f" length="%f" color="%s" vClass="%s" minGap="0" tau="0.1"/>"""
                % (OverSpeedVeh.CarID, OverSpeedVeh.accel, OverSpeedVeh.decel, OverSpeedVeh.sigma,
                   OverSpeedVeh.maxSpeed, OverSpeedVeh.length,
                   OverSpeedVeh.color, OverSpeedVeh.vClass), file=routes)
            print("""<route id="Straight" edges= "Lane"/>""", file=routes)
            lastvehstarttime = 0
            for i in range(self.NoOfOtherVehicles):
                # print("""<vehicle id="%s%d" type="%s" route="Straight" depart="%f"/>""" % (NormalCarFeature.CarID, i, NormalCarFeature.CarID, i*10), file=routes)
                vehStarttime = random.uniform(lastvehstarttime, 40)
                print("""<vehicle id="%s%d" type="%s" route="Straight" depart="%f"/>""" % (NormalCarFeature.CarID, i, NormalCarFeature.CarID, vehStarttime), file=routes)
                lastvehstarttime = vehStarttime

            print("""<vehicle id="%s" type="%s" route="Straight" depart="%f"/>""" % (AutoCarFeature.CarID, AutoCarFeature.CarID, 40), file=routes)
            # print("""<vehicle id="%s" type="%s" route="Straight" depart="%f"/>""" % (AutoCarFeature.CarID, AutoCarFeature.CarID, self.NoOfOtherVehicles*10), file=routes)
            lastvehstarttime = 40
            for i in range(self.NoOfOverSpeedVehicle):
                # print("""<vehicle id="%s%d" type="%s" route="Straight" depart="%f"/>""" % (OverSpeedVeh.CarID, i, OverSpeedVeh.CarID, ((i+1) * 10)+(self.NoOfOtherVehicles*10)), file=routes)
                vehStarttime = random.uniform(lastvehstarttime, 140)
                print("""<vehicle id="%s%d" type="%s" route="Straight" depart="%f"/>""" % (OverSpeedVeh.CarID, i, OverSpeedVeh.CarID, vehStarttime), file=routes)
            print("</routes>", file=routes)

# if __name__ == "__main__":
#     RouteCreatorObj = RouteCreator()
#     RouteCreator.CreateRoute(RouteCreator)
