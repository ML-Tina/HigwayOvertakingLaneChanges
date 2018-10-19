import matplotlib.pyplot as plt
import numpy as np
import pickle
FilePath = "C:\Program Files (x86)\Eclipse\Sumo\Projects\HigwayOvertakingLaneChanges\Result\SavedResults"
if __name__ == "__main__":
    # Plot RMS Error
    FPRMSError = FilePath + "\TrainRMSError"
    with open(FPRMSError, "rb") as fp:
        RMSErrorList = pickle.load(fp)
    ax = plt.axes()
    # x = list(range(1, len(TrainRewards) + 1))
    ax.set_ylim(0,500)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(200.0))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
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
