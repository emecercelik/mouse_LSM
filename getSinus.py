import matplotlib.pyplot as plt
import numpy as np
import pickle

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

fps=50
positions=GetPickle('positions')
regOut=GetPickle('RegressedOutput')
testPos=GetPickle('positionsTest')
controlSig=GetPickle('controlSignals')

positions=np.vstack((positions[2:,:],positions[-1.:]))
regOut=regOut[1:,:]
testPos=np.vstack((testPos[2:,:],testPos[-1,:]))
controlSig=controlSig[1:,:]

numJoint=positions.shape[1]

numPlot=3
inp_len=positions.shape[0]

joints=["wrist.L_FLEX","wrist.L_EXT","wrist.R_FLEX","wrist.R_EXT","forearm.L_FLEX","forearm.L_EXT",\
        "forearm.R_FLEX","forearm.R_EXT","upper_arm.L_FLEX","upper_arm.L_EXT","upper_arm.R_FLEX",\
        "upper_arm.R_EXT","shin_lower.L_FLEX","shin_lower.L_EXT","shin_lower.R_FLEX","shin_lower.R_EXT",\
        "shin.L_FLEX","shin.L_EXT","shin.R_FLEX","shin.R_EXT","thigh.L_FLEX","thigh.L_EXT",\
        "thigh.R_FLEX","thigh.R_EXT"]


for i in range(len(joints)):
    numPlot=i
    plt.figure(i)
    time=np.arange(0,inp_len/fps,1/fps)

    #plt.axis([0,(2*np.pi)*7/speed,-0.1,1.1])

    plt.plot(time,positions[:,numPlot],'bo')
    plt.plot(time,positions[:,numPlot],'b',label='actual positions')
    plt.plot(time,controlSig[:,numPlot],'co')
    plt.plot(time,controlSig[:,numPlot],'c',label='Real control signals')
    plt.plot(time,testPos[:,numPlot],'ro')
    plt.plot(time,testPos[:,numPlot],'r',label='actual Test positions')
    plt.plot(time,regOut[:,numPlot],'go')
    plt.plot(time,regOut[:,numPlot],'g',label='regressed outputs to joints')

    
    plt.xlabel('Time (sec)')
    plt.ylabel('Position of Servos [0,1]')
    plt.legend()
    plt.title('{0:s} '.format(joints[numPlot]))
    plt.savefig('input_output_servo.png')
plt.show()
