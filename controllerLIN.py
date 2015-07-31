# =================================================================================================================================================
#                                       Import modules
import sys
sys.path.append('/home/ercelik/opt1/nest/lib/python3.4/site-packages/')
import nest
import pickle
import random
import numpy as np

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def RandomConnect(Source,Dest,numCon,Weight,Delay,Model):
    #SourceNeurons=random.sample(Source,numCon)
    for i in range(len(Source)):
        DestNeurons=random.sample(Dest,numCon)
        nest.Connect(Source[i:i+1],DestNeurons,{'rule':'all_to_all'},{'weight':Weight,'model':Model,'delay':Delay})
def PoissonRate(poi,rate):
    nest.SetStatus(poi,"rate",rate+0.0)

def SpikeNum(SpikeGen,Stime,Ftime):
    timeArray=nest.GetStatus(SpikeGen,"events")[0]['times']
    betweenTimeInt=[j for j in range(len(timeArray)) \
                    if (timeArray[j]>=Stime and timeArray[j]<=Ftime)]
    return len(betweenTimeInt)+.0

def SetNeuronInput(Neuron,Sti):
    nest.SetStatus(Neuron,"I_e",Sti)

def CalcPoiRate(coeff,offset,maxRate,minRate,value):
    return np.exp(-coeff*(value+offset)**2)*(maxRate-minRate)+minRate

def ActFunc(x):
    return -0.135*x*np.exp(0.05*x)#-0.675*x*np.exp(0.25*x)#

def InputFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def InputAcc(value):
    global nPoi,minPoiRate,maxPoiRate,maxAcc,minAcc
    n=nPoi
    step=(maxAcc-minAcc)/n
    coeff=1/(step*4.)
    offset=np.arange(minAcc,maxAcc,step)+step/2.
    Poi=[CalcPoiRate(coeff,offset[i],maxPoiRate,minPoiRate,value) for  i in range(n)]
    return Poi


bpy.context.scene.game_settings.fps=50.
dt=1000./bpy.context.scene.game_settings.fps


#nest.sli_func('synapsedict info')
# =================================================================================================================================================
#                                       Creating muscles

FF=1.5
FF2=.05
muscle_ids = {}
[ muscle_ids["wrist.L_FLEX"], muscle_ids["wrist.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_wrist.L",  attached_object_name = "obj_forearm.L",  maxF = FF2)
[ muscle_ids["wrist.R_FLEX"], muscle_ids["wrist.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_wrist.R",  attached_object_name = "obj_forearm.R",  maxF = FF2)
[ muscle_ids["forearm.L_FLEX"], muscle_ids["forearm.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxF = FF)
[ muscle_ids["forearm.R_FLEX"], muscle_ids["forearm.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_forearm.R",  attached_object_name = "obj_upper_arm.R",  maxF = FF)

[ muscle_ids["upper_arm.L_FLEX"], muscle_ids["upper_arm.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L",  maxF = FF)
[ muscle_ids["upper_arm.R_FLEX"], muscle_ids["upper_arm.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R",  maxF = FF)
[ muscle_ids["shin_lower.L_FLEX"], muscle_ids["shin_lower.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin_lower.L",  attached_object_name = "obj_shin.L",  maxF = FF2)
[ muscle_ids["shin_lower.R_FLEX"], muscle_ids["shin_lower.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin_lower.R",  attached_object_name = "obj_shin.R",  maxF = FF2)

[ muscle_ids["shin.L_FLEX"], muscle_ids["shin.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin.L",  attached_object_name = "obj_thigh.L",  maxF = FF)
[ muscle_ids["shin.R_FLEX"], muscle_ids["shin.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin.R",  attached_object_name = "obj_thigh.R",  maxF = FF)
[ muscle_ids["thigh.L_FLEX"], muscle_ids["thigh.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_thigh.L",  attached_object_name = "obj_hips",  maxF = FF2)
[ muscle_ids["thigh.R_FLEX"], muscle_ids["thigh.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_thigh.R",  attached_object_name = "obj_hips",  maxF = FF2)


joints=["wrist.L_FLEX","wrist.L_EXT","wrist.R_FLEX","wrist.R_EXT","forearm.L_FLEX","forearm.L_EXT",\
        "forearm.R_FLEX","forearm.R_EXT","upper_arm.L_FLEX","upper_arm.L_EXT","upper_arm.R_FLEX",\
        "upper_arm.R_EXT","shin_lower.L_FLEX","shin_lower.L_EXT","shin_lower.R_FLEX","shin_lower.R_EXT",\
        "shin.L_FLEX","shin.L_EXT","shin.R_FLEX","shin.R_EXT","thigh.L_FLEX","thigh.L_EXT",\
        "thigh.R_FLEX","thigh.R_EXT"]


# =================================================================================================================================================
#                                       Network creation
## Nest Kernel Initialization
T=8
nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True,  "print_time": True})
nest.SetKernelStatus({"local_num_threads": T})
#aa=np.random.randint(1,100)
#nest.SetKernelStatus({'rng_seeds' : range(aa,aa+T)})
nest.sr("M_ERROR setverbosity")

## Network Parameters
neuronModel='aeif_cond_exp'
numSC=100   # Number of Spinal Cord neurons
dx=10
dy=5
dz=2
dim=3
inpDim=24   # Dimension of input variables
connProb=.3
synType='tsodyks2_synapse'
numInp=(numSC*1.)
numOut=24
Weight=10.

## Input Parameters
maxPoiRate=30.#1000.
minPoiRate=3.#10.
nPoi=5 # Dimension of activation function for each input variable
maxAcc=1.
minAcc=0.

## Network Definitions and Connections
conn=[(i-1,i+1,i-dx,i+dx,i-dx*dy,i+dx*dy) for i in range(numSC)] # nor recursive conn
SpinCord=nest.Create(neuronModel,numSC)

###########################################
Record_or_Test=0 # At first record the activities with zero, then regress a wout matrix, and test the locomotion with 1 
###########################################
contActRec=np.zeros((1,len(joints)))
outVRec=np.zeros((1,numOut))
afterRegressOutRec=np.zeros((1,numOut))
#wout=np.random.rand(2*numOut,numOut)
#intercept=np.zeros((numOut,1))
RecPositions=np.zeros((1,len(joints)))

if Record_or_Test==0:
    connection=[]
    for i in range(numSC):
        for j in range(dim*2):
            if np.random.rand()<connProb and conn[i][j]>=0 and conn[i][j]<numSC:
                nest.Connect(i+1,conn[i][j]+1,{'weight':Weight,'delay':1.,'model':synType})
                connection.append([i+1,conn[i][j]+1])
    PickleIt(connection,'connection')
    ## IDs of Input and Output Neurons
    InpNeurons=tuple(np.random.randint(np.amin(SpinCord),np.amax(SpinCord)+1,size=(1,numInp))[0])
    OutNeurons=tuple(np.random.randint(np.amin(SpinCord),np.amax(SpinCord)+1,size=(1,numOut))[0])
    PickleIt((InpNeurons,OutNeurons),'InpOutNeurons')
    
    
elif Record_or_Test==1: # Test case
    connect=np.array(GetPickle('connection'))
    for i in range(connect.shape[0]):
        nest.Connect(connect[i,0],connect[i,1],{'weight':Weight,'delay':1.,'model':synType})
    InpNeurons,OutNeurons=GetPickle('InpOutNeurons')
    wout=GetPickle('wout')
    intercept=GetPickle('intercept')
    posRecorded=GetPickle('positions')


    ## Recording Devices
outSpikes=nest.Create("spike_detector", 1, {"to_file": False})
nest.ConvergentConnect(OutNeurons, outSpikes, model="static_synapse")
    ## Input Stimulus 
poisson = nest.Create( 'poisson_generator' , nPoi*inpDim , { 'rate' : minPoiRate }) # for Inp1
nest.Connect(poisson,InpNeurons[:],{'rule':'all_to_all'},{'weight':40.,'model':'static_synapse','delay':1.})


RecordTime=200 #Record data to file once in every RecordTime








# =================================================================================================================================================
#                                       Evolve function
def evolve():
    print("Step:", i_bl, " Time:{0:.2f}".format(t_bl))
    global joints, contActRec, outVRec, wout,Record_or_Test, intercept
    global RecPositions,afterRegressOutRec
    global dt,numOut,wout,RecordTime,posRecorded
    
    # ------------------------------------- Visual ------------------------------------------------------------------------------------------------
    #visual_array     = getVisual(camera_name = "Meye", max_dimensions = [256,256])
    #scipy.misc.imsave("test_"+('%05d' % (i_bl+1))+".png", visual_array)
    # ------------------------------------- Olfactory ---------------------------------------------------------------------------------------------
    olfactory_array  = getOlfactory(olfactory_object_name = "obj_nose", receptor_names = ["smell1", "plastic1"])
    # ------------------------------------- Taste -------------------------------------------------------------------------------------------------
    taste_array      = getTaste(    taste_object_name =     "obj_mouth", receptor_names = ["smell1", "plastic1"], distance_to_object = 1.0)
    # ------------------------------------- Vestibular --------------------------------------------------------------------------------------------
    vestibular_array = getVestibular(vestibular_object_name = "obj_head")
    #print (vestibular_array)
    # ------------------------------------- Sensory -----------------------------------------------------------------------------------------------
    # ------------------------------------- Proprioception ----------------------------------------------------------------------------------------
    #~ spindle_FLEX = getMuscleSpindle(control_id = muscle_ids["forearm.L_FLEX"])
    #~ spindle_EXT  = getMuscleSpindle(control_id = muscle_ids["forearm.L_EXT"])
    # ------------------------------------- Neural Simulation -------------------------------------------------------------------------------------
    # nest.Simulate()

    positions=np.array([getMuscleSpindle(control_id = muscle_ids[joints[i]])[0] for i in range(len(joints))])
    RecPositions=np.vstack((RecPositions,positions))
    if np.mod(i_bl,RecordTime)==0 and Record_or_Test==0:
        PickleIt(RecPositions,'positions')
    elif np.mod(i_bl,RecordTime)==0 and Record_or_Test==1:
        PickleIt(RecPositions,'positionsTest')

    #print(InputAcc(positions))

    rates=np.array(InputAcc(positions))
    #print(rates)
    #rates=np.array(InputAcc(posRecorded[i_bl,:]))
    rates=list(rates.reshape(1,rates.shape[0]*rates.shape[1])[0])
    
    nest.SetStatus(poisson,'rate',rates)
    nest.Simulate(dt)

    ## Get Events
    T_times=nest.GetStatus(outSpikes,'events')[0]['times']
    T_senders=nest.GetStatus(outSpikes,'events')[0]['senders']
    #print(len(T_times))
    ## Time Shifting and Activity Calc
    time=nest.GetKernelStatus()['time']
    T_times-=time
    T_act=ActFunc(T_times)

    ## Output to Limbs
    outV=[[T_act[j] for j in range(T_senders.size) if OutNeurons[i]==T_senders[j]]for i in range(numOut)]
    
    outV=np.array([sum(outV[i]) for i in range(numOut)])
    #outV=outV/numOut
    #print(outV)

    ## Delete previous spikes
    nest.SetStatus(outSpikes, 'n_events', 0)


    
    # ------------------------------------- Muscle Activation -------------------------------------------------------------------------------------

    speed_ = 20.0
    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
    anti_act_tmp    = 1.0 - act_tmp
    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
    anti_act_tmp_p1 = 1.0 - act_tmp_p1
    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
    anti_act_tmp_p2 = 1.0 - act_tmp_p2




    contAct_real=[.4,.6,.4,.6,0.8*act_tmp,1.-0.8*act_tmp,0.8*anti_act_tmp,1.-0.8*anti_act_tmp,\
                  1.0*act_tmp_p1,1.-1.0*act_tmp_p1,1.0*anti_act_tmp_p1,1.-1.0*anti_act_tmp_p1,\
                  0.8*anti_act_tmp,1-0.8*anti_act_tmp,0.8*act_tmp,1.-0.8*act_tmp,\
                  0.5*anti_act_tmp_p1,1.-0.5*anti_act_tmp_p1,0.5*act_tmp_p1,1.-0.5*act_tmp_p1,\
                  0.5*anti_act_tmp,1.-0.5*anti_act_tmp,0.5*act_tmp,1.-0.5*act_tmp]

    
    

    if Record_or_Test==0:
        for i in range(len(joints)):
            controlActivity(control_id = muscle_ids[joints[i]], control_activity = contAct_real[i])
        contActRec=np.vstack((contActRec,contAct_real))
        outVRec=np.vstack((outVRec,outV))
        if np.mod(i_bl,RecordTime)==0:
            PickleIt(contActRec,'controlSignals')
            PickleIt(outVRec,'readOuts')

    elif Record_or_Test==1: # Test case
        out=wout.dot(outV)+intercept
        for i in range(len(joints)):
            controlActivity(control_id = muscle_ids[joints[i]], control_activity = out[i])
        afterRegressOutRec=np.vstack((afterRegressOutRec,out.T))
        if np.mod(i_bl,RecordTime)==0:
            PickleIt(afterRegressOutRec,'RegressedOutput')
    if i_bl>5201:
        bge.logic.endGame()

##    
