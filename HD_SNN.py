# Specify a name for the model to avoid overwriting old results
folder = 'My_Network'

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import nest as sim # type: ignore
import numpy as np
import pandas
from collections import Counter
import time as tm
import scipy.stats
import os

# Used to generate non-constant AHVs
def smoothclamp(x, mi, mx): return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

# Used to generate population weight matrices
def insertVec(m, v, x, y, axis = 0):
    if axis == 1:
        for i in range(len(v)):
            m[y,x+i]=v[i]
    else:
        for i in range(len(v)):
            m[y+i,x]=v[i]
    return m

# Used to calculate the current to be injected into the stationary ring
# c describes vertical scale
# s describes steepness
# b is vertical offset
def velCurve(x, c, s, b):
    y = np.empty_like(x)
    for i in range(len(x)):
        y[i] = c/(s*abs(x[i])+1) + b
    return y

deg_sign = u'\N{DEGREE SIGN}'



# misc params
use_stationary_ring = False # Creates a 5th ring. Connections are not currently set
vel_rescale = False     # Non-functional
long = False            # If true, runs a 720deg rotation instead of 180. Used for noisy models

# params describing number of cells in each population
N_ex = 180
N_in = N_ex
N_cj = N_ex # number of cj POPULATIONS, not neurons

# params for describing connection weights between populations
sigma = 0.12    # def 0.12
delay = 0.1     # def 0.1
base_ex = 4000  # def 4000    EX->IN 
base_in = 450   # def 450     IN->EX 
base_cj = 25    # def 169     CJ->EX   #narrow 25   #halfvel 36
base_ex_cj = 660        # def 660
I_ex = 450.0            # def 450 pA
vel_mult = 3500         # def 3500                  #halfvel 1750
sh = 150                # def 150    
in_sigma_scale = 0.5    # def 1        #narrow 0.5                 # multiplier of Sigma for the IN->EX distribution

base_st_ex = 0   #      ST->EX
I_st = 0.0        # def ? pA   peak current into the stationary layer, (c in the curve function)

# params describing size , duration and excitatory cell number to initialize the bump
I_init = 300.0 #pA
I_init_dur = 100.0 #ms
I_init_pos = N_ex//2

# PopCoding parameters
cj_population_size = 1      # set to 1 to not use pop-coding
pop_input_vari = 0.5        # EX->CJ variance
pop_output_vari = 0.0       # CJ->EX variance

# figure rendering tweak
ringhist_pad = 0.01


# ensure file struct
if not os.path.exists(f'results/{folder}'):
    os.makedirs(f'results/{folder}')
if not os.path.exists(f'results/{folder}/ring_hist'):
    os.makedirs(f'results/{folder}/ring_hist')
if not os.path.exists(f'results/{folder}/drift_hist'):
    os.makedirs(f'results/{folder}/drift_hist')
# Record parameter details
f = open("results/"+folder+"/Parameters.txt", "w")
f.write("N_ex " + str(N_ex) + "\n")
f.write("N_in " + str(N_in) + "\n")
f.write("N_cj " + str(N_cj) + "\n")
f.write("sigma " + str(sigma) + "\n")
f.write("base_ex " + str(base_ex) + "\n")
f.write("base_in " + str(base_in) + "\n")
f.write("base_cj " + str(base_cj) + "\n")
f.write("base_ex_cj " + str(base_ex_cj) + "\n")
f.write("I_ex " + str(I_ex) + "\n")
f.write("vel_mult " + str(vel_mult) + "\n")
f.write("sh " + str(sh) + "\n")
f.write("base_st_ex " + str(base_st_ex) + "\n")
f.write("I_st " + str(I_st) + "\n")
f.write("cj_population_size " + str(cj_population_size) + "\n")
f.write("in_sigma_scale " + str(in_sigma_scale) + "\n")
f.write("pop_input_vari " + str(pop_input_vari) + "\n")
f.write("pop_output_vari " + str(pop_output_vari) + "\n")
f.write("long " + str(long))
f.close()

# scale the the CJ->EX weight value to be appropriate for pop-size
base_cj /= cj_population_size # CJ -> EX#


# Everything in one big function to reset the kernal just in case
def run(myVel, graph_end=20, smoothVel = False, save_data = True):
    sim.ResetKernel()



    # V_th default = -55
    exc = sim.Create("iaf_psc_alpha",N_ex, params={"I_e": I_ex})
    inh = sim.Create("iaf_psc_alpha",N_in)

    l = sim.Create("iaf_psc_alpha",N_cj * cj_population_size)
    r = sim.Create("iaf_psc_alpha",N_cj * cj_population_size)
    st = sim.Create("iaf_psc_alpha",N_cj * cj_population_size)    # stationary ring


    # connections between excitatory and inhibitory rings described in weight matrices
    w_ex = np.empty((N_in,N_ex))
    w_in = np.empty((N_ex,N_in))

    #loop through each excitatory and inhibitory cell
    for e in range(N_ex):
        for i in range(N_in):
            # find the smallest distance between the excitatory and inhibitory cell, looking both ways around the ring
            d1 = abs(e/N_ex - i/N_in)
            d2 = abs(e/N_ex - i/N_in -1)
            d3 = abs(e/N_ex - i/N_in +1)
            d = min(abs(d1),abs(d2),abs(d3))
            
            #gaussian function finds the distance dependent connection strength
            w_gauss = np.exp(-(d)**2/2/sigma**2)
            w_ring = np.exp(-(d - 0.5)**2/2/(sigma*in_sigma_scale)**2) #inhibitory connections are ofset by parameter mu  # reduce deviation on inhib
            # scale by base weight and add to matrix
            w_ex[i,e] = base_ex * w_gauss
            w_in[e,i] = base_in * w_ring# * in_sigma_scale    # account for increased distrib        # MISTAKE HERE

    # set all very small weights to zero to reduce total number of connections
    w_ex[w_ex<10]=0
    w_in[w_in<10]=0



    ###
    #
    #   Build CJ weights
    #
    ###

    w_ex_cj = np.identity(N_ex) * base_ex_cj
    if(cj_population_size > 1):
        w_ex_cj = np.repeat(w_ex_cj, cj_population_size, axis=0)
        vec = np.linspace(base_ex_cj * (1+pop_input_vari), base_ex_cj  * (1-pop_input_vari), cj_population_size)
        for i in range(N_ex):
            w_ex_cj = insertVec(w_ex_cj, vec, i, i * cj_population_size, axis = 0)

    w_st_ex = np.identity(N_cj) * base_st_ex
    w_st_ex = np.repeat(w_st_ex, cj_population_size, axis=1)

    # connections between conjuntive layers and excitatory ring
    w_l = np.empty((N_ex,N_cj))
    w_r = np.empty((N_ex,N_cj)) # (row major)
    for c in range(N_cj):
        for e in range(N_ex):
            d1 = abs((e-1)/N_cj - c/N_ex)
            d2 = abs((e-1)/N_cj - c/N_ex -1)
            d3 = abs((e-1)/N_cj - c/N_ex +1)
            d = min(abs(d1),abs(d2),abs(d3))
            w_l[e,c] = base_cj * (np.exp(-(d)**2/2/sigma**2))
            
            d1 = abs((e+1)/N_cj - c/N_ex)
            d2 = abs((e+1)/N_cj - c/N_ex -1)
            d3 = abs((e+1)/N_cj - c/N_ex +1)
            d = min(abs(d1),abs(d2),abs(d3))
            w_r[e,c] = base_cj * (np.exp(-(d)**2/2/sigma**2))
    #temp1 = w_l
    #for c in range(N_cj):
    #    for e in range(N_ex):
    #        if w_l[e,c] <= w_l[(e+1)%(N_ex),c]:
    #            temp1[e,c] = 0
            #if w_r[(e+1)%(N_ex),c] <= w_r[e,c]:
            #    temp2[(e+1)%(N_ex),c] = 0
    #w_l = temp1
    #w_r = np.transpose(w_l)

    # this only takes the highest value. Takes the blurred line and turns it into a single thin line
    m = np.amax(w_l)
    w_l[w_l<m] = 0  
    m = np.amax(w_r)
    w_r[w_r<m] = 0

    
    if(cj_population_size > 1):
        w_r = np.repeat(w_r, cj_population_size, axis=1)
        w_l = np.repeat(w_l, cj_population_size, axis=1)
        w_r = np.zeros_like(w_l)
        w_l = np.zeros_like(w_r)
        vec = np.linspace(base_cj * (1+pop_output_vari), base_cj * (1-pop_output_vari), cj_population_size)
        for i in range(N_ex):
            w_r = insertVec(w_r, vec, i * cj_population_size, i-1, axis = 1)
        for i in range(N_ex):
            w_l = insertVec(w_l, vec, (i-1) * cj_population_size, i, axis = 1)

    c = plt.matshow(w_l)
    plt.colorbar(c)
    plt.xlabel("CJ neuron index")
    plt.ylabel("EX neuron index")
    plt.tick_params(labelbottom=False,labeltop=True)
    plt.savefig("W_cjL-ex.png", dpi=100*cj_population_size,bbox_inches="tight")
    plt.clf()
    c = plt.matshow(w_r)
    plt.colorbar(c)
    plt.xlabel("CJ neuron index")
    plt.ylabel("EX neuron index")
    plt.tick_params(labelbottom=False,labeltop=True)
    plt.savefig("W_cjR-ex.png", dpi=100*cj_population_size,bbox_inches="tight")
    plt.clf()
    c = plt.matshow(w_ex)
    plt.colorbar(c)
    plt.xlabel("EX neuron index")
    plt.ylabel("IN neuron index")
    plt.tick_params(labelbottom=False,labeltop=True)
    plt.savefig("W_ex-in.png", dpi=300,bbox_inches="tight")
    plt.clf()
    c = plt.matshow(-w_in, cmap=plt.get_cmap("viridis_r"))
    plt.colorbar(c)
    plt.xlabel("IN neuron index")
    plt.ylabel("EX neuron index")
    plt.tick_params(labelbottom=False,labeltop=True)
    plt.savefig("W_in-ex.png", dpi=300,bbox_inches="tight")
    plt.clf()
    c = plt.matshow(w_ex_cj)
    plt.colorbar(c)
    plt.xlabel("EX neuron index")
    plt.ylabel("CJ neuron index")
    plt.tick_params(labelbottom=False,labeltop=True)
    plt.savefig("W_ex_cj.png", dpi=300,bbox_inches="tight")
    plt.clf()
    print("weights vis generated")


    sim.Connect(exc,inh,'all_to_all',syn_spec={'weight': w_ex, 'delay': delay})     # connect Excitatory to Inhibitory
    sim.Connect(inh,exc,'all_to_all',syn_spec={'weight': -w_in, 'delay': delay})    # connect Inhibitory to Excitatory

    sim.Connect(exc,l,'all_to_all',syn_spec={'weight': w_ex_cj, 'delay': delay})    # connect Excitatory to AHV_L
    sim.Connect(exc,r,'all_to_all',syn_spec={'weight': w_ex_cj, 'delay': delay})    # connect Excitatory to AHV_R

    sim.Connect(l,exc,'all_to_all',syn_spec={'weight': w_l, 'delay': delay})        # connect AHV_L to Excitatory
    sim.Connect(r,exc,'all_to_all',syn_spec={'weight': w_r, 'delay': delay})        # connect AHV_R to Excitatory
    if(use_stationary_ring):
        sim.Connect(st,exc,'all_to_all',syn_spec={'weight': w_st_ex, 'delay': delay})   # connect Stationary to Excitatory

    #os._exit(0) 

    exc_spikes = sim.Create("spike_detector", 1, params={"withgid": True,"withtime": True}) # write to variable
    sim.Connect(exc,exc_spikes)
    l_spikes = sim.Create("spike_detector", 1, params={"withgid": True,"withtime": True}) # write to variable
    sim.Connect(l,l_spikes)
    r_spikes = sim.Create("spike_detector", 1, params={"withgid": True,"withtime": True}) # write to variable 
    sim.Connect(r,r_spikes) 
    inh_spikes = sim.Create("spike_detector", 1, params={"withgid": True,"withtime": True}) # write to variable 
    sim.Connect(inh,inh_spikes) 
    st_spikes = sim.Create("spike_detector", 1, params={"withgid": True,"withtime": True}) # write to variable 
    sim.Connect(st,st_spikes) 


    ###
    #
    #   Create vel curve
    #
    ###


    # posedata describes an actual path, but I can just set vel to a desired value 
    #posedata = pandas.read_csv(f'data/{folder}/{folder}.csv')

    dt = 20 # 20ms, 50Hz
    #theta = posedata['field.theta']
    #theta = np.array(theta)
    targetPos = np.pi #*4 # radians  
    if(long): 
        targetPos *= 4
    duration = (targetPos / (myVel * np.pi/180)) * 1000
    framecount = duration // dt
    theta = np.linspace(0,targetPos,int(framecount)) # in radians
    if(smoothVel): theta = smoothclamp(theta,0,targetPos) #smooth clamp velocity
    #print(theta)
    angle_per_cell = (2*np.pi)/N_ex
    I_init_pos = np.around((theta[0]//angle_per_cell)+(N_ex//2)).astype(int)
    


    t = np.arange(0,len(theta)*dt,dt*1.) #assume 20ms timestep
    sim_len = len(theta) * 20
    time = [i for i in t if i < sim_len]

    vel = np.diff(np.unwrap(theta))
    print(len(theta))


    #arbitary scaling
            #vel rescale
    #print("downscaled vel: " + str(vel[0]))
    #if(vel_rescale == True):
    #    vel = (vel + 14) / 26
    
    Ivel = (vel) * vel_mult
    #Ivel *= Ivel

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(t[1:], Ivel)
    fig.savefig("base vel over time.png")
    plt.clf()

    #os._exit(0)
    #print(vel)
    #print(Ivel)

    
    #st_i = velCurve(Ivel, I_st, 1, 0)
    st_i = np.ones_like(Ivel)
    st_i *= I_st
    #Ivel = velCurve(Ivel, -I_st, 1, I_st)

    go_l,go_r = Ivel,-Ivel #separate into clockwise and anticlockwise movements
    go_l = go_l+sh
    go_r = go_r+sh
    go_l[go_l<=sh] = 0 # everything below the threshold set to 0pA
    go_r[go_r<=sh] = 0


    # Connect AV input to conjunctive layers
    l_input = sim.Create('step_current_generator', 1)
    sim.SetStatus(l_input,{'amplitude_times': t[1:],'amplitude_values': go_l})  # current in picoamperes
    r_input = sim.Create('step_current_generator', 1)
    sim.SetStatus(r_input,{'amplitude_times': t[1:],'amplitude_values': go_r})
    st_input = sim.Create('step_current_generator', 1)
    sim.SetStatus(st_input,{'amplitude_times': t[1:],'amplitude_values': st_i})

    sim.Connect(r_input,r,'all_to_all')
    sim.Connect(l_input,l,'all_to_all')
    sim.Connect(st_input,st,'all_to_all')

    #os._exit(0)


    bump_init = sim.Create('step_current_generator', 1, params = {'amplitude_times':[0.1,0.1+I_init_dur],'amplitude_values':[I_init,0.0]})
    sim.Connect(bump_init,[exc[I_init_pos]])



    tic = tm.time()
    sim.Simulate(sim_len)
    print(f'Simulation run time: {np.around(tm.time()-tic,2)} s  Simulated time: {np.around(sim_len/1000,2)} s')





    if myVel >= 5:
        r_st = 0
        # Render all neuron histories
        T = np.arange(0,(len(theta)*dt),dt)   # Column
        fig_h = 2*(4+r_st) 
        fig_l = len(T)/180 * fig_h
        fig, ax=plt.subplots(4+r_st,1, figsize=(fig_l, fig_h), sharex=True,)
        plt.xlabel("Timestep (20ms)")

        offset = 1 
        ex_mat = np.zeros((N_ex, len(T)))
        ev = sim.GetStatus(exc_spikes)[0]['events']
        t = ev['times'] # time of each spike (in simulation time)
        sp = ev['senders'] # sender of each spike (cell ID)
        for i in range(len(T)-1):
            idx = (t>T[i])*(t<T[i+1]) # find all spikes in the bin
            sender_list = sp[np.where(idx)] # get the senders of each spike in the bin#
            for j in sender_list:
                ex_mat[j-offset][i] += 1
        ax[2+r_st].set_title("Exitatory ring spike history")
        plot = ax[2+r_st].imshow(ex_mat, cmap='hot', interpolation='nearest')
        ax[2+r_st].set_ylabel("Neuron index")
        plt.colorbar(plot,ax=ax[2+r_st],pad=ringhist_pad)
        
        offset += N_ex
        in_mat = np.zeros((N_in, len(T)))
        ev = sim.GetStatus(inh_spikes)[0]['events'] 
        t = ev['times'] # time of each spike (in simulation time)
        sp = ev['senders'] # sender of each spike (cell ID)
        for i in range(len(T)-1):
            idx = (t>T[i])*(t<T[i+1]) # find all spikes in the bin
            sender_list = sp[np.where(idx)] # get the senders of each spike in the bin#
            for j in sender_list:
                in_mat[j-offset][i] += 1
        ax[3+r_st].set_title("Inhibitory ring spike history")
        plot = ax[3+r_st].imshow(in_mat, cmap='hot', interpolation='nearest')
        ax[3+r_st].set_ylabel("Neuron index")
        plt.colorbar(plot,ax=ax[3+r_st],pad=ringhist_pad)
        
        offset += N_in
        cjl_mat = np.zeros((N_cj, len(T)))
        ev = sim.GetStatus(l_spikes)[0]['events'] 
        t = ev['times'] # time of each spike (in simulation time)
        sp = ev['senders'] # sender of each spike (cell ID)
        for i in range(len(T)-1):
            idx = (t>T[i])*(t<T[i+1]) # find all spikes in the bin
            sender_list = sp[np.where(idx)] # get the senders of each spike in the bin#
            for j in sender_list:
                cjl_mat[(j-offset)//cj_population_size][i] += 1
        ax[1+r_st].set_title("Left Conjunctive ring spike history")
        plot = ax[1+r_st].imshow(cjl_mat, cmap='hot', interpolation='nearest')
        if(cj_population_size>1):
            ax[1+r_st].set_ylabel("Population index")
        else:
            ax[1+r_st].set_ylabel("Neuron index")
        plt.colorbar(plot,ax=ax[1+r_st],pad=ringhist_pad)
        
        offset += N_cj*cj_population_size
        cjr_mat = np.zeros((N_cj, len(T)))
        ev = sim.GetStatus(r_spikes)[0]['events'] 
        t = ev['times'] # time of each spike (in simulation time)
        sp = ev['senders'] # sender of each spike (cell ID)
        for i in range(len(T)-1):
            idx = (t>T[i])*(t<T[i+1]) # find all spikes in the bin
            sender_list = sp[np.where(idx)] # get the senders of each spike in the bin#
            for j in sender_list:
                cjr_mat[(j-offset)//cj_population_size][i] += 1
        ax[0+r_st].set_title("Right Conjunctive ring spike history")
        plot = ax[0+r_st].imshow(cjr_mat, cmap='hot', interpolation='nearest')
        if(cj_population_size>1):
            ax[0+r_st].set_ylabel("Population index")
        else:
            ax[0+r_st].set_ylabel("Neuron index")
        plt.colorbar(plot,ax=ax[0+r_st],pad=ringhist_pad)

        if(r_st == 1):
            offset += N_cj*cj_population_size
            st_mat = np.zeros((N_cj, len(T)))
            ev = sim.GetStatus(st_spikes)[0]['events'] 
            t = ev['times'] # time of each spike (in simulation time)
            sp = ev['senders'] # sender of each spike (cell ID)
            for i in range(len(T)-1):
                idx = (t>T[i])*(t<T[i+1]) # find all spikes in the bin
                sender_list = sp[np.where(idx)] # get the senders of each spike in the bin#
                for j in sender_list:
                    st_mat[(j-offset)//cj_population_size][i] += 1
            ax[0].set_title("Stationary Conjunctive ring spike history")
            plot = ax[0].imshow(st_mat, cmap='hot', interpolation='nearest')
            if(cj_population_size>1):
                ax[0].set_ylabel("Population index")
            else:
                ax[0].set_ylabel("Neuron index")
            plt.colorbar(plot,ax=ax[0],pad=ringhist_pad)

        fig.tight_layout()    
        plt.savefig("results/"+folder+"/ring_hist/Ring histories " + str(myVel) +".png", dpi=150,bbox_inches="tight")
        plt.clf()






    
    T1 = np.arange(0,(len(theta)*dt),dt)   # 20ms bins
    offset = N_ex + N_in + 1
    ## Diagnose CJ spikes
    ev = sim.GetStatus(l_spikes)[0]['events'] 
    t = ev['times'] # time of each spike (in simulation time)
    sp = ev['senders'] # sender of each spike (cell ID)
    rate = np.zeros(len(T1)-1)
    for i in range(len(T1)-1):
        idx = (t>T1[i])*(t<T1[i+1]) # find all spikes in the bin
        sender_list = sp[np.where(idx)] # get the senders of each spike in the bin
        #print(sender_list)
        pop_list = np.zeros(N_cj)
        for j in sender_list:
            pop_list[(j-offset)//cj_population_size] += 1
        #print(pop_list)
        mode = pop_list.max() # find most common sender
        rate[i] = mode # spikerate in Hz
    #print("AHV spikes: " + str(rate))
    average_rate = rate.sum()/len(rate) #avg per timestep
    average_rate = average_rate * 50 #avg in seconds
    print("cj Hz: " + str(average_rate))
    average_cj_rate = average_rate

    ## ST spike rate
    ev = sim.GetStatus(st_spikes)[0]['events'] 
    t = ev['times'] # time of each spike (in simulation time)
    sp = ev['senders'] # sender of each spike (cell ID)
    modes = np.zeros(len(T1))
    modes[:] = np.nan
    rate = np.zeros(len(T1)-1)
    for i in range(len(T1)-1):
        idx = (t>T1[i])*(t<T1[i+1]) # find all spikes in the bin
        sender_list = sp[np.where(idx)] # get the senders of each spike in the bin
        occurence_count = Counter(sender_list) 
        mode = occurence_count.most_common(1) # find most common sender
        if(len(mode)>0):
            rate[i] = mode[0][1] # spikerate in Hz
        else:
            rate[i] = 0
    average_rate = rate.sum()/len(rate) #avg per timestep
    average_rate = average_rate * 50 #avg in seconds
    print("st Hz: " + str(average_rate))
    
    ## ex spike rate
    ev = sim.GetStatus(exc_spikes)[0]['events']
    t = ev['times'] # time of each spike (in simulation time)
    sp = ev['senders'] # sender of each spike (cell ID)
    modes = np.zeros(len(T1))
    modes[:] = np.nan
    rate = np.zeros(len(T1)-1)
    for i in range(len(T1)-1):
        idx = (t>T1[i])*(t<T1[i+1]) # find all spikes in the bin
        sender_list = sp[np.where(idx)] # get the senders of each spike in the bin
        occurence_count = Counter(sender_list) 
        mode = occurence_count.most_common(1) # find most common sender
        if(len(mode)>0):
            rate[i] = mode[0][1] # spikerate in Hz
        else:
            rate[i] = 0
    average_rate = rate.sum()/len(rate) #avg per timestep
    average_rate = average_rate * 50 #avg in seconds
    print("ex Hz: " + str(average_rate))
    average_ex_rate = average_rate







    T2 = np.arange(0,(len(theta)*dt),dt*2)   # 40ms bins

    # get the spike times of the cells from the spike recorder
    ev = sim.GetStatus(exc_spikes)[0]['events']
    t = ev['times'] # time of each spike (in simulation time)
    sp = ev['senders'] # sender of each spike (cell ID)
    #find the most active cell in each 40ms bin
    modes = np.zeros(len(T2))
    modes[:] = np.nan
    for i in range(len(T2)-1):
        idx = (t>T2[i])*(t<T2[i+1]) # find all spikes in the bin
        lst = sp[np.where(idx)] # get the senders of those spikes
        occurence_count = Counter(lst) 
        mode = occurence_count.most_common(1) # find most common sender
        if len(mode):
            modes[i] = mode[0][0]
    step = (2*np.pi)/N_ex
    modes = (modes*step) - np.pi

    if  myVel >= 5:
        # fig, ax = plt.subplots(1, 1,figsize=(5, 2),facecolor='w')
        fig, ax = plt.subplots(1, 1,figsize=(8, 3),facecolor='w')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Head angle (deg)')
        ax.set_title('AHV = ' + str(myVel) + " " + deg_sign + "/s")

        theta_wrap = (theta + np.pi) % (2*np.pi) - np.pi
        #print(theta_wrap)
        ax.plot(T2/1000,modes*(180/np.pi),'.',label='HD signal',color='steelblue')
        ax.plot(np.array(time)/1000,(theta_wrap[:len(time)])*(180/np.pi),'.',markersize=.5,label='ground truth',color='black')
        ax.set_xlim([0,sim_len/1000])
        ax.set_yticks([-180,-135,-90,-45,0,45,90,135,180])
        #ax.set_yticks([0,45,90,135,180])
        #ax.set_ylim([-10, 190])
        #ax2 = ax.twinx()
        #ax2.plot(T1[1:]/1000,rate, 'ro-', markersize=.5, alpha=0.6, linewidth=1)
        #ax2.set_ylabel("AHV cell firerate per timestep",color="red",fontsize=14)
        #ax2.set_ylim([0, 8])

        fig.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig("results/"+folder+"/drift_hist/"+str(myVel)+"_drift.png", dpi=300,bbox_inches="tight")
        plt.clf()


    # Compare against ground truth
    nanidx = np.where(~np.isnan(modes[:-1]))
    modes = modes[nanidx]
    T2=T2[nanidx]

    est = np.unwrap(modes)
    groundTruth = np.unwrap(theta[:len(time)])

    estimate = np.interp(time, T2, est)

    hd_estimate = np.vstack([T2,modes])
    np.save(f'results/{folder}/{folder}_estimate.npy',hd_estimate)


    # Find net RMSE
    fig, ax = plt.subplots(1, 1,figsize=(5, 2),facecolor='w')

    d = (estimate-groundTruth)

    if  myVel >= 5:
        plt.plot(np.array(time)/1000,abs(d)*(180/np.pi),color='steelblue')
        plt.xlabel('Time (s)')
        plt.ylabel('Abs Error (deg)')
        plt.xlim([0,sim_len/1000])
        plt.yticks([-45,0,45,90,135,180])
        plt.tight_layout()
        plt.savefig(f'results/{folder}/{folder}_error.png', bbox_inches="tight")
        plt.clf

    RMSE = np.sqrt(np.sum(d**2)/len(d))
    print("RMSE: " + str(RMSE*(180/np.pi)))



    drift = d[-1]*(180/np.pi) * 50 / len(theta) # drift per second

    print("Vel: " + str(myVel))
    print("Ivel: " + str(Ivel[1]))
    print("Total drift: " + str(d[-1]*(180/np.pi)))
    print("Drift per second: " + str(drift))

    if not save_data:
        return
    veldrift_f = open("results/" + folder + "/vel_drift_data.txt", "a")
    veldrift_f.write(str(myVel) + " " + str(drift) + " " + str(average_ex_rate) + " " + str(average_cj_rate) + "\n")
    veldrift_f.close()




    # draw all drift against vel values
    X = []
    Y = []
    Z = []

    f = open("results/"+folder+"/vel_drift_data.txt")
    for row in f:
        row = row.split(' ')
        X.append(float(row[0]))
        Y.append(float(row[1]))
        Z.append(float(row[3]))

    Y = [y for _,y in sorted(zip(X,Y))]
    Z = [z for _,z in sorted(zip(X,Z))]
    X.sort()

    fig,ax = plt.subplots()
    ax.hlines(0, 0, 1000, 'k', alpha=0.5)
    ax.plot(X, Y, 'bo-', markersize=3)
    ax.set_xlim([0, 80])
    ax.set_ylabel("Rate of Drift (" + deg_sign + "/s)",color="Blue",fontsize=14)
    ax.set_xlabel("Fixed AHV (" + deg_sign + "/s)",fontsize=14)
    ax2=ax.twinx()
    ax2.plot(X, Z, 'ro-', markersize=0.5, alpha=0.7)
    ax2.set_ylabel("Average peak CJ firerate (Hz)",color="red",fontsize=14)
    ax2.set_ylim([20, 70])
    fig.savefig("results/"+folder+"/DriftThruVel.png")
    plt.clf()
    print("DONE")

    plt.close('all')




loop = True
if(loop):
    start = 5
    end = 80
    step = 1
    for i in np.arange(start, end+step, step):
        print("\n\nRunning vel " + str(i))
        run(i, graph_end = end)
    print("\nCompleted batch " + folder + "\n")
else:
    run(50, save_data = False)