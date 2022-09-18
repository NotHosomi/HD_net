import matplotlib.pyplot as plt
import numpy as np
import os

deg_sign = u'\N{DEGREE SIGN}'
folder4 = ""
folder3 = ""
folder2 = ""
folder1 = "default"
#specify model names below here:

#folder1 = "narrow0.50-cj18.0"
#folder1 = "narrow_in-low_cj5-long"
#folder2 = "sigma0.5-in450-cj25-long"
#folder2 = "narrow_in-low_cj-fixed_in"
#folder1 = "sigma0.5-in450-cj25"
#folder2 = 'halfVel_cj38_long'
#folder3 = "halfVel_cj36_long"
#folder1 = "sigma0.5-in450-cj25"
#folder2 = "sigma0.5-in450-cj25-long"

#folder1 ="narrow0.50-cj18.0"
#folder2 ="sigma0.5-in450-cj25"

#folder1 = "sigma0.5_cj25"
#folder2 = "LinPop10_sigma0.5_ovar0.3"

#folder2 = "pop4-narrow"
#folder2 = "pop5-narrow"
#folder3 = "pop6-narrow"


#folder1 = 'sigma0.5_cj25'
#folder1 = "sigma0.5-in450-cj25-long"
#folder2 = 'LinPop10_sigma0.5_ivar0.1_ovar0_long'
#folder1 = 'LinPop10_sigma0.5_ovar0.3'
# folder1 = 'sigma0.5_cj35'
# folder3 = 'sigma0.5_cj15'
# folder2 = 'LinPop10_sigma0.5_ovar0.3_cj35'
# folder4 = 'LinPop10_sigma0.5_ovar0.3_cj15'

# folder1 = "LinPop10_sigma0.5_ivar0.1_ovar0_long"
# folder2 = "LinPop10_sigma0.5_ivar0.5_ovar0_long"

#folder1 = "LinPop10_sigma0.5_ivar0.1_ovar0_long"
#folder1 = "LinPop10_sigma0.5_ivar0.5_ovar0_long"
#folder2 = "LinPop10_sigma0.5_ivar0.5_ovar0.5_excj550"
#folder3 = "LinPop10_sigma0.5_ivar0.5_ovar0.5_excj750"

#folder1 = "sigma0.5-in450-cj25-long"
#folder1 = "LinPop10_sigma0.5_ivar0.1_ovar0_long"
#folder2 = "LinPop10_sigma0.5_ivar0.3_ovar0"
#folder3 = "LinPop10_sigma0.5_ivar0.5_ovar0_long"

#folder1 = "sigma0.5-in450-cj25-long"
folder1 = "sigma0.5_cj25"

dir = ""
if folder2=="":
    dir = "results/comparisons/"+folder1+"/"
    if not os.path.exists(f'results/comparisons/{folder1}'):
        os.makedirs(f'results/comparisons/{folder1}')
elif folder3=="":
    dir = "results/comparisons/"+folder1+" "+folder2+"/"
    if not os.path.exists(f'results/comparisons/{folder1} {folder2}'):
        os.makedirs(f'results/comparisons/{folder1} {folder2}')
elif folder4=="":
    dir = "results/comparisons/"+folder1+" "+folder2+" "+folder3+"/"
    if not os.path.exists(f'results/comparisons/{folder1} {folder2} {folder3}'):
        os.makedirs(f'results/comparisons/{folder1} {folder2} {folder3}')
else:
    dir = "results/comparisons/"+folder1+" "+folder2+" "+folder3+" "+folder4+"/"
    if not os.path.exists(f'results/comparisons/{folder1} {folder2} {folder3} {folder4}'):
        os.makedirs(f'results/comparisons/{folder1} {folder2} {folder3} {folder4}')
    

def fetch(folder):
    X = []
    Y = []
    Z = []
    W = []
    f = open("results/" + folder + "/vel_drift_data.txt")
    for row in f:
        row = row.split(' ')
        X.append(float(row[0]))
        Y.append(float(row[1]))
        Z.append(float(row[2]))
        if(len(row)>3):
            W.append(float(row[3]))
    Y = [y for _,y in sorted(zip(X,Y))]
    Z = [z for _,z in sorted(zip(X,Z))]
    if(len(W)>0):
        W = [w for _,w in sorted(zip(X,W))]
    X.sort()
    params = open("results/" + folder + "/Parameters.txt").readlines()
    pop = int(params[11].split()[1])
    W = [elem / pop for elem in W]
    return X, Y, Z, W

def Denoise(Y):
    nY = Y
    nY[0] = 0.67 * Y[0] + 0.33*Y[1]
    nY[-1] = 0.67 * Y[-1] + 0.33*Y[-2]
    for i in range(1,len(Y)-1):
        nY[i] = 0.33 * Y[i-1] + 0.34 * Y[i] + 0.33 * Y[i+1]
    return nY

def draw(x_lim, x_min = 0, rates=False, denoise=False):
    X1, _Y1, Z1, W1 = fetch(folder1)
    if(folder2!=""):
        X2, _Y2, Z2, W2 = fetch(folder2)    
    if(folder3!=""):
        X3, _Y3, Z3, W3 = fetch(folder3)
    if(folder4!=""):
        X4, _Y4, Z4, W4 = fetch(folder4)
    if(denoise):
        if(folder1!="default"):
            _Y1 = Denoise(_Y1)
        if(folder2!=""):
            _Y2 = Denoise(_Y2)
        if(folder3!=""):
            _Y3 = Denoise(_Y3)
        if(folder4!=""):
            _Y4 = Denoise(_Y4)
    fig,ax = plt.subplots()
    ax.hlines(0, 0, 1000, 'k', alpha=0.5)
    ax.plot(X1, _Y1, 'bo-', markersize=1, label="Rate of Drift")#"180" + deg_sign + " rotation")
    if(folder2!=""):
        ax.plot(X2, _Y2, 'go-', markersize=1, label=folder2)#"720" + deg_sign + " rotation")
    if(folder3!=""):
        ax.plot(X3, _Y3, 'ro-', markersize=1, label=folder2)#label="Reduced AHV current and Narrow Inhibition")
    if(folder4!=""):
        ax.plot(X4, _Y4, 'o-', color='orange', markersize=1, label=folder4)#label="Reduced AHV current and Narrow Inhibition")
    ax.set_xlim([x_min, x_lim])
    #plt.ylim([-80, 80])
    ax.set_ylabel("Rate of drift (" + deg_sign + "/s)",color="Black",fontsize=14)
    ax.set_xlabel("Constant AHV (" + deg_sign + "/s)",fontsize=14)
    ax.set_ylim([-80, 20])
    if folder2=="" and rates:
        ax2=ax.twinx()
        ax2.plot(X1, W1, 'ro-', markersize=0.5, alpha=0.7, label="CJ spike rate")
        ax2.plot(X1, Z1, 'o-', color='orange', markersize=0.5, alpha=0.7, label="EX spike rate")
        ax2.set_ylabel("Average peak CJ firerate (Hz)",color="Black",fontsize=14)
        ax2.set_ylim([0, 100])
    if folder2!="" and rates:
        ax2=ax.twinx()
        ax2.plot(X1, W1, 'ro-', markersize=0.5, alpha=0.7, label=folder1+" CJ spike rate")
        ax2.plot(X2, W2, 'o-', color='orange', markersize=0.5, alpha=0.7, label=folder2+" CJ spike rate")
        ax2.set_ylabel("Average peak CJ firerate (Hz)",color="Black",fontsize=14)
    if folder3!="" and rates:
        ax2.plot(X3, W3, 'yo-', markersize=0.5, alpha=0.7, label=folder3+" CJ spike rate")
    fig.legend()
    dn = ""
    if(denoise):
        dn="_denoise"
    fig.savefig(dir+"/DriftThruVel_"+str(x_lim)+dn+".png", dpi=300,bbox_inches="tight")
    plt.clf()

draw(80, rates=True, denoise=False)
#draw(80, denoise=True)
print("DONE")