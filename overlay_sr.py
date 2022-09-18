import matplotlib.pyplot as plt
import numpy as np
import os

deg_sign = u'\N{DEGREE SIGN}'
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

# folder1 = "sigma0.5_cj25"
# folder2 = "LinPop10_sigma0.5_ovar0.3"

folder1 = "sigma0.5_cj25"
folder2 = "LinPop10_sigma0.5_ivar0.1_ovar0_long"
folder3 = "LinPop10_sigma0.5_ivar0.5_ovar0_long"


dir = ""
if folder2=="":
    dir = "results/comparisons/"+folder1+"/"
    if not os.path.exists(f'results/comparisons/{folder1}'):
        os.makedirs(f'results/comparisons/{folder1}')
elif folder3=="":
    dir = "results/comparisons/"+folder1+" "+folder2+"/"
    if not os.path.exists(f'results/comparisons/{folder1} {folder2}'):
        os.makedirs(f'results/comparisons/{folder1} {folder2}')
else:
    dir = "results/comparisons/"+folder1+" "+folder2+" "+folder3+"/"
    if not os.path.exists(f'results/comparisons/{folder1} {folder2} {folder3}'):
        os.makedirs(f'results/comparisons/{folder1} {folder2} {folder3}')
    

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
        W.append(float(row[3]))
    Y = [y for _,y in sorted(zip(X,Y))]
    Z = [z for _,z in sorted(zip(X,Z))]
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
    if(denoise):
        if(folder1!="default"):
            W1 = Denoise(W1)
        if(folder2!=""):
            W2 = Denoise(W2)
        if(folder3!=""):
            W3 = Denoise(W3)
    #fig,ax = plt.subplots()
    plt.hlines(0, 0, 1000, 'k', alpha=0.5)
    if(folder2!=""):
        plt.plot(X1, W1, 'ro-', markersize=1, label=folder1)
        plt.plot(X2, W2, 'o-', color="orange", markersize=1, label=folder2)
    if(folder2==""):
        plt.plot(X1, W1, 'ro-', markersize=1, label="CJ spikerate")
        plt.plot(X1, Z1, 'o-', color="orange", markersize=1, label="EX spikerate")
    if(folder3!=""):
        plt.plot(X3, W3, 'yo-', markersize=1, label=folder3)
    plt.xlim([x_min, x_lim])
    #plt.ylim([-80, 80])
    plt.ylabel("Average peak spike rate (Hz)",color="Black",fontsize=14)
    plt.xlabel("Constant AHV (" + deg_sign + "/s)",fontsize=14)
    plt.ylim([30, 55])
    plt.legend()
    dn = ""
    if(denoise):
        dn="_denoise"
    plt.savefig(dir+"/SpikeRate_"+str(x_lim)+dn+".png", dpi=300,bbox_inches="tight")
    plt.clf()

draw(80, rates=True, denoise=False)
#draw(80, denoise=True)
print("DONE")