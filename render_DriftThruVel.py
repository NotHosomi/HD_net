from re import T
import matplotlib.pyplot as plt
import numpy as np

deg_sign = u'\N{DEGREE SIGN}'

#folder = "BaseMetric"
#folder = "LinPop10_sigma0.5_ovar0.3_cj30"
#folder = "halfVel_cj36"
#folder = "sigma0.5_cj25"
#folder = "sigma0.5-in450-cj25"
folder = "default"

X = []
_Y = []
Z = []

f = open("results/" + folder + "/vel_drift_data.txt")
for row in f:
    row = row.split(' ')
    X.append(float(row[0]))
    _Y.append(float(row[1]))
    Z.append(float(row[2]))

_Y = [y for _,y in sorted(zip(X,_Y))]
Z = [z for _,z in sorted(zip(X,Z))]
X.sort()


def Denoise(Y):
    nY = Y
    nY[0] = 0.67 * Y[0] + 0.33*Y[1]
    nY[-1] = 0.67 * Y[-1] + 0.33*Y[-2]
    for i in range(1,len(Y)-1):
        nY[i] = 0.33 * Y[i-1] + 0.34 * Y[i] + 0.33 * Y[i+1]
    return nY

def draw(x_lim, x_min = 0, cj=False, denoise=False):
    Y=_Y
    if(denoise):
        Y = Denoise(_Y)
    fig,ax = plt.subplots()
    ax.hlines(0, 0, 1000, 'k', alpha=0.5)
    ax.plot(X, Y, 'bo-', markersize=1)
    ax.set_xlim([x_min, x_lim])
    ax.set_ylabel("Rate of drift (" + deg_sign + "/s)",color="Blue",fontsize=14)
    ax.set_xlabel("Fixed AHV (" + deg_sign + "/s)",fontsize=14)
    ax.set_ylim([-80, 20])
    #ax.set_xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120, 140])
    if cj:
        ax2=ax.twinx()
        ax2.plot(X, Z, 'ro-', markersize=0.5, alpha=0.7)
        ax2.set_ylabel("Average peak CJ firerate (Hz)",color="red",fontsize=14)
        #ax2.set_ylim([400, 600])
    dn = ""
    if(denoise):
        dn="_denoise"
    fig.savefig("results/"+folder+"/DriftThruVel_"+str(x_lim)+dn+".png", dpi=300,bbox_inches="tight")
    plt.clf()

#draw(34, 8)
#draw(40)
draw(80, cj=True, denoise=False)
print("DONE")