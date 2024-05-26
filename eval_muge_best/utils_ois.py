import numpy as np
import sys
eps=sys.float_info.epsilon
np.seterr(divide='ignore',invalid='ignore')
def findBestRPF(T,R,P):
    if len(T)==1:
        bestT,bestR,bestP=T,R,P
        bestF=(2*P*R)/max(eps,P+R)
    else:
        A=np.linspace(0,1,100)
        B=1-A
        bestF=-1
        for j in range(1,len(T)):
            Rj=R[j]*A+R[j-1]*B
            Pj = P[j]* A + P[j - 1]* B
            Tj = T[j]* A + T[j - 1] * B

            Fj = 2 * Pj* Rj/(Pj+Rj)
            f,k=np.max(Fj),np.argmax(Fj)
            if f>bestF:
                bestT=Tj[k]
                bestR=Rj[k]
                bestP=Pj[k]
                bestF=f
    return bestR,bestP,bestF,bestT

def computeRPF(cntR,sumR,cntP,sumP):
    
    R = cntR / max(sumR,eps)
    P = cntP /  max(sumP,eps)
    F = (2 * P * R) /max(P+R,eps)

    return R,P,F

def computeRPF_numpy(cntR,sumR,cntP,sumP):
    
    R = cntR / (sumR)
    P = cntP /  (sumP)
    F = (2 * P * R) /(P+R)

    return R,P,F
