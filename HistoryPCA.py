import numpy as np
import time

def generate_X_i(A, sigma, B, p):
	A = np.matrix(A)
	k = A.shape[1]
	w = np.random.multivariate_normal(mean = np.zeros(p), cov = pow(sigma, 2) * np.identity(p), size = B)
	z = np.random.multivariate_normal(np.zeros(k), np.identity(k), B)
	x = np.matrix(z).dot(A.T) + w
	return x

def distance(mat1, mat2):
    return( np.abs(1 - np.linalg.norm(mat1.T.dot(mat2), 2)))

def ours(U, X, k, B, maxiter):
    n, d = X.shape
    V = np.random.randn(d,k)
    S = np.zeros((k,k))
    totaltime, timelist, iterlist, acclist = 0, [], [], []
    alltrace = pow(np.linalg.norm(X, ord = "fro"), 2)
    c = 0.1
    totalsample = 0 
    totalsample_prev = 0
    maxinner = 5 # the value of m
    ttt = np.random.permutation(n)
    X = X[ttt,:]
    XT = X.T
    nblock = int(n / B)
    maxiter_b = nblock * maxiter
    for i in range(1, maxiter_b+1):
        timebegin = time.time()
        if i <= nblock:
            nowid = i
        elif nblock == 1:
            nowid = 1
        else:
            nowid = np.random.randint(1, high = nblock, size=1)[0]
            
        my_begin = (nowid - 1) * B + 1
        my_end = nowid * B + 1
    
        if nowid == nblock:
            my_end = n
        
        XnowT = XT[:, my_begin:my_end]
        Xnow = XnowT.T
        

        newV = V
        for j in range(maxinner):
            newV = (XnowT.dot(Xnow.dot(newV))) / float(i)  + V.dot(S*((i-1)/float(i))).dot(V.T.dot(newV))
            newV, _ = np.linalg.qr(newV) # QR decomposition

        aa = newV.T.dot(XnowT)
        bb = V.T.dot(newV)
        S = aa.dot(aa.T)/float(i) + (bb.T.dot(S*(i-1)/float(i))).dot(bb)
        V = newV

        totaltime = totaltime + time.time() - timebegin
        totalsample = totalsample + B
        if (totalsample == B or totalsample > totalsample_prev + n/10):
            timelist.append(totaltime)
            iterlist.append(totalsample)
            acc = distance(V, U)
            acclist.append(acc)       
            totalsample_prev = totalsample
    return timelist, acclist, iterlist, V




    
