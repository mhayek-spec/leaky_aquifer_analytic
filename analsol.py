import numpy as np

def data(ell, kf, kv, b, B, N):
    lam, alpha, C, S = [], [], [], []
    for i in range(N):
        X = np.sqrt(b * B * kf[i] / kv[i])
        lam.append(X)
        Y = (ell[i+1] - ell[i]) / lam[i]
        C.append(np.cosh(Y))
        S.append(np.sinh(Y))
        Z = kf[i] / (lam[i] * S[i])
        alpha.append(Z)
    return lam, alpha, C, S

def vector_rho(ell, kf, kv, b, B, H0, HL, h0, N, casename="Case1"):
    lam, alpha, C, S = data(ell, kf, kv, b, B, N)
    rho = np.zeros((N-1))
    for i in range(N-1):
        rho[i] = (alpha[i] * (C[i] - 1.0) + alpha[i+1] * (C[i+1] - 1.0)) * h0
    if casename == "Case1": 
        rho[0] += alpha[0] * H0
    elif casename == "Case2":
        rho[0] += alpha[0] * h0 * (1.0 - 1.0 / C[0])
    else:
        raise Exception("case name error not applicable: case name should be Case1 or Case2")
    rho[N-2] += alpha[N-1] * HL
    return rho

def recurrence_d(ell, kf, kv, b, B, N, casename="Case1"):
    lam, alpha, C, S = data(ell, kf, kv, b, B, N)
    d = np.zeros(N-1)
    d[N-2] = alpha[N-2] * C[N-2] + alpha[N-1] * C[N-1]
    for i in range(N-3,0,-1):
        d[i] = alpha[i] * C[i] + alpha[i+1] * C[i+1] - alpha[i+1] ** 2.0 / d[i+1]
    if casename == "Case1" and N > 2:
        d[0] = alpha[0] * C[0] + alpha[1] * C[1] - alpha[1] ** 2.0 / d[1]
    elif casename == "Case2" and N > 2:
        d[0] = alpha[0] * C[0] + alpha[1] * C[1] - alpha[0] / C[0] - alpha[1] ** 2.0 / d[1]
    return d

def recurrence_v(ell, kf, kv, b, B, N, casename="Case1"):
    lam, alpha, C, S = data(ell, kf, kv, b, B, N)
    d = recurrence_d(ell, kf, kv, b, B, N, casename=casename)
    v = np.zeros(N-1)
    v[0] = 1.0 / d[0]
    for i in range(1,N-1):
        v[i] = np.prod(alpha[1:i+1]) / np.prod(d[0:i+1])
    return v

def recurrence_delta(ell, kf, kv, b, B, N, casename="Case1"):
    lam, alpha, C, S = data(ell, kf, kv, b, B, N)
    delta = np.zeros(N-1)
    if casename == "Case1": 
        delta[0] = alpha[0] * C[0] + alpha[1] * C[1]
    elif casename == "Case2":
        delta[0] = alpha[0] * C[0] + alpha[1] * C[1] - alpha[0] / C[0]
    for i in range(1,N-1):
        delta[i] = alpha[i] * C[i] + alpha[i+1] * C[i+1] - alpha[i] ** 2.0 / delta[i-1]
    return delta

def recurrence_u(ell, kf, kv, b, B, N, casename="Case1"):
    lam, alpha, C, S = data(ell, kf, kv, b, B, N)
    v = recurrence_v(ell, kf, kv, b, B, N, casename=casename)
    delta = recurrence_delta(ell, kf, kv, b, B, N, casename=casename)
    u = np.zeros(N-1)
    u[N-2] = 1.0 / (delta[N-2] * v[N-2])
    for i in range(1,N-1):
        u[N-2-i] = np.prod(alpha[N-1-i:N-1]) / (v[N-2] * np.prod(delta[N-2-i:N-1]))
    return u

def matrix_uv(ell, kf, kv, b, B, N, casename="Case1"):
    lam, alpha, C, S = data(ell, kf, kv, b, B, N)
    A = np.zeros((N-1, N-1))
    u = recurrence_u(ell, kf, kv, b, B, N, casename=casename)
    v = recurrence_v(ell, kf, kv, b, B, N, casename=casename)
    for i in range(N-1):
        for j in range(N-1):
            A[i][j] = u[min(i,j)] * v[max(i,j)]
    return A

def vector_internal_h_anal(ell, kf, kv, b, B, H0, HL, h0, N, casename="Case1"):
    invA = matrix_uv(ell, kf, kv, b, B, N, casename=casename)
    rho = vector_rho(ell, kf, kv, b, B, H0, HL, h0, N, casename=casename)
    h = np.zeros(N-1)
    for i in range(N-1):
        sum = 0.0 
        for j in range(N-1):
            sum += invA[i][j] * rho[j]
        h[i] = sum
    return h

def vector_h_anal(ell, kf, kv, b, B, H0, HL, h0, N, casename="Case1"):
    lam, alpha, C, S = data(ell, kf, kv, b, B, N)
    x = vector_internal_h_anal(ell, kf, kv, b, B, H0, HL, h0, N, casename=casename)
    if casename == "Case1":
        Hb = H0
    elif casename == "Case2":
        Hb = h0 + (x[0] - h0) / C[0]
    h = [Hb]
    for i in range(len(x)):
        h.append(x[i])
    h.append(HL)
    return h

def generate_log_uniform(min_val, max_val, n):
    # Generate n random numbers between log(min) and log(max)
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    
    # Generate n uniformly distributed random numbers in the log space
    uniform_randoms = np.random.uniform(log_min, log_max, n)
    
    # Exponentiate to get the log-uniform distribution
    log_uniform_randoms = np.exp(uniform_randoms)
    
    return log_uniform_randoms

def h_heter(x, l1, l2, kf, kv, b, B, h0, Hu, Hd):
    lam = np.sqrt(kf * B * b / kv)
    HU = Hu - h0
    HD = Hd - h0
    S = np.sinh((l2 - l1) / lam)
    S1 = np.sinh((x - l1) / lam)
    S2 = np.sinh((x - l2) / lam)
    h = h0 + HD * S1 / S - HU * S2 / S
    return h