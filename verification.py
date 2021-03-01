########################################
########################################
###########  Prawitz stuff  ############
########################################
########################################

from math import erf, exp, pi, sin, cos
from scipy.integrate import quad

# Pr[Z > x] where Z is a guassian
def gauss_tail(x):
    return (1-erf(x/2**0.5))/2

# Characteristic function of a standard normal variable
def f_Z(x):
    return exp(-x*x/2.0)

# An upper bound on |f_X(v)|, given an upper bound on a1
def f_X_bound(v, a1):
    # The solution of exp(-x^2/2)+cos(x) = 0 with x in [0, pi]
    theta = 1.7780882886686339603
    if a1 * v < theta: return exp(-v*v/2)
    elif a1*v < pi   : return (-cos(a1 * v))**(1/a1**2)
    else             : return 1

# An upper bound on |f_X(v)-f_Z(v)|, given an upper bound on a1
def f_X_f_Z_bound(v, a1):
    if a1 * v < pi / 2:
        return exp(-v*v/2.) - cos(a1 * v) ** (1. / a1**2)
    return 1 + exp(-v*v/2.)

# an upper bound on
#     Pr[X < x]
# for X with maximal weight <= a1.
# bound: Pr[X < x] < Pr[Z < x] + ret_value
def prawitz_bound(a1, x, T, q):
    import random
    Tx = T*x
    def k(u):
        piu = pi*u
        Txu = Tx*u
        return ((1-u) * sin(piu + Txu)/sin(piu) + sin(Txu)/pi)
    S1 = quad(lambda u: abs(k(u)) * f_X_f_Z_bound(u*T, a1), 0, q, epsabs=1E-9, limit=2000) if q != 0 else (0, 0)
    S2 = quad(lambda u: abs(k(u)) * f_X_bound(u*T, a1),     q, 1, epsabs=1E-9, limit=2000)
    S3 = quad(lambda u:     k(u)  * f_Z(u*T),               0, q, epsabs=1E-9, limit=2000) if q != 0 else (0, 0)
    S4 = gauss_tail(x) - 0.5

    # estimation for the integration error
    assert S1[1] + S2[1] + S3[1] < 1E-8

    # estimation of the sum of the integrals
    return (S1[0] + S2[0] + S3[0] + S4)

def memorize(f):
    d = {}
    def a(*t):
        if t not in d:
            d[t] = f(*t)
        return d[t]
    return a

# lower bound on Pr[X > x] for a Rademacher sum X
# with largest weight <= a.
@memorize
def simple_prawitz(a, x):
    if a/2 < x < 2*a:
        return gauss_tail(x) - prawitz_bound(a, x, pi/a, 0)
    else:
        return gauss_tail(x) - prawitz_bound(a, x, pi/a, 0.5)


########################################
########################################
########  Dynamic Programming  #########
########################################
########################################


M = 500
N = 500
K = 3*N
ITERS = 25
# Pr[X > x] >= A[int(a1*M)][int(x*N)+K]
# given the largest weight of X
# is <= a1.
A = [[0 for x in range(-K, K)] for a in range(0, M)]

def LB(a1, x):
    # A[-1] represents already a1 = 1...
    a = min(int(a1 * M), len(A)-1)
    coord = max(int(x * N) + K, 0)
    if coord >= len(A[a]): return 0.
    return A[a][coord]

# to calculate a lower bound on
# A[a][x] we round
# a to int(a*M+1)/M  and
# x to int(x*N+1)/N

def arange(s,e,t):
    for i in range(t+1):
        yield (s*(t-i)+e*i) / float(t)

def frange(s,e,d):
    if s <= e:
        while s < e:
            yield s
            s += d
        yield e
    yield e

# round to the next half multiple of g.
# could also (x+g-1) // g * g
def round_up(x, g):
    return (x + g//2) // g * g + g//2

for x in range(-K, K):
    if x % 100 == 0: print(x+K, "/", 2*K)
    for a1 in range(0, M):
        # The round-up is a (pessimistic) speedup. To allow caching.
        A[a1][x+K] = simple_prawitz(float(round_up(a1,10)+1)/M, float(round_up(x,5)+1)/N)
        if x < 0:
            A[a1][x+K] = max(A[a1][x+K], 0.5)

for i in range(ITERS):
    print(i, "/", ITERS, "  \t", round(LB(0.31, 1), 6))
    AA = [[elem for elem in arr] for arr in A]
    for x in range(-K, K):
        xx = (x+1) / N
        # notice the infimum is incremental
        mn = 1
        for a1 in range(0, M):
            # we only consider these a in [a1/M, (a1+1)/M], and use trivial bounds.
            mna = a1/M
            mxa = (a1+1)/M
            mnd = (1-mxa*mxa)**0.5
            # if xx < mna < a, clearly 1/4 lower bound --
            # both 'a' positive, and the rest.
            cur = float(xx < mna) / 4
            if a1+1 < M:
                cur = max(cur, (LB(mxa / mnd, (xx-mna)/mnd) + LB(mxa / mnd, (xx+mxa)/mnd)) / 2)
            mn = min(mn, cur)
            AA[a1][x+K] = max(A[a1][x+K], mn)
    A = AA

# proves Pr[X > 1] > to_prove, in case
# a_1+a_2+a_3 <= 1,
# a_1 in [0.3, 0.7]
# a_3 <= a3_bound
def case_a123_leq_1(to_prove, a3_bound, delta):
    for a1 in frange(0.3, 0.7, delta/10):
        for a2 in frange(0, min(a1, 1-a1), delta/10):
            max_a3 = min(1-a1-a2, a3_bound, a2)
            sig2 = (1-a1**2-a2**2)**0.5
            val = (
                LB(max_a3/sig2+delta, (1-a1-a2)/sig2+delta) +
                LB(max_a3/sig2+delta, (1-a1+a2)/sig2+delta) +
                LB(max_a3/sig2+delta, (1+a1-a2)/sig2+delta) +
                LB(max_a3/sig2+delta, (1+a1+a2)/sig2+delta)
            ) / 4
            if val <= to_prove: return False
    return True

# proves Pr[X > 1] > to_prove, in case
# a_1+a_2+a_3 >= 1,
# a_1+a_2 < 1,
# a_1 in [0.3, 0.7]
# a_3 <= a3_bound
def case_a123_geq_1(delta):
    for a1 in frange(0.333, 0.7, delta/15):
        for a2 in frange((1-a1)/2, min(1-a1, a1), delta/15):
            for a3 in frange(1-a1-a2, a2, delta/15):
                sig3 = (1-a1**2-a2**2-a3**2)**0.5
                L1 = a1+a2+a3-1
                L2 = 1-a1-a2+a3
                L3 = 1-a1+a2-a3
                L4 = 1+a1-a2-a3
                mx_a4 = min(a3, sig3)
                minimal_sig4 = (sig3**2 - mx_a4**2)**0.5
                if (L2 - L1 + delta / 2 < 0.35 * minimal_sig4 or
                    abs(a1-0.5) + abs(a2-0.5) + abs(a3-0.5) < 0.01
                ):
                    mx_a4 = min(mx_a4, 1-a1-a3)
                if (LB(mx_a4 / sig3 + delta, L2 / sig3 + delta) +
                        LB(mx_a4 / sig3 + delta, L3 / sig3 + delta) +
                        LB(mx_a4 / sig3 + delta, L4 / sig3 + delta)
                ) <= 0.25:
                    return False
    return True

print("verifying case a_1+a_2+a_3 <= 1")
assert case_a123_leq_1(3/32, 0.325, 0.004)
assert LB(0.3, 1) > 3/32
assert LB(0.3/(1-0.7**2)**0.5, 0.3/(1-0.7**2)**0.5) > 3/16
assert LB(0.35, 0.35) > 0.25
print("done verifying case a_1+a_2+a_3 <= 1")

print("verifying case a_1+a_2+a_3 >= 1")
assert case_a123_geq_1(0.015)
print("done verifying case a_1+a_2+a_3 >= 1")



################################################
####### Pr[X > 1] >= 1/16
################################################
def case_a12_leq_1(to_prove, delta):
    for a1 in frange(0.4, 0.6, delta/10):
        for a2 in frange(0, min(a1, 1-a1), delta/10):
            sig = (1-a1**2-a2**2)**0.5
            a3 = min(1-a1-a2, a2)
            val = (LB(a3/sig + delta, (1-a1-a2)/sig + delta) +
                   LB(a3/sig + delta, (1-a1+a2)/sig + delta) +
                   LB(a3/sig + delta, (1+a1-a2)/sig + delta) +
                   LB(a3/sig + delta, (1+a1+a2)/sig + delta)
            ) / 4
            if val <= to_prove: return False
    return True

assert LB(0.4, 1) > 1/12
assert LB(1/2, 1/2) > 1/6
assert case_a12_leq_1(1/12, 0.01)
