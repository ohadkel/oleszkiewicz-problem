##############################################
## Jan 3, 2021. Vojtech Dvorak, Ohad Klein ##
##############################################

# A code companioned to the paper
#    https://arxiv.org/abs/2104.10005
# which verifies some numerical claims.

from math import erf, exp, pi, sin, cos
from scipy.integrate import quad
import functools

# Determines whether, in the numerical computation of Prawitz' bound,
# to use scipy's standard integrator (True),
# or a self-made integrator which only
# relies on that the integrands are Lipshitz functions (False),
# and produces rigorous estimates of Prawitz bound.
#
# Both options are available; the latter is slower.
use_scipy_integrator = True

########################################
########################################
####  Prawitz bound implementation  ####
########################################
########################################

# Main function in this section is "prawitz_bound"

# Characteristic function of a standard normal variable
def f_Z(x):
    return exp(-x*x/2.0)

# The solution of exp(-x^2/2)+cos(x) = 0 with x in [0, pi]
Theta = 1.7780882886686339603
# An upper bound on |f_X(v)|, given an upper bound on a1
def f_X_bound(v, a1):
    # The bound is correct only in "a1 * v < pi" range, which we assert.
    assert a1*v < pi
    if a1 * v < Theta: return exp(-v*v/2)
    elif a1*v < pi   : return (-cos(a1 * v))**(1/a1**2)

# An upper bound on |f_X(v)-f_Z(v)|, given an upper bound on a1
def f_X_f_Z_bound(v, a1):
    # the bound is correct only in "a1 * v < pi / 2" range, which we assert.
    assert a1 * v <= pi / 2
    return exp(-v*v/2.) - cos(a1 * v) ** (1. / a1**2)

# k(u, x, T) from the paper.
def k(u, x, T):
    piu = pi*u
    Txu = T*x*u
    if u == 0: return 1 + Tx/pi
    if u == 1: return 0
    return ((1-u) * sin(piu + Txu)/sin(piu) + sin(Txu)/pi)

# integrates f from a to b, with a list of singularity points
# returns two values: integral, and an error estimation.
def quad_with_singularities(f, a, b, singularities=[]):
    singularities = [x for x in singularities if a < x < b]
    # endpoints are singularities plus a,b
    endpoints = sorted(singularities + [a,b])
    integral = 0
    error_estimation = 0
    # we integrate between every two consecutive singularities
    for s,e in zip(singularities, singularities[1:]):
        # using scipy's standard integrator
        val = quad(f, s, e)
        integral += val[0]
        error_estimation += val[1]
    return integral, error_estimation

# a lower bound on
#     Pr[X > x]
# for a Rademacher sum X with maximal coefficient <= a1, and Var(X) = 1.
# This is the function F(a, x, T, q) given in the paper.
def F(a1, x, T, q, eps=0.01):
    # The three integrals appearing in F's definition in the paper.
    # The maximal additive errors sum to < eps.
    S1 = quad(lambda u: abs(k(u, x, T)) * f_X_f_Z_bound(u*T, a1), 0, q)
    S2 = quad_with_singularities(lambda u: abs(k(u, x, T)) * f_X_bound(u*T, a1),     q, 1, singularities=[Theta/(a1*T)])
    S3 = quad(lambda u:     k(u, x, T)  * f_Z(u*T), 0, q)

    # make sure the estimation of the integration error is small
    assert S1[1] + S2[1] + S3[1] < eps / 10

    # estimation of F as in the paper, minus error of the integration.
    # Note that eps / 10 (or even eps) is not a rigorous bound on the
    #     integration error. To get such a bound, use:
    #     use_scipy_integrator = False
    return 0.5 - eps - (S1[0] + S2[0] + S3[0])

# lower bound on Pr[X > x] for a Rademacher sum X
# with largest coefficient <= a, and Variance = 1.
# Just an application of F with T = pi/a, q = 0.5.
@functools.cache
def prawitz_bound(a, x):
    # If a1 is small, we increase it for efficiency reasons.
    # This is allowed --
    #   F2 lower bounds the supremum of Pr[X > x] where X is a
    #   normalized Rademacher sums with largest coefficient <= a1.
    if a < 0.2:
        return prawitz_bound(0.2, x)
    if use_scipy_integrator:
        # We compute F using scipy's standard integrator.
        return max(F(a, x, pi/a, 0.5), 0)
    else: # pedantic mode
        # F2 is defined below
        return max(F2(a, x, pi/a, 0.5), 0)

# A pedantic version of F.
def F2(a1, x, T, q, eps=0.01):
    Tx = abs(T*x)
    # The three integrands are Lipschitz with the following constants.
    # The Bounds are derived in Appendix titled "Numeric integration in our proofs"
    # of:   https://arxiv.org/pdf/2006.16834.pdf
    B_1 = T*(1+2*Tx/pi)+1.1*(Tx**2/(2*pi)+pi)
    B_2 = T*(1+2*Tx/pi)+Tx**2/(2*pi)+pi
    B_3 = 2*T/3*(1+2*Tx/pi)+Tx**2/(2*pi)+pi
    # Computing the integrated functions has absolute error < abs_error
    abs_error = 2**-40 * (2+abs(T*x))

    # the maximal additive errors sum to < eps
    S1 = lipschitz_integrate(lambda u: abs(k(u, x, T)) * f_X_f_Z_bound(u*T, a1), 0, q, eps/4, B_1, abs_error)
    S2 = lipschitz_integrate(lambda u: abs(k(u, x, T)) * f_X_bound(u*T, a1),     q, 1, eps/4, B_2, abs_error)
    S3 = lipschitz_integrate(lambda u:     k(u, x, T)  * f_Z(u*T),               0, q, eps/4, B_3, abs_error)

    # the value of F, minus the additive error allowed in the integration.
    return 0.5 - eps - (S1 + S2 + S3)

# Estimates the integral of f from a to b
# to within epsabs additive error, given
# a bound B on |f'|, and a bound C
# on the additive error involved in
# the computation (and summation) of f.
def lipschitz_integrate(f, a, b, epsabs, B, C):
    N = int(2 + B*(a-b)**2 / (4*epsabs + 4*C*(a-b)))
    # ensures the implied error is smaller than epsabs
    assert (B * (b-a)**2 / (4*N) + (b-a) * C) < epsabs
    sm = 0
    for k in range(1, N+1):
        sm += f(a + (2*k-1)*(b-a)/(2*N))
    return (b-a) * sm / N

########################################
########################################
########  Dynamic Programming  #########
########################################
########################################

# suppose we wish to lower bound
#     Pr[X > x]
# given X is a Rademacher sum, with Variance = 1
# and largest coefficient <= a_1.

# granularity of largest coefficient (a_1) is 1/M
M = 400
# granularity of threshold (x) is 1/N
N = 400
# we use dynamic programming for thresholds x = -K/N .. K/N
K = 3*N

#
# A[a][y+K] lower bounds Pr[X >= (y+1)/N]
# assuming X is Rademacher sum, with largest coefficient <= (a+1)/M
# and X has variance 1.
#
# In particular,
#      Pr[X > x] >= A[ceil(a_1*M)-1][floor(x*N)+K]
# for all Rademacher sums X, with Variance = 1,
# given that the largest coefficient of X is <= a_1
A = [[0 for y in range(0, 2*K)] for a in range(0, M)]

def D(a_1, x):
    # A[M-1] represents a_1 = 1 case.
    a = min(int(a_1 * M), len(A)-1)
    y = max(int(x * N) + K, 0)
    # A clear lower bound
    if y >= len(A[a]):
        return 0.
    return A[a][y]

def precompute_A(iters_num):
    import sys
    # round v to the next multiple of g.
    def round_up(v, g):
        return (v+g-1) // g * g

    print("Precomputation #1, {0} steps: ".format(2*K), end="")
    for y in range(-K, K):
        if y % 200 == 0:
            print("...{0}".format(y+K), end="")
            sys.stdout.flush()
        for a in range(0, M):
            # The round-up is a (pessimistic) speedup. To allow caching.
            A[a][y+K] = prawitz_bound(float(round_up(a,16)+1)/M,
                                      float(round_up(y,8)+1)/N)
            # If threshold < 0, then Pr[X > threshold] >= 1/2.
            if y < 0:
                A[a][y+K] = max(A[a][y+K], 0.5)

    print("")
    print("Precomputation #2, {0} steps: ".format(iters_num), end="")

    for i in range(iters_num):
        if i % 2 == 0:
            print("...{0}".format(i), end="")
            sys.stdout.flush()
        for y in range(-K, K):
            # The threshold we consider.
            t = (y+1) / N
            for a in range(0, M):
                # In A[a][y+K] we assign a lower bound to Pr[X >= t],
                # given a_1 <= (a+1)/M.
                # We split into two cases:
                #   a_1 <= a/M
                #   a_1 in [a/M, (a+1)/M]
                # The first case may be lower bounded using A[a-1, y+K].
                # The second case is lower bounded by elimination
                # of largest coefficient, and trivial bounds.
                # The lower bound is the minimum of the two cases.
                #
                # We start with the second case:
                min_a_1 = a/M
                max_a_1 = (a+1)/M
                # minimum variance of a_2 * epsilon_2 + ... + a_n * epsilon_n
                min_sigma = (1 - max_a_1**2)**0.5
                # if t <= a_1, clearly Pr[X >= t]
                # is lower bounded by 1/4:
                # the Rademacher sum is larger than t whenever both
                #   sign of a_1 is positive (probability 1/2)
                #   sign of the rest of the process is positive (probability >= 1/2)
                bound = float(t <= min_a_1) / 4
                # Note that the case a = M-1 which includes a_1 = 1,
                # for which elimination is prohibited, is handled correctly.
                if a+1 < M:
                    bound = max(bound,
                                (D(max_a_1 / min_sigma, (t-min_a_1)/min_sigma) +
                                 D(max_a_1 / min_sigma, (t+max_a_1)/min_sigma))
                                / 2)
                # We now consider the case a_1 <= a/M, and take the minimum.
                if a > 0:
                    bound = min(bound, A[a-1][y+K])
                # If we got a better lower bound to A[a][y+K], we update it.
                if bound > A[a][y+K]:
                    A[a][y+K] = bound

    print("")

precompute_A(10)

########################################
########################################
##########  Helper procedures  #########
########################################
########################################

# verifying s1 > s2
def verify_greater(s1, s2, verbose=True):
    v1 = eval(s1)
    v2 = eval(s2)
    if verbose:
        print("    {4} | verifying {0} > {1} | i.e. {2} > {3}".format(s1, s2, round(v1, 5), round(v2, 5), v1 > v2))
    assert v1 > v2

def frange(s,e,d):
    while s < e:
        yield s
        s += d
    yield e

########################################
########################################
#### Pr[X >= 0.35 Var(X)^0.5] > 1/2 ####
########################################
########################################

print("")
print("Verifying Pr[X > 0.35 Var(X)^0.5] >= 1/4")
verify_greater("D(0.35, 0.35)", "0.25")
print("Done verification Pr[X > 0.35 Var(X)^0.5] >= 1/4")

########################################
########################################
### 3/32 bound under a_1+a_2+a_3 <= 1 ##
########################################
########################################

print("")
print("Verifying Pr[X >= 1] >= 3/32 in case a_1+a_2+a_3 <= 1")

# proof in case a_1 <= 0.3
verify_greater("D(0.3, 1)", "3.0/32")
# proof in case a_1 >= 0.7
verify_greater("D(0.3/0.51**0.5, 0.3/0.51**0.5)", "3.0/16")

# verifications for case a_1 + a_2 + a_3 <= 1,
# a_1 in [0.3, 0.7] and a_3 <= 0.325.
def verify_case_a1_a2_a3_leq_1(delta):
    global minimal_sum_of_four_D
    a3_bound = 0.325
    minimal_sum_of_four_D = 1
    for a1 in frange(0.3, 0.7, delta/10):
        for a2 in frange(0, min(a1, 1-a1), delta/10):
            max_a3 = min(1-a1-a2, a3_bound, a2)
            sigma2 = (1-a1**2-a2**2)**0.5
            val = (
                D(max_a3/sigma2 + delta, (1-a1-a2)/sigma2 + delta) +
                D(max_a3/sigma2 + delta, (1-a1+a2)/sigma2 + delta) +
                D(max_a3/sigma2 + delta, (1+a1-a2)/sigma2 + delta) +
                D(max_a3/sigma2 + delta, (1+a1+a2)/sigma2 + delta)
            ) / 4
            minimal_sum_of_four_D = min(minimal_sum_of_four_D, val)
    verify_greater("minimal_sum_of_four_D", "3.0/32")

verify_case_a1_a2_a3_leq_1(0.005)

# verification required for a \in \mathcal{A}_1
verify_greater("D(0.216, 0.032)+2*D(0.216, 0.828)+D(0.216, 0.858)+"
               "D(0.216, 1.634)+2*D(0.216, 1.654)+D(0.216, 2.452)",
               "3.0/4")

# verification required for a \in \mathcal{A}_2
verify_greater("793/2048+2*D(0.41, 0.828)+D(0.41, 0.858)+"
               "D(0.41, 1.634)+2*D(0.41, 1.654)+D(0.41, 2.452)",
               "3.0/4")

verify_greater("D(0.41,0.032)+2*37/256+D(0.41, 0.858)+"
               "D(0.41, 1.634)+2*D(0.41, 1.654)+D(0.41, 2.452)",
               "3.0/4")

print("Done verification Pr[X >= 1] >= 3/32 in case a_1+a_2+a_3 <= 1")

########################################
########################################
### 3/32 bound under a_1+a_2+a_3 >= 1 ##
########################################
########################################

print("")
print("Verifying Pr[X >= 1] >= 3/32 in case a_1+a_2+a_3 >= 1")

# proves Pr[X > 1] > to_prove, in case
# a_1+a_2+a_3 >= 1,
# a_1+a_2 < 1,
# a_1 in [0.3, 0.7]
# a_3 <= a3_bound
def verify_case_a1_a2_a3_geq_1(delta):
    global minimal_sum_of_three_D
    minimal_sum_of_three_D = 1
    for a1 in frange(0.333, 0.7, delta/15):
        for a2 in frange((1-a1)/2, min(1-a1, a1), delta/15):
            for a3 in frange(1-a1-a2, a2, delta/15):
                sigma3 = (1-a1**2-a2**2-a3**2)**0.5
                L1 = a1+a2+a3-1
                L2 = 1-a1-a2+a3
                L3 = 1-a1+a2-a3
                L4 = 1+a1-a2-a3
                max_a4 = min(a3, sigma3)
                minimal_sigma4 = (sigma3**2 - max_a4**2)**0.5
                if (L2 - L1 + delta / 2 < 0.35 * minimal_sigma4
                    or
                    abs(a1-0.5) + abs(a2-0.5) + abs(a3-0.5) < 0.01
                ):
                    max_a4 = min(max_a4, 1-a1-a3)
                val = (D(max_a4 / sigma3 + delta, L2 / sigma3 + delta) +
                        D(max_a4 / sigma3 + delta, L3 / sigma3 + delta) +
                        D(max_a4 / sigma3 + delta, L4 / sigma3 + delta)
                )
                minimal_sum_of_three_D = min(minimal_sum_of_three_D, val)
    verify_greater("minimal_sum_of_three_D", "0.25")

verify_case_a1_a2_a3_geq_1(0.03)

print("Done verification Pr[X >= 1] >= 3/32 in case a_1+a_2+a_3 >= 1")

########################################
## Pr[X>1]>=1/12 under a_1+a_2+a_3<=1 ##
############ unless a_1 = 1 ############
########################################

print("")
print("Verifying Pr[X > 1] >= 1/12 in case a_1+a_2+a_3 <= 1")

def verify_case_a1_a2_leq_1(delta):
    minimal_sum_of_four_D = 1
    for a1 in frange(0.4, 0.6, delta/10):
        for a2 in frange(0, min(a1, 1-a1), delta/10):
            max_a3 = min(1-a1-a2, a2)
            s2 = (1-a1**2-a2**2)**0.5
            val = (D(max_a3/s2 + delta, (1-a1-a2)/s2 + delta) +
                   D(max_a3/s2 + delta, (1-a1+a2)/s2 + delta) +
                   D(max_a3/s2 + delta, (1+a1-a2)/s2 + delta) +
                   D(max_a3/s2 + delta, (1+a1+a2)/s2 + delta)
            ) / 4
            minimal_sum_of_four_D = min(minimal_sum_of_four_D, val)
    verify_greater("minimal_sum_of_four_D", "1.0/12")

verify_greater("D(0.4, 1)","1.0/12")
verify_greater("D(0.5, 0.5)","1.0/6")
verify_case_a1_a2_leq_1(0.01)

print("Done verification Pr[X > 1] >= 1/12 in case a_1+a_2+a_3 <= 1")
