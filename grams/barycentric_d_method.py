import numpy as np
from scipy.special import binom


"""
Barycentric method by Klein, Berrut: 
Klein, G., Berrut, JP. Linear barycentric rational quadrature. Bit Numer Math 52, 407–424 (2012). https://doi.org/10.1007/s10543-011-0357-x
"""

def create_barycentric_weights(N, d):
    xs = np.linspace(-1, 1, N)

    I = np.arange(0, N - d)
    # remember: computation of alphas not a bottleneck
    alphas = np.empty(N)
    for k in range(len(alphas)):
        ini = k - d if k >= d else 0
        idx_set = I[ini:k + 1]
        alphas[k] = (-1) ** (k - d) * sum([binom(d, k - i) for i in idx_set])

    xs_lg, w_lg = np.polynomial.legendre.leggauss(
        int(N / 2))

    # warn: this algorithm can lead to divison by zero if xs_lg = xs[k], hence the check below
    # todo: this loop can be optimized because we have two ordered lists?
    # if np.intersect1d(xs,xs_lg): assert False
    for xi in xs_lg:
        if xi in xs:
            print("Failure: There exists some xs_lg = xs[k]")
            assert False

    cs = np.zeros(len(xs_lg))
    for k in range(N):
        cs += alphas[k] / (xs_lg - xs[k])
    cs = 1 / cs
    w = np.empty(N)

    for k in range(N):
        w[k] = alphas[k] * np.sum((1 / (xs_lg - xs[k])) * cs * w_lg)
    return w


if __name__ == '__main__':
    w = create_barycentric_weights(1000, 10)

    # print("w[0]", w[0])
    # print(w)
