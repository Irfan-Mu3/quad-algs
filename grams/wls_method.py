import numpy as np
from scipy import integrate

"""
Weighted Least Square method, attributed to Huybrechs: Stable high-order quadrature rules with equidistant points.
https://doi.org/10.1016/j.cam.2009.05.018
"""


def create_wls_weights(m, N=3, C=2.5):
    # Unfortunately, C is sensitive to num_points: m
    # max 2.5 for(50- 500), min 1.85 (for 100000), 2 for 1000-4000, 1.95 for 5k-8k, 1.9 for 9k -11k, 1.85 for 12k-18k
    # Remember Simpsons 3/8 rule has N=3, 7 is also good

    Total = m
    subs = int(Total / N)

    an, _ = integrate.newton_cotes(N, 1)  # N = polynomial degree
    an = list(an)
    overlap = an[0] + an[-1]
    inner_w = ([overlap] + an[1:-1]) * (subs - 2)
    nc_r = an[:-1] + inner_w + [overlap] + an[1:]
    len_Tiled_N = len(nc_r)

    # warn: is this correct?
    # nc_r = np.ones(len_Tiled_N)
    # nc_r = np.sqrt(nc_r)

    xs = np.linspace(-1, 1, len_Tiled_N)
    m = len_Tiled_N - 1
    max_d = int(np.sqrt(m / C))
    print("WLS degree:", max_d)

    print("num_points, max_d:", m, max_d)

    # a_0,a_1,a_2,...,a_m+1
    alphas = np.empty(max_d + 2, dtype=float)
    alphas[0] = 0
    betas = np.empty(max_d + 2, dtype=float)
    betas[0] = 0

    w = np.zeros(m + 1, dtype=float)

    # we start from P_0,P_1
    r_p = np.ones(m + 1, dtype=float)
    alphas[1] = np.sum(nc_r * xs * r_p * r_p) / np.sum(nc_r * r_p ** 2)
    r_n = np.multiply(xs - alphas[1], r_p) - 0

    xs_lg, w_lg = np.polynomial.legendre.leggauss(int(max_d / 2 + 1))  # warn: memory error!
    b_p = w_lg * np.ones(int(max_d / 2 + 1), dtype=float)  # b_O, r_j can be used since we use the same length weights
    b_n = np.multiply(xs_lg - alphas[1], b_p) - 0

    w += (2 * np.ones(m + 1, dtype=float)) / np.sum(nc_r * r_p ** 2)

    assert len(r_n) == m + 1

    for j in range(0, max_d + 1):
        w, r_p, r_n, b_n, b_p = update_w(w, r_p, r_n, b_n, b_p, nc_r, xs, alphas, betas, xs_lg, j)

    return w

# todo: check initial beta
def update_w(_w, r_p, r_n, b_n, b_p, nc_r, xs, alphas, betas, xs_lg, j):
    _w += (sum(b_n) * r_n) / np.sum(r_n ** 2)  # starting from P_1*b_1

    # update for j=1,2,3,...max_d+1
    alphas[j + 1] = np.sum(nc_r * xs * r_n ** 2) / np.sum(nc_r * r_n ** 2)

    betas[j + 1] = np.sum(nc_r * xs * r_n * r_p) / np.sum(nc_r * r_p ** 2)

    r_t = np.multiply(xs - alphas[j + 1], r_n) - (betas[j + 1] * r_p)  # update latest
    r_p = r_n  # update previous
    r_n = r_t

    b_t = b_n.copy()
    b_n = np.multiply(xs_lg - alphas[j + 1], b_n) - (betas[j + 1] * b_p)
    b_p = b_t
    return _w, r_p, r_n, b_n, b_p
