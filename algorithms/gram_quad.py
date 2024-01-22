import numpy as np
from numba import njit

"""
Gram quadrature: Numerical integration with Gram polynomials, Irfan Muhammad
https://arxiv.org/abs/2106.14875.
"""

def create_gram_weights(m):
    """
    : param m : the degree of the l a r g e s t p o s s i b l e Gram p o l y n o m i a l
    for m + 1 points .
    : return : gram w e i g h t s for gram q u a d r a t u r e of m + 1 points .
    """
    max_d = int(np.sqrt(m))
    xs = np.linspace(-1, 1, m + 1)
    alphas = [1] + [alpha_func(n, m) for n in range(0, max_d + 1)
                    ]
    alphas = np.asarray(alphas)
    w = np.zeros(m + 1, dtype=float)
    # get GQ points and w e i g h t s
    xs_lg, w_lg = np.polynomial.legendre.leggauss(int(max_d / 2 +
                                                      1))
    # denote A_p as the p r e v i o u s row of A being worked on . A_n is the newest row.
    A_p = np.zeros(m + 1, dtype=float)
    A_n = np.asarray([(m + 1) ** (-0.5)] * (m + 1))
    q_p = np.zeros(int(max_d / 2 + 1), dtype=float)
    q_n = w_lg * np.asarray([(m + 1) ** (-0.5)] * int(max_d / 2 +
                                                      1))
    # update w for n = 0 , 1 , 2 , 3 , ... , max_d
    for j in range(0, max_d + 1):
        w, A_p, A_n, q_n, q_p = update_w(w, A_p, A_n, q_n, q_p,
                                         alphas, xs, xs_lg, j)
    return w

@njit
def alpha_func(n, m):
    a1 = m / (n + 1)
    a2 = (4 * np.power(n + 1, 2)) - 1
    a3 = np.power(m + 1, 2) - np.power(n + 1, 2)
    a4 = a2 / a3
    res = a1 * np.sqrt(a4)
    return res

@njit
def update_w(w, A_p, A_n, q_n, q_p, alphas, xs, xs_lg, j):
    w += (np.sum(q_n) * A_n)
    # update for j = 1 , 2 , 3 , ... , max_d + 1
    8
    A_temp = (alphas[j + 1] * (xs * A_n)) - ((alphas[j + 1] /
                                              alphas[j]) * A_p)
    A_p = A_n
    A_n = A_temp
    q_temp = q_n.copy()
    q_n = (alphas[j + 1] * xs_lg * q_n) - (alphas[j + 1] / alphas
    [j] * q_p)
    q_p = q_temp
    return w, A_p, A_n, q_n, q_p


if __name__ == '__main__':
    m = 100
    xs = np.linspace(-1, 1, m + 1)
    gram_weights = create_gram_weights(m)
    # the sum of stable w e i g h t s is equal to 2 .
    print(" Sum of Gram weights :", sum(gram_weights))
    # test integration , i n t e g r a t e f below b e t w e e n [ -1 , 1 ]

    a, b = -1, 1
    f = lambda x: 9 * x ** 2 + 45 * 13 * x ** 3 + 16 * x ** 4
    gram_quad = np.sum(gram_weights * f(xs), axis=-1)
    print(" Approx . integration :", gram_quad)
