from scipy import integrate
from scipy.stats import uniform, expon, chi2
import numpy as np
from matplotlib import pyplot as plt
import time
import pyfftw as fftw

fftw.interfaces.cache.enable()
pfftw = fftw.interfaces.numpy_fft.fft
ipfftw = fftw.interfaces.numpy_fft.ifft

def fft_conv_alg(_xs, _w, _dx, dist1, dist2, both=False, w0_zero=False):
    temp_w = _w.copy()
    if w0_zero:
        temp_w[0] = 0

    if both:
        return 0.5 * (fft_alg3(_xs, temp_w, _dx, dist1, dist2, upper=True)
                      + fft_alg3(_xs, temp_w, _dx, dist1, dist2, upper=False))
    else:
        return fft_alg3(_xs, temp_w, _dx, dist1, dist2, upper=True)


def fft_alg3(_xs, _w, _dx, dist1, dist2, upper=True):
    extra_zeros = [0] * (len(_xs))

    if upper:
        FFT_wF = pfftw(np.append((_w * dist1.pdf(_xs)), extra_zeros))
        FFT_G = pfftw(np.append(dist2.pdf(_xs), extra_zeros))
    else:
        FFT_wF = pfftw(np.append((dist1.pdf(_xs)), extra_zeros))
        FFT_G = pfftw(np.append(_w * dist2.pdf(_xs), extra_zeros))

    return _dx * np.real(ipfftw(FFT_wF * FFT_G))[:len(_xs)]


if __name__ == '__main__':
    # step: Pacal

    from pacal import *

    # remember: we can get good accuracy if both distribution is the same!

    # Z = FDistr(25, 0.5) + ChiSquareDistr(2)

    # remember: beta is not continuous
    # remember: ChiSquared distribution has a parameter which is a whole number!

    a = 0
    b = 10  # remember: interval determines the success of the interpolant (see pdf error for barycentric error when b = 20)
    N_guess = 501  # remember: the number of points used should be a function of the interval length, e.g. 10*(|b-a|). This is to avoid really distributions that exist only in small regions

    # step: NC
    NC_d = 3  # remember Simpsons 3/8 rule has N=3, Trapezoid has N=1
    subs = int(N_guess / NC_d)
    start = time.time()
    flattened_xs = np.linspace(a, b, NC_d * subs + 1)  # remember: not between -1, 1. Therefore we only need dx

    N = NC_d * subs + 1
    an, _ = integrate.newton_cotes(NC_d, 1)  # N = polynomial degree
    dx = (b - a) / (NC_d * subs)
    an = list(an)
    overlap = an[0] + an[-1]
    inner_w = ([overlap] + an[1:-1]) * (subs - 2)
    simps_w = an[:-1] + inner_w + [overlap] + an[1:]

    adapted_trapz_w = np.asarray([1] + ([1] * (N - 1)))
    # adapted_trapz_w[1] += 0.4
    print("Done NC")

    # step: Grams
    scale = (b - a) / 2.0
    import gram_quad

    gram_w = gram_quad.create_gram_weights(N - 1)  # remember: because m = N-1
    print("Done Grams")

    # step: WLS
    import wls_method

    wls_w = wls_method.create_wls_weights(N - 1, NC_d, C=2.5)  # remember: because m =N-1
    print("Done WLS")

    # step: barycentric
    import barycentric_d_method

    bary_w = barycentric_d_method.create_barycentric_weights(N, 1)  # remember: best with d=1 !!
    print("Done Bary")

    # STEP: problem 1

    # # # # step: problem 1.1

    # step: recording

    fig, fig_axes = plt.subplots(ncols=3, nrows=3, constrained_layout=True)

    Z = ExponentialDistr() + ExponentialDistr()

    i_a = 0
    j_axis = 0

    dista = expon()
    distb = expon()

    # substep: Part a
    formula2 = False
    w0zero = False

    h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, formula2, w0zero)
    h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, formula2, w0zero)
    h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, formula2, w0zero)
    h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, formula2, w0zero)

    #substep: plot pdfs

    fig_axes[i_a, j_axis].plot(flattened_xs, Z.pdf(flattened_xs), label='Pacal')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_simp_dist, '--',label='Simps 3/8')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_gram_dist, '--',label='Gram')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_wls_dist, '--',label='WLS-S.3/8')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_bary_dist,'--', label='Bary')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist,'--', label='Rectangular')

    fig_axes[i_a, j_axis].set_title('Approximation of pdf: Expon(1) + Expon(1)')
    fig_axes[i_a, j_axis].set_xlabel('x')
    fig_axes[i_a, j_axis].set_ylabel('pdf(x)')
    fig_axes[i_a, j_axis].legend()

    # substep: pdf Error plot 1
    fig_axes[i_a, 1].set_yscale('log')
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), ':',linewidth=3,label='Simps 3/8 error',color='pink',zorder=10)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), ':',linewidth=3,label='Gram error',color='maroon')
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), ':',linewidth=5,label='WLS-S.3/8 error',color='limegreen',zorder=9)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':',linewidth=3,label='Bary error',color='cornflowerblue',zorder=0)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist), label='Rectangular error',linewidth=2,color='darkslategrey',zorder=11)

    fig_axes[i_a, 1].set_title('Using original weights')
    fig_axes[i_a, 1].set_xlabel('x')
    fig_axes[i_a, 1].set_ylabel('error')
    fig_axes[i_a, 1].set_ylim(1e-1,1e-20)
    fig_axes[i_a, 1].invert_yaxis()
    fig_axes[i_a, 1].legend()

    # substep: Part b
    formula2 = False
    w0zero = True

    h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, formula2, w0zero)
    h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, formula2, w0zero)
    h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, formula2, w0zero)
    h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, formula2, w0zero)

    # substep: pdf Error plot 1
    fig_axes[i_a, 2].set_yscale('log')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist),  '--',label='Simps 3/8 error',color='pink')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), label='Gram error',linewidth=2,color='maroon')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), label='WLS-S.3/8 error',linewidth=2,color='limegreen')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':', linewidth=3, label='Bary error',color='cornflowerblue')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist), label='Right Riemann error',linewidth=2,color='darkslategrey')

    fig_axes[i_a, 2].set_title('Using weights[0] = 0')
    fig_axes[i_a, 2].set_xlabel('x')
    fig_axes[i_a, 2].set_ylabel('error')
    fig_axes[i_a, 2].set_ylim(1e-1,1e-20)
    fig_axes[i_a, 2].invert_yaxis()
    fig_axes[i_a, 2].legend()

    # step: problem 1.2
    Z = UniformDistr(0,3) + UniformDistr(0,3)

    i_a = 1
    j_axis = 0


    dista = uniform(0,3)
    distb = uniform(0,3)

    # substep: Part a
    formula2 = False
    w0zero = False

    h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, formula2, w0zero)
    h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, formula2, w0zero)
    h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, formula2, w0zero)
    h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, formula2, w0zero)

    fig_axes[i_a, j_axis].plot(flattened_xs, Z.pdf(flattened_xs), label='Pacal')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_simp_dist, '--', label='Simps 3/8')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_gram_dist, '--', label='Gram')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_wls_dist, '--', label='WLS-S.3/8')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_bary_dist, '--', label='Bary')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist, '--', label='Rectangular')

    fig_axes[i_a, j_axis].set_title('Approximation of pdf: Uniform(0,3) + Uniform(0,3)')
    fig_axes[i_a, j_axis].set_xlabel('x')
    fig_axes[i_a, j_axis].set_ylabel('pdf(x)')
    fig_axes[i_a, j_axis].legend()

    # substep: pdf Error plot 1
    fig_axes[i_a, 1].set_yscale('log')
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), ':',linewidth=3,label='Simps 3/8 error',color='pink',zorder=10)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), ':',linewidth=3,label='Gram error',color='maroon')
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), ':',linewidth=5,label='WLS-S.3/8 error',color='limegreen',zorder=9)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':',linewidth=3,label='Bary error',color='cornflowerblue',zorder=0)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist), label='Rectangular error',linewidth=2,color='darkslategrey',zorder=11)

    fig_axes[i_a, 1].set_title('Using original weights')
    fig_axes[i_a, 1].set_xlabel('x')
    fig_axes[i_a, 1].set_ylabel('error')
    fig_axes[i_a, 1].set_ylim(1e-1,1e-20)
    fig_axes[i_a, 1].invert_yaxis()
    fig_axes[i_a, 1 ].legend()

    # substep: Part b
    formula2 = False
    w0zero = True

    h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, formula2, w0zero)
    h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, formula2, w0zero)
    h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, formula2, w0zero)
    h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, formula2, w0zero)

    # substep: pdf Error plot 1
    fig_axes[i_a, 2].set_yscale('log')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist),  '--',label='Simps 3/8 error',color='pink')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), label='Gram error',linewidth=2,color='maroon')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), label='WLS-S.3/8 error',linewidth=2,color='limegreen')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':', linewidth=3, label='Bary error',color='cornflowerblue')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist), label='Right Riemann error',linewidth=2,color='darkslategrey',zorder=11)

    fig_axes[i_a, 2].set_title('Using weights[0] = 0')
    fig_axes[i_a, 2].set_xlabel('x')
    fig_axes[i_a, 2].set_ylabel('error')
    fig_axes[i_a, 2].set_ylim(1e-1,1e-20)
    fig_axes[i_a, 2].invert_yaxis()
    fig_axes[i_a, 2].legend()

    # # step: problem 1.3
    Z = ChiSquareDistr(3) + ChiSquareDistr(3)

    i_a = 2
    j_axis = 0


    dista = chi2(3)
    distb = chi2(3)

    # substep: Part a
    formula2 = False
    w0zero = False

    h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, formula2, w0zero)
    h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, formula2, w0zero)
    h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, formula2, w0zero)
    h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, formula2, w0zero)



    fig_axes[i_a, j_axis].plot(flattened_xs, Z.pdf(flattened_xs), label='Pacal')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_simp_dist, '--', label='Simps 3/8')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_gram_dist, '--', label='Gram')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_wls_dist, '--', label='WLS-S.3/8')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_bary_dist, '--', label='Bary')
    fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist, '--', label='Rectangular')

    fig_axes[i_a, j_axis].set_title('Approximation of pdf: ChiSquared(3) + ChiSquared(3)')
    fig_axes[i_a, j_axis].set_xlabel('x')
    fig_axes[i_a, j_axis].set_ylabel('pdf(x)')
    fig_axes[i_a, j_axis].legend()

    # substep: pdf Error plot 1
    fig_axes[i_a, 1].set_yscale('log')
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), ':',linewidth=3,label='Simps 3/8 error',color='pink',zorder=10)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), ':',linewidth=3,label='Gram error',color='maroon')
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), ':',linewidth=5,label='WLS-S.3/8 error',color='limegreen',zorder=9)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':',linewidth=3,label='Bary error',color='cornflowerblue',zorder=0)
    fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist), label='Rectangular error',linewidth=2,color='darkslategrey',zorder=11)

    fig_axes[i_a, 1].set_title('Using original weights')
    fig_axes[i_a, 1].set_xlabel('x')
    fig_axes[i_a, 1].set_ylabel('error')
    fig_axes[i_a, 1].set_ylim(1e-1,1e-20)
    fig_axes[i_a, 1].invert_yaxis()
    fig_axes[i_a, 1 ].legend()

    # substep: Part b
    formula2 = False
    w0zero = True

    h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, formula2, w0zero)
    h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, formula2, w0zero)
    h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, formula2, w0zero)
    h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, formula2, w0zero)

    # substep: pdf Error plot 1
    fig_axes[i_a, 2].set_yscale('log')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist),  '--',label='Simps 3/8 error',color='pink')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), label='Gram error',linewidth=2,color='maroon')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), label='WLS-S.3/8 error',linewidth=2,color='limegreen')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':', linewidth=3, label='Bary error',color='cornflowerblue')
    fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist), label='Right Riemann error',linewidth=2,color='darkslategrey')

    fig_axes[i_a, 2].set_title('Using weights[0] = 0')
    fig_axes[i_a, 2].set_xlabel('x')
    fig_axes[i_a, 2].set_ylabel('error')
    fig_axes[i_a, 2].set_ylim(1e-1,1e-20)
    fig_axes[i_a, 2].invert_yaxis()
    fig_axes[i_a, 2 ].legend()

    plt.show()
    plt.savefig("Problem1:w0.pdf", dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', papertype='a4', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


    #############################################################################################
    #
    # fig, fig_axes = plt.subplots(ncols=2, nrows=3, constrained_layout=True)
    #
    # # # step: problem 2.1
    # Z = ChiSquareDistr(2) + ExponentialDistr()
    #
    # i_a = 0
    # j_axis = 0
    #
    # dista = chi2(2)
    # distb = expon()
    #
    # # substep: Part a
    # # formula2 = True
    # w0zero = True
    #
    # h_adpt_reimann_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, False, w0zero)
    # h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, True, w0zero)
    # h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, False, w0zero)
    # h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, False, w0zero)
    # h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, False, w0zero)
    # h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, False, w0zero)
    #
    # fig_axes[i_a, j_axis].plot(flattened_xs, Z.pdf(flattened_xs), label='Pacal')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_simp_dist, '--', label='Simps 3/8')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_gram_dist, '--', label='Gram')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_wls_dist, '--', label='WLS-S.3/8')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_bary_dist, '--', label='Bary')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_reimann_dist, '--', label='Right Riemann')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist, '--', label='Adapted trapezoid')
    #
    # fig_axes[i_a, j_axis].set_title('Approximation of pdf: Expon(1) + ChiSquared(2)')
    # fig_axes[i_a, j_axis].set_xlabel('x')
    # fig_axes[i_a, j_axis].set_ylabel('pdf(x)')
    # fig_axes[i_a, j_axis].legend()
    #
    # # substep: pdf Error plot 1
    # fig_axes[i_a, 1].set_yscale('log')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), ':',linewidth=5,label='Simps 3/8 error',color='cornflowerblue')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), ':',linewidth=5,label='Gram error',color='tomato')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist),':', linewidth=5,label='WLS-S.3/8 error',color='springgreen')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':',linewidth=5,label='Bary error',color='pink')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_reimann_dist), label='Right Riemann error',linewidth=2,color='blueviolet')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist),label='Adapted trapezoid error',linewidth=2,color='black')
    #
    # fig_axes[i_a, 1].set_title('Error plot')
    # fig_axes[i_a, 1].set_xlabel('x')
    # fig_axes[i_a, 1].set_ylabel('error')
    # fig_axes[i_a, 1].set_ylim(1e-1, 1e-20)
    # fig_axes[i_a, 1].invert_yaxis()
    # fig_axes[i_a, 1].legend()
    #
    # # # substep: Part b
    # # formula2 = True
    # # w0zero = True
    # #
    # # h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    # # h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, False, w0zero)
    # # h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, False, w0zero)
    # # h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, False, w0zero)
    # # h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, False, w0zero)
    # #
    # # # substep: pdf Error plot 2
    # # fig_axes[i_a, 2].set_yscale('log')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), label='Simps 3/8 error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), label='Gram error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), label='WLS-S.3/8 error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), label='Bary error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist), '--',label='Right Riemann error')
    # #
    # # fig_axes[i_a, 2].set_title('Using improved h[t]')
    # # fig_axes[i_a, 2].set_xlabel('x')
    # # fig_axes[i_a, 2].set_ylabel('error')
    # # fig_axes[i_a, 2].set_ylim(1e-1, 1e-20)
    # # fig_axes[i_a, 2].invert_yaxis()
    # # fig_axes[i_a, 2].legend()
    #
    # # # step: problem 1.2
    # Z =  FDistr(10, 0.5) + ChiSquareDistr(3)
    #
    # i_a = 1
    # j_axis = 0
    #
    #
    # dista = f(10,0.5)
    # distb = chi2(3)
    #
    # # substep: Part a
    # # formula2 = True
    # w0zero = True
    #
    # h_adpt_reimann_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, False, w0zero)
    # h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, True, w0zero)
    # h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, False, w0zero)
    # h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, False, w0zero)
    # h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, False, w0zero)
    # h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, False, w0zero)
    #
    #
    # fig_axes[i_a, j_axis].plot(flattened_xs, Z.pdf(flattened_xs), label='Pacal')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_simp_dist, '--', label='Simps 3/8')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_gram_dist, '--', label='Gram')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_wls_dist, '--', label='WLS-S.3/8')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_bary_dist, '--', label='Bary')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_reimann_dist, '--', label='Right Riemann')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist, '--', label='Adapted trapezoid')
    #
    # fig_axes[i_a, j_axis].set_title('Approximation of pdf: F(25,0.5) + ChiSquared(15) ')
    # fig_axes[i_a, j_axis].set_xlabel('x')
    # fig_axes[i_a, j_axis].set_ylabel('pdf(x)')
    # fig_axes[i_a, j_axis].legend()
    #
    # # #remember: test
    # # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist, label='NC orig. h[t]')
    #
    #
    # # substep: pdf Error plot 1
    # fig_axes[i_a, 1].set_yscale('log')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), ':',linewidth=5,label='Simps 3/8 error',color='cornflowerblue')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), ':',linewidth=5,label='Gram error',color='tomato')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist),':', linewidth=5,label='WLS-S.3/8 error',color='springgreen')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':',linewidth=5,label='Bary error',color='pink')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_reimann_dist), label='Right Riemann error',linewidth=2,color='blueviolet')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist),label='Adapted trapezoid error',linewidth=2,color='black')
    #
    # fig_axes[i_a, 1].set_title('Error plot')
    # fig_axes[i_a, 1].set_xlabel('x')
    # fig_axes[i_a, 1].set_ylabel('error')
    # fig_axes[i_a, 1].set_ylim(1e-1, 1e-20)
    # fig_axes[i_a, 1].invert_yaxis()
    # fig_axes[i_a, 1].legend()
    #
    # # # substep: Part b
    # # formula2 = True
    # # w0zero = True
    # #
    # # h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    # # h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, False, w0zero)
    # # h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, False, w0zero)
    # # h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, False, w0zero)
    # # h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, False, w0zero)
    # #
    # # # #remember: test
    # # # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist, label='NC improved h[t]')
    # #
    # # # substep: pdf Error plot 2
    # # fig_axes[i_a, 2].set_yscale('log')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), label='Simps 3/8 error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), label='Gram error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), label='WLS-S.3/8 error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), label='Bary error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist),'--', label='Right Riemann error')
    # #
    # # fig_axes[i_a, 2].set_title('Using improved h[t]')
    # # fig_axes[i_a, 2].set_xlabel('x')
    # # fig_axes[i_a, 2].set_ylabel('error')
    # # fig_axes[i_a, 2].set_ylim(1e-1, 1e-20)
    # # fig_axes[i_a, 2].invert_yaxis()
    # # fig_axes[i_a, 2].legend()
    #
    # # # step: problem 1.3
    # Z = UniformDistr(0,3) + ExponentialDistr(1)
    #
    # i_a = 2
    # j_axis = 0
    #
    # dista = UniformDistr(0,3)
    # distb = ExponentialDistr(1)
    #
    # # substep: Part a
    # # formula2 = True
    # w0zero = True
    #
    #
    # h_adpt_reimann_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, False, w0zero)
    # h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, True, w0zero)
    # h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, False, w0zero)
    # h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, False, w0zero)
    # h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, False, w0zero)
    # h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, False, w0zero)
    #
    # fig_axes[i_a, j_axis].plot(flattened_xs, Z.pdf(flattened_xs), label='Pacal')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_simp_dist, '--', label='Simps 3/8')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_gram_dist, '--', label='Gram')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_wls_dist, '--', label='WLS-S.3/8')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_bary_dist, '--', label='Bary')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_reimann_dist, '--', label='Right Riemann')
    # fig_axes[i_a, j_axis].plot(flattened_xs, h_adpt_trapz_dist, '--', label='Adapted trapezoid')
    #
    # fig_axes[i_a, j_axis].set_title('Approximation of pdf: Uniform(0,3) + Expon(1)')
    # fig_axes[i_a, j_axis].set_xlabel('x')
    # fig_axes[i_a, j_axis].set_ylabel('pdf(x)')
    # fig_axes[i_a, j_axis].legend()
    #
    # # substep: pdf Error plot 1
    # fig_axes[i_a, 1].set_yscale('log')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), ':',linewidth=5,label='Simps 3/8 error',color='cornflowerblue')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), ':',linewidth=5,label='Gram error',color='tomato')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist),':', linewidth=5,label='WLS-S.3/8 error',color='springgreen')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), ':',linewidth=5,label='Bary error',color='pink')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_reimann_dist), label='Right Riemann error',linewidth=2,color='blueviolet')
    # fig_axes[i_a, 1].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist),label='Adapted trapezoid error',linewidth=2,color='black')
    #
    # fig_axes[i_a, 1].set_title('Error plot')
    # fig_axes[i_a, 1].set_xlabel('x')
    # fig_axes[i_a, 1].set_ylabel('error')
    # fig_axes[i_a, 1].set_ylim(1e-1, 1e-20)
    # fig_axes[i_a, 1].invert_yaxis()
    # fig_axes[i_a, 1].legend()
    #
    # # substep: Part b
    # # formula2 = True
    # # w0zero = True
    # #
    # # h_adpt_trapz_dist = fft_conv_alg(flattened_xs, adapted_trapz_w, dx, dista, distb, formula2, w0zero)
    # # h_simp_dist = fft_conv_alg(flattened_xs, simps_w, dx, dista, distb, False, w0zero)
    # # h_gram_dist = fft_conv_alg(flattened_xs, gram_w, scale, dista, distb, False, w0zero)
    # # h_wls_dist = fft_conv_alg(flattened_xs, wls_w, scale, dista, distb, False, w0zero)
    # # h_bary_dist = fft_conv_alg(flattened_xs, bary_w, scale, dista, distb, False, w0zero)
    # #
    # # # substep: pdf Error plot 2
    # # fig_axes[i_a, 2].set_yscale('log')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_simp_dist), label='Simps 3/8 error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_gram_dist), label='Gram error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_wls_dist), label='WLS-S.3/8 error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_bary_dist), label='Bary error')
    # # fig_axes[i_a, 2].plot(flattened_xs, abs(Z(flattened_xs) - h_adpt_trapz_dist),'--', label='Right Riemann error')
    # #
    # # fig_axes[i_a, 2].set_title('Using improved h[t]')
    # # fig_axes[i_a, 2].set_xlabel('x')
    # # fig_axes[i_a, 2].set_ylabel('error')
    # # fig_axes[i_a, 2].set_ylim(1e-1, 1e-20)
    # # fig_axes[i_a, 2].invert_yaxis()
    # # fig_axes[i_a, 2].legend()
    #
    # plt.show()