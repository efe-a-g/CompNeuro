import numpy as np
from numba import njit
from scipy import linalg
from scipy.stats import chi2, dirichlet, invgamma, t

def addWGN(signal, SNR):
    """
    Adding white Gaussian Noise to the input signal
    signal              : Input signal, numpy array of shape (number of sources, number of samples)
                          If your signal is a 1D numpy array of shape (number of samples, ), then reshape it
                          by signal.reshape(1,-1) before giving it as input to this function
    SNR                 : Desired input signal to noise ratio
    print_resulting_SNR : If you want to print the numerically calculated SNR, pass it as True

    Returns
    ============================
    signal_noisy        : Output signal which is the sum of input signal and additive noise
    """
    sigpow = np.mean(signal**2, axis=1)
    noisepow = 10 ** (-SNR / 10) * sigpow
    noise = np.sqrt(noisepow)[:, np.newaxis] * np.random.randn(
        signal.shape[0], signal.shape[1]
    )
    signal_noisy = signal + noise
    return signal_noisy

def generate_uncorrelated_uniform_sources(n_sources=5, size_sources=500000):
    """
    Generates uncorrelated uniform sources

    required libraries:
    import numpy as np
    """
    S = np.random.uniform(-1, 1, size=(n_sources, size_sources))
    return S

def generate_correlated_copula_sources(
    rho=0.0, df=4, n_sources=5, size_sources=500000, decreasing_correlation=True
):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho**j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = (
            np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho
        )

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size=size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df / s) * Z  # chi-square method
    S = t.cdf(X, df).T
    return S