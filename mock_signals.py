import numpy as np
import nifty4 as ift
from scipy.stats import lognorm


def mock_signal_time():
        # setting spaces
    npix = np.array([2**12])  # number of pixels
    total_volume = 350.  # total length, here about 350s

    # setting signal parameters
    lambda_s = .5  # signal correlation length
    sigma_s = 1.5  # signal variance

    # calculating parameters
    k_0 = 4. / (2 * np.pi * lambda_s)
    a_s = sigma_s ** 2. * lambda_s * total_volume

    # creation of spaces
    x1 = ift.RGSpace(npix, distances=total_volume / npix)
    k1 = x1.get_default_codomain()

    # # creating power_field with given spectrum
    spec = (lambda k: a_s / (1 + (k / k_0) ** 2) ** 2)
    S = ift.create_power_operator(k1, power_spectrum=spec)

    # creating FFT-Operator and Response-Operator with Gaussian convolution
    HTOp = ift.HarmonicTransformOperator(domain=k1, target=x1)

    # drawing a random field
    sk = S.draw_sample()
    s = HTOp(sk)

    # apply lognormal distribution to make signal look more realistic
    # s = ift.Field(s.domain, val=np.random.lognormal(s.val),
    # dtype=np.float64)

    ift.plot(ift.exp(s), name='./trash/mock_signal_time.png')

    return s


if __name__ == "__main__":
    mock_signal_time()


def mock_signal_energy():
    return NotImplemented
