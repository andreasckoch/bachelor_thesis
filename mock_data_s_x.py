# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik

from __future__ import division
import nifty4 as ift
import numpy as np


# generating mock signal
# Poisson log normal, 1D spectra & one Field
# lam = R.times(exp(s))
#########################################################################

def mock_signal_s_x(npix):
    # setting random seed to get comparable results_1e3
    np.random.seed(28)

    # setting spaces
    total_volume = 127.  # total length

    # setting signal parameters
    lambda_s = .5  # signal correlation length
    sigma_s = 1.5  # signal variance

    # calculating parameters
    k_0 = 4. / (2 * np.pi * lambda_s)
    a_s = sigma_s ** 2. * lambda_s * total_volume

    # creation of spaces
    x1 = ift.RGSpace(npix, distances=total_volume / npix)
    k1 = x1.get_default_codomain()
    p1 = ift.PowerSpace(harmonic_partner=k1)

    # # creating power_field with given spectrum
    spec = (lambda k: a_s / (1 + (k / k_0) ** 2) ** 2)
    p_field = ift.PS_field(p1, spec)
    S = ift.create_power_operator(k1, power_spectrum=spec)

    # creating FFT-Operator and Response-Operator with Gaussian convolution
    HTOp = ift.HarmonicTransformOperator(domain=k1, target=x1)

    # drawing a random field
    sk = S.draw_sample()
    s = HTOp(sk)

    # setting dead times of instrument
    holes = 20
    dead = np.random.randint(0, npix, size=holes)
    length = np.random.randint(5, 50, size=holes)

    dead_times = np.ones(npix, dtype=np.float)

    for ii in range(holes):
        dead_times[dead[ii]:dead[ii]+length[ii]] = 0.

    M = ift.DiagonalOperator(ift.Field(x1, dead_times))

    # mock Response
    R = ift.GeometryRemover(x1)*M

    lam = R.times(ift.exp(s))

    data = ift.Field(lam.domain, val=np.random.poisson(lam.val),
                     dtype=np.float64, copy=True)

    # mark dead times using heuristic function inspect with threshold = 3:
    #data_mask, dead_count = utilities.find_dead_times(10, data)

    ift.plot(ift.Field(x1, val=lam.val), name='./trash/lambda.png')
    ift.plot(ift.Field(x1, val=data.val), name='./trash/data.png')
    #ift.plot(ift.Field(x1, val=data_mask.val), name='./trash/data_mask.png')
    ift.plot(ift.exp(s), name='./trash/signal.png')

    dic_config = {}
    dic_config['Response'] = R
    dic_config['data'] = data
    #dic_config['data_mask'] = data_mask
    dic_config['signal'] = s
    dic_config['spec'] = p_field
    #dic_config['dead_count'] = dead_count

    return dic_config


if __name__ == "__main__":
    dic_config = mock_signal_s_x()
    print(dic_config['dead_count'])
