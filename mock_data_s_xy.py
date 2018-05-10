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
import numpy as np
import nifty4 as ift
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plotpath = 'trash'


def mock_signal_s_xy(t_pix, e_pix, t_volume, e_volume, smoothing_time, smoothing_energy, smoothness_sigma_time, smoothness_sigma_energy):
    # setting random seed to get comparable results_1e3
    np.random.seed(42)
    dic_config = {}
    #########################################################################
    # The spaces on which the inferred fields live are defined here
    #########################################################################

    # creation of spaces
    x_0 = ift.RGSpace(2 * t_pix, distances=t_volume / t_pix)
    k_0 = x_0.get_default_codomain()
    p_0 = ift.PowerSpace(harmonic_partner=k_0)

    # creation of spaces
    x_1 = ift.RGSpace(2 * e_pix, distances=e_volume / e_pix)
    k_1 = x_1.get_default_codomain()
    p_1 = ift.PowerSpace(harmonic_partner=k_1)

    FFTOp_0 = ift.FFTOperator(domain=(x_0, x_1), target=k_0, space=0)
    FFTOp_1 = ift.FFTOperator(domain=(k_0, x_1), target=k_1, space=1)

    #########################################################################
    # defining a Response operator, incl. a gaussian convolution & exposure
    #########################################################################

    #kernel_0 = 0. * x_0.distances[0]
    #kernel_1 = 0. * x_1.distances[0]
    R = ift.GeometryRemover(ift.DomainTuple.make((x_0, x_1)))
    # R = QPO.Response(ift.DomainTuple.make((x_0, x_1)))
    # R = ift.ResponseOperator([x_0, x_1], sigma=[kernel_0, kernel_1],
    #                         exposure=[1., 1.])

    #########################################################################
    # setting up diffuse source
    #########################################################################

    s_config = {}

    #########################################################################
    # setting spatial correlation
    #########################################################################

    # setting signal parameters
    lambda_s_0 = 10.  # signal correlation length
    sigma_s_0 = 2.  # signal variance

    # calculating parameters
    k0_0 = 4. / (2 * np.pi * lambda_s_0)
    a_s_0 = sigma_s_0 ** 2. * lambda_s_0 * t_volume

    # creating Power Operator with given spectrum
    spec_s_0 = (lambda k: a_s_0 / (1 + (k / k0_0) ** 2) ** 2)
    p_field_s_0 = ift.Field(p_0, val=spec_s_0(p_0.k_lengths), dtype=np.float64)

    # save original power spectrum
    dic_config['or_spec_0'] = p_field_s_0.copy()

    # generate delta peak
    kernel_0 = [1., 80., 1.]
    kernel_1 = [1., 1., 1.]
    position_0 = 0.05
    position_1 = 0.05

    for i in range(len(kernel_0)):
        p_field_s_0.val[int(position_0 * p_field_s_0.shape[0]):int(position_0 * p_field_s_0.shape[0]) + len(kernel_0)] = kernel_0 * \
            p_field_s_0.val[int(position_0 * p_field_s_0.shape[0]):int(position_0 * p_field_s_0.shape[0]) + len(kernel_0)]

    dic_config['spec_0'] = p_field_s_0

    #########################################################################
    # setting energy correlation
    #########################################################################

    # setting signal parameters
    lambda_s_1 = 10.  # signal correlation length
    sigma_s_1 = .3  # signal variance

    # calculating parameters
    k0_1 = 4. / (2 * np.pi * lambda_s_1)
    a_s_1 = sigma_s_1 ** 2. * lambda_s_1 * e_volume

    # creating Power Operator with given spectrum
    spec_s_1 = (lambda k: a_s_1 / (1 + (k / k0_1) ** 2) ** 2)
    p_field_s_1 = ift.Field(p_1, val=spec_s_1(p_1.k_lengths), dtype=np.float64)

    # save original power spectrum
    dic_config['or_spec_1'] = p_field_s_1.copy()

    # generate delta peak
    for i in range(len(kernel_1)):
        p_field_s_1.val[int(position_1 * p_field_s_1.shape[0]):int(position_1 * p_field_s_1.shape[0]) + len(kernel_1)] = kernel_1 * \
            p_field_s_1.val[int(position_1 * p_field_s_1.shape[0]):int(position_1 * p_field_s_1.shape[0]) + len(kernel_1)]

    dic_config['spec_1'] = p_field_s_1

    #########################################################################
    # drawing a diffuse random field with above defined correlation structure
    #########################################################################

    fp_s_0 = ift.Field(p_0, val=(lambda k: spec_s_0(k))(p_0.k_lengths))
    fp_s_1 = ift.Field(p_1, val=(lambda k: spec_s_1(k))(p_1.k_lengths))

    # generate delta peaks
    for i in range(len(kernel_0)):
        fp_s_0.val[int(position_0 * fp_s_0.shape[0]):int(position_0 * fp_s_0.shape[0]) + len(kernel_0)] = kernel_0 * \
            fp_s_0.val[int(position_0 * fp_s_0.shape[0]):int(position_0 * fp_s_0.shape[0]) + len(kernel_0)]
    for i in range(len(kernel_1)):
        fp_s_1.val[int(position_1 * fp_s_1.shape[0]):int(position_1 * fp_s_1.shape[0]) + len(kernel_1)] = kernel_1 * \
            fp_s_1.val[int(position_1 * fp_s_1.shape[0]):int(position_1 * fp_s_1.shape[0]) + len(kernel_1)]

    outer_s = np.outer(fp_s_0.val, fp_s_1.val)

    S_0 = ift.create_power_operator((k_0, k_1), fp_s_0, space=0)
    S_1 = ift.create_power_operator((k_0, k_1), fp_s_1, space=1)
    sk = (S_0*S_1).draw_sample()

    # sk = ift.power_synthesize(fp_s, spaces=(0, 1), real_signal=True)

    s = FFTOp_0.inverse_times(FFTOp_1.inverse_times(sk))
    # removing imaginary part
    s = ift.Field(s.domain, val=s, dtype=np.float64)

    dic_config['signal'] = s

    #########################################################################
    # drawing a Poissonian data sample from lambda
    #########################################################################

    lam = R.times(ift.exp(s))
    lam_val = np.clip(lam.val, np.exp(-10), lam.max())
    lam_poisson = np.random.poisson(lam_val)
    data = ift.Field(lam.domain, val=np.clip(lam_poisson, np.exp(-10), lam_poisson.max()), dtype=np.float64, copy=True)

    dic_config['diffuse'] = s_config
    dic_config['lam'] = lam
    dic_config['data'] = data
    dic_config['Response'] = R

    return dic_config
