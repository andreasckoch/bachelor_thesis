from __future__ import division
from __future__ import print_function

import nifty4 as ift
import numpy as np
import time
import sys
import plot_data as pd


from d4po.Energy import PLNMapEnergy
#from d4po.problem import Problem
#from d4po.utilities import save_intermediate
from d4po import HamiltonianNormController


class D4PO_solver(object):
    def __init__(self, Problem):

        self._P = Problem
        self._verbose = verbose
        self._timestamp = timestamp

        self._update_para(0)

    @property
    def results(self):
        return self._P

    def _make_energy(self):
        P = self._P

        self._E_map_s = PLNMapEnergy(P.maps[0], component_no=0, meta_information=P,
                                     minimizer_controller=self._map_crtl_i)

    def _make_convergence_criteria(self, jj):

        data = self._P.data

        # setting convergence critiria, loosly in the begining, becoming stricter later on
        tag_map_outer = self._P.maps[0].size * .0005 * np.sqrt(self._P.maps[0].scalar_weight()) / (jj + 1)
        tag_map_inner = tag_map_outer * (jj ** 4. + 1)

        d_h = .001 / (jj ** 2. + 1)

        map_crtl_o = HamiltonianNormController(name='MSC', tol_abs_gradnorm=tag_map_outer,
                                               tol_abs_hamiltonian=d_h, convergence_level=3,
                                               iteration_limit=data.size, verbose=True)

        self._map_crtl_i = HamiltonianNormController(tol_abs_gradnorm=tag_map_inner, convergence_level=3,
                                                     iteration_limit=data.size, tol_abs_hamiltonian=d_h)

        self._map_s_minimizer_SD = ift.SteepestDescent(map_crtl_o)
        self._map_s_minimizer_BFGS = ift.VL_BFGS(map_crtl_o, max_history_length=100)
        self._map_s_minimizer_NT = ift.RelaxedNewton(map_crtl_o)

    def _update_para(self, jj):
        self._make_convergence_criteria(jj)
        self._make_energy()

    def __call__(self, iterations):

        tick = time.time()

        for jj in range(iterations):

            sys.stdout.flush()

            print('Solver Iteration #%d' % jj)
            s = self._map_s_minimizer_NT(self._E_map_s)[0].position
            self._P.maps = 0, ift.Field(s.domain, val=s)
            self._make_energy()

            self._update_para(jj + 1)

            pd.plot_iteration(self._P, timestamp=self._timestamp, jj=jj)

            if self._verbose:
                m, s = divmod(time.time()-tick, 60)
                h, m = divmod(m, 60)
                print('Solver Iteration #%d took: %dh%02dmin%02ds\n' % (jj, h, m, s))
                tick = time.time()
