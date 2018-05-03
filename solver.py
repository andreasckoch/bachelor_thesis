import nifty4 as ift
from d4po.Energy import PLNMapEnergy, TauEnergy
from d4po.utilities import save_intermediate
from d4po.sampling import Sampler
from d4po import HamiltonianNormController
import numpy as np
import time
import sys
import plot_data as pd

# in plnlikelihoodmap.py
# print('Min: {}@{} | Max: {}@{} | l.sum()={} | energy={}'.format(np.min(position),np.argmin(position),np.max(position),np.argmax(position),l.sum(),self._value))


class D4PO_solver(object):
    def __init__(self, Problem, plotpath, timestamp=None,
                 ncpu=1):

        self._P = Problem

        self._ncpu = int(ncpu)
        self._nprobes = 2*ncpu
        self._timestamp = timestamp
        self._plotpath = plotpath

        self._update_para(0)

    @property
    def results(self):
        return self._P

    def _make_energy(self):
        P = self._P

        self._E_map_s = PLNMapEnergy(P.maps[0], component_no=0, meta_information=P,
                                     minimizer_controller=self._map_s_crtl_i)

        self._E_tau_s_0 = TauEnergy(P.tau[0][0], component_no=0, sub_domain_no=0,
                                    meta_information=P,
                                    minimizer_controller=self._t_crtl_i)

        ift.extra.operator_tests.adjoint_implementation(self._E_tau_s_0.curvature, np.float64, np.float64, 0., 1e-7)
        self._E_tau_s_1 = TauEnergy(P.tau[0][1], component_no=0, sub_domain_no=1,
                                    meta_information=P,
                                    minimizer_controller=self._t_crtl_i)

        ift.extra.operator_tests.adjoint_implementation(self._E_tau_s_1.curvature, np.float64, np.float64, 0., 1e-7)

    def _make_convergence_criteria(self, jj):

        data = self._P.data

        # setting convergence critiria, loosly in the begining, becoming stricter later on
        fudge = np.sqrt(self._P.maps[0].scalar_weight())
        tag_map_outer = self._P.maps[0].size * .0005 * fudge / (jj + 1)
        tag_map_inner = tag_map_outer * (jj ** 4. + 1) * fudge

        d_h = .001 / (jj ** 2. + 1)

        d_h_p = .0001 / (jj ** 2. + 1)

        tag_power_outer = max(data.shape) * .05 / (jj + 1)
        tag_power_inner = max(data.shape) * .05 / (jj + 1)

        map_s_crtl_o = HamiltonianNormController(name='MSC', tol_abs_gradnorm=tag_map_outer,
                                                 tol_abs_hamiltonian=d_h, convergence_level=3,
                                                 iteration_limit=data.size, verbose=True)
        t_crtl_o = HamiltonianNormController(name='PC', tol_abs_gradnorm=tag_power_outer,
                                             tol_abs_hamiltonian=d_h_p,
                                             convergence_level=5, iteration_limit=30 * max(data.shape),
                                             verbose=True)

        self._t_crtl_i = HamiltonianNormController(tol_abs_gradnorm=tag_power_inner,
                                                   convergence_level=5, iteration_limit=30 * max(data.shape), verbose=False)
        self._map_s_crtl_i = HamiltonianNormController(tol_abs_gradnorm=tag_map_inner, convergence_level=3,
                                                       iteration_limit=data.size, tol_abs_hamiltonian=d_h, verbose=False)

        self._map_s_minimizer_SD = ift.SteepestDescent(map_s_crtl_o)
        self._map_s_minimizer_BFGS = ift.VL_BFGS(map_s_crtl_o, max_history_length=100)
        self._map_s_minimizer_NT = ift.RelaxedNewton(map_s_crtl_o)

        self._power_minimizer_SD = ift.SteepestDescent(t_crtl_o)
        self._power_minimizer_BFGS = ift.VL_BFGS(t_crtl_o, max_history_length=100)
        self._power_minimizer_NT = ift.RelaxedNewton(t_crtl_o)

    def _update_para(self, jj):
        self._make_convergence_criteria(jj)
        self._make_energy()

    def __call__(self, iterations):

        for jj in range(iterations):

            sys.stdout.flush()

            if jj == 0:
                tick = time.time()

                print('Solver Iteration #%d' % jj)
                # print('optimizing diffuse map')
                print(self._E_map_s.position.min(), self._E_map_s.position.max())
                s = self._map_s_minimizer_NT(self._E_map_s)[0].position
                self._P.maps = 0, ift.Field(s.domain, val=np.clip(s.val, -s.max(), s.max()))
                self._make_energy()

                pd.plot_iteration(self._P, timestamp=self._timestamp, jj=jj, plotpath=self._plotpath)

                m, s = divmod(time.time()-tick, 60)
                h, m = divmod(m, 60)
                print('Solver Iteration #%d took: %dh%02dmin%02ds\n' % (jj, h, m, s))

            else:
                for ii in range(4):
                    tick = time.time()

                    print('Solver Iteration #%d.%d' % (jj, ii))
                    tack = time.time()
                    print("generating probes of diffuse map")
                    probes_s = Sampler(self._E_map_s, meta_information=self._P, component=0,
                                       nprobes=self._nprobes, diffuse_like=True, ncpu=self._ncpu)()
                    self._P.probes = 0, probes_s
                    # self._P.maps_uncertainty = 0, get_uncertainty(probes_s)
                    self._make_energy()

                    m, s = divmod(time.time()-tack, 60)
                    h, m = divmod(m, 60)
                    print('Probing took: %dh%02dmin%02ds\n' % (h, m, s))

                    if ii % 2 == 0:
                        print("optimizing tau-s in 1st-direction")
                        (E_tau_final_s_0, power_convergence_BFGS) = self._power_minimizer_NT(self._E_tau_s_0)
                        self._P.tau = 0, [E_tau_final_s_0.position, self._P.tau[0][1]]

                    elif ii % 2 == 1:
                        print("optimizing tau-s in 2nd-direction")
                        (E_tau_final_s_1, power_convergence_BFGS) = self._power_minimizer_NT(self._E_tau_s_1)
                        self._P.tau = 0, [self._P.tau[0][0], E_tau_final_s_1.position]

                    self._make_energy()

                    print('optimizing diffuse map')
                    s = self._map_s_minimizer_BFGS(self._E_map_s)[0].position
                    self._P.maps = 0, ift.Field(s.domain, val=np.clip(s.val, -s.max(), s.max()))
                    self._make_energy()

                    pd.plot_iteration(self._P, timestamp=self._timestamp, jj=jj, plotpath=self._plotpath, ii=ii)

                    m, s = divmod(time.time()-tick, 60)
                    h, m = divmod(m, 60)
                    print('Solver Iteration #%d took: %dh%02dmin%02ds\n' % (jj, h, m, s))

            self._update_para(jj + 1)


class D4PO_map_solver(object):
    def __init__(self, Problem, plotpath, timestamp=None, verbose=True):

        self._P = Problem
        self._verbose = verbose
        self._timestamp = timestamp
        self._plotpath = plotpath

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

            pd.plot_iteration(self._P, timestamp=self._timestamp, jj=jj, plotpath=self._plotpath)

            if self._verbose:
                m, s = divmod(time.time()-tick, 60)
                h, m = divmod(m, 60)
                print('Solver Iteration #%d took: %dh%02dmin%02ds\n' % (jj, h, m, s))
                tick = time.time()
