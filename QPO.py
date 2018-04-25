import nifty4 as ift
import numpy as np
import utilities as QPOutils


class Response(ift.LinearOperator):

    # Input: 2D Histogram des Signals, dim: 2xt_pix x e_pix [wegen des paddings 2xt_pix]
    # domain muss tuple aus Zeit x Energie Räumen sein
    # Output: 3 x 2D Histogramme der Daten, instruments x time_bins x energy channels
    # Bilde Signalvektor auf jeweiligen Datenraum ab (Zeit, Energie[verschiedene Binnings für verschiedene instrumente])
    # Bei feinem Signalvektor weniger Unstimmigkeiten mit Datenbinning!
    # Mask muss Input Dimensions haben!!
    def __init__(self, domain, mask=None):
        super(Response, self).__init__()

        self._domain = ift.DomainTuple.make(domain)
        self._time_new_domain = ift.RGSpace(domain.shape[0] // 2, distances=domain[0].distances[0])
        self._x_new_domain = ift.DomainTuple.make((self._time_new_domain, domain[1]))
        self._target = ift.DomainTuple.make(ift.UnstructuredDomain((3, self._x_new_domain[0].shape[0], 256)))

        self._energy_response = QPOutils.build_energy_response(self._domain)
        self._energy_response_adjoint = self._energy_response.adjoint()
        self._energy_dicts, self._energies = QPOutils.get_dicts(True, True)
        self._instrument_factors = QPOutils.get_instrument_factors(length=3)

        if mask is None:
            self._M = ift.DiagonalOperator(ift.Field.ones(self._x_new_domain))
        else:
            self._M = ift.DiagonalOperator(mask)

    def set_mask(self, mask):
        self._M = ift.DiagonalOperator(mask)

    @property
    def time_padded_domain(self):
        """DomainTuple : the operator's transition domain after the
                         input field's time dimension was cut in half"""
        return self._x_new_domain

    @property
    def domain(self):
        """DomainTuple : the operator's input domain

            The domain on which the Operator's input Field lives."""
        return self._domain

    @property
    def target(self):
        """DomainTuple : the operator's output domain

            The domain on which the Operator's output Field lives."""
        return self._target

    @property
    def capability(self):
        """int : the supported operation modes

        Returns the supported subset of :attr:`TIMES`, :attr:`ADJOINT_TIMES`,
        :attr:`INVERSE_TIMES`, and :attr:`ADJOINT_INVERSE_TIMES`,
        joined together by the "|" operator.
        """
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        """ Applies the Operator to a given `x`, in a specified `mode`.

        Parameters
        ----------
        s : Field
            The input Field, living on the Operator's domain or target,
            depending on mode.

        mode : int
            - :attr:`TIMES`: normal application
            - :attr:`ADJOINT_TIMES`: adjoint application
            - :attr:`INVERSE_TIMES`: inverse application
            - :attr:`ADJOINT_INVERSE_TIMES` or
              :attr:`INVERSE_ADJOINT_TIMES`: adjoint inverse application

        Returns
        -------
        Field
            The processed Field living on the Operator's target or domain,
            depending on mode.
        """
        if mode == self.TIMES:
            # Zero padding (teile entfernen) und time mask
            x = ift.Field(self._x_new_domain, val=x.val[x.shape[0]//4: x.shape[0]//4 * 3, :])
            x = self._M.times(x)

            # Energy Response und (normierte) instrument factors
            lam = self._energy_response(x.val.reshape(-1))
            lam = lam.reshape(self._target.shape)
            lam = QPOutils.scale_and_normalize(lam, self._instrument_factors)

            return ift.Field(self._target, val=lam, dtype=np.float)

        elif mode == self.ADJOINT_TIMES:
            # Adjungierte Energy Response
            x = QPOutils.scale_and_normalize(x, self._instrument_factors)
            x = self._energy_response_adjoint(x.reshape(-1))
            x = x.reshape(self._x_new_domain.shape)

            # do time response (mask)
            x = self._M.times(ift.Field(self._x_new_domain, val=x))

            # Zero padding
            padd = np.zeros(self._domain.shape)
            padd[padd.shape[0]//4: padd.shape[0]//4 * 3, :] = x.val

            return ift.Field(self._domain, val=padd)

        else:
            raise NotImplementedError('Mode %d currently not supported.' % mode)
