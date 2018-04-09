import nifty4 as ift
import numpy as np
import utilities as QPOutils


class EnergyResponse(ift.LinearOperator):

    # Input: 1D Histogram des Signals
    # Output: 1D Histogram der Daten
    # Bilde Signalvektor auf jeweiligen Datenraum ab (Zeit, Energie[verschiedene Binnings für verschiedene instrumente])
    # Bei feinem Signalvektor weniger Unstimmigkeiten mit Datenbinning!
    def __init__(self, domain):
        super(EnergyResponse, self).__init__()
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(ift.UnstructuredDomain((3, 256)))
        self._energy_dicts, self._energies = QPOutils.get_dicts(True, True)
        self._instrument_factors = QPOutils.get_instrument_factors()

    def domain(self):
        """DomainTuple : the operator's input domain

            The domain on which the Operator's input Field lives."""
        return self._domain

    def target(self):
        """DomainTuple : the operator's output domain

            The domain on which the Operator's output Field lives."""
        return self._target

    def capability(self):
        """int : the supported operation modes

        Returns the supported subset of :attr:`TIMES`, :attr:`ADJOINT_TIMES`,
        :attr:`INVERSE_TIMES`, and :attr:`ADJOINT_INVERSE_TIMES`,
        joined together by the "|" operator.
        """
        return self.TIMES

    def apply(self, s, mode):
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
        if mode != 1:
            raise NotImplementedError('Mode %d currently not supported.' % mode)

        # Zero padding (teile entfernen)

        # signal aus richtigem Abschnitt nehmen und photon counts auf channels aufteilen
        # ACHTUNG: alles über maximalem Energy Bin (~127) juckt Response nicht!!!
        lam = QPOutils.energy_response(s, self._energy_dicts, self._energies)

        # Normierung
        lam = QPOutils.scale_and_normalize(lam, self._instrument_factors)

        return ift.Field(ift.UnstructuredDomain(lam.shape), val=lam)

        '''
        # Zero Padding
        # Betrachte nur Hälfte des Signal Felds, ACHTUNG: Richtiger Abschnitt sollte durchgegeben werden!!!!!
        s_new_domain = ift.RGSpace(
            (s.size // 2), distances=s.domain[0].distances[0])
        s = ift.Field(s_new_domain, val=s.val[s.size//4: s.size//4 * 3])
        R = ift.GeometryRemover(s.domain)
        '''


class TimeResponse(ift.LinearOperator):

    # Input/Output wie bei Energie
    def __init__(self, mask):
        self._mask = mask

    def __call__(self, s):
        # Betrachte nur Hälfte des Signal Felds, ACHTUNG: Richtiger Abschnitt sollte durchgegeben werden!!!!!
        s_new_domain = ift.RGSpace(
            (s.size // 2), distances=s.domain[0].distances[0])
        s = ift.Field(s_new_domain, val=s.val[s.size//4: s.size//4 * 3])

        M = ift.DiagonalOperator(ift.Field(s.domain, val=self._mask))

        R = ift.GeometryRemover(s.domain) * M
        return R.times(s)


"""

Total Photon Counts per Instrument (SGR1806 data): {PCU0: 341909, PCU2: 335600, PCU3: 329606}





setze initial guesses
setzte uncertainites
setze initial parameter value

definiere domains
baue Response R

lese data ein

packe alles in Problem P

lasse solver über P laufen

"""
