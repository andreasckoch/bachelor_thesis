import nifty4 as ift
import utilities as QPOutils


class EnergyTimeResponse(ift.LinearOperator):

    # Input: 2D Histogram des Signals, domain muss tuple aus Zeit x Energie Räumen sein
    # Output: 3 x 2D Histogramme der Daten, time_bins x instruments x energy channels
    # Bilde Signalvektor auf jeweiligen Datenraum ab (Zeit, Energie[verschiedene Binnings für verschiedene instrumente])
    # Bei feinem Signalvektor weniger Unstimmigkeiten mit Datenbinning!
    # Mask muss Input Dimensions haben!!
    def __init__(self, domain, mask=None):
        super(EnergyTimeResponse, self).__init__()
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(ift.UnstructuredDomain((3, domain.shape[0], 256)))
        self._energy_dicts, self._energies = QPOutils.get_dicts(True, True)
        self._instrument_factors = QPOutils.get_instrument_factors()
        if mask is None:
            self._M = ift.DiagonalOperator(ift.Field.ones(self._domain))
        else:
            self._M = ift.DiagonalOperator(ift.Field(self._domain, mask))

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
        return self.ADJOINT_TIMES

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
        if mode == 1:
            # iterate over energy dimension to get all rows living in time dimension and other way around
            # Input for Time/Energy-Responses need to be ift Fields!

            # Energy Response und Normierung
            x = self._M.times(x)
            lam = QPOutils.energy_response(x, self._energy_dicts, self._energies)
            lam = QPOutils.scale_and_normalize(lam, self._instrument_factors)

            return ift.Field(ift.UnstructuredDomain(lam.shape), val=lam)

        elif mode == 2:
            # Instrumenten response adjungiert multiplizieren
            x = QPOutils.scale_and_normalize(x, 1/self._instrument_factors)
            s = QPOutils.energy_response_adjoint(x, self._domain, self._energy_dicts, self._energies)

            s = self._M.times(s)
            return ift.Field(self._domain, val=s.val)

        else:
            raise NotImplementedError('Mode %d currently not supported.' % mode)


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
        return self.ADJOINT_TIMES

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
        # signal aus richtigem Abschnitt nehmen und photon counts auf channels aufteilen
        # ACHTUNG: alles über maximalem Energy Bin (~127) juckt Response nicht!!!
        if mode == 1:
            lam = QPOutils.energy_response(x, self._energy_dicts, self._energies)

            # Normierung
            lam = QPOutils.scale_and_normalize(lam, self._instrument_factors)

            return ift.Field(ift.UnstructuredDomain(lam.shape), val=lam)
        elif mode == 2:
            # Instrumenten response adjungiert multiplizieren (wie?):
            x = QPOutils.scale_and_normalize(x, 1/self._instrument_factors)
            s = QPOutils.energy_response_adjoint(x, self._domain, self._energy_dicts, self._energies)

            # return signal field
            return ift.Field(domain=self._domain, val=s)
        else:
            raise NotImplementedError('Mode %d currently not supported.' % mode)

        '''
        # Zero Padding
        # Betrachte nur Hälfte des Signal Felds, ACHTUNG: Richtiger Abschnitt sollte durchgegeben werden!!!!!
        s_new_domain = ift.RGSpace(
            (s.size // 2), distances=s.domain[0].distances[0])
        s = ift.Field(s_new_domain, val=s.val[s.size//4: s.size//4 * 3])
        R = ift.GeometryRemover(s.domain)
        '''


class TimeResponse(ift.LinearOperator):

    # Verwende gleiches Binning im Signal und Datenraum
    def __init__(self, domain, mask=None):
        super(TimeResponse, self).__init__()
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(ift.UnstructuredDomain(domain.shape))
        if mask is None:
            self._M = ift.DiagonalOperator(ift.Field.ones(self._target))
        else:
            self._M = ift.DiagonalOperator(ift.Field(self._target, mask))

    def domain(self):
        """The domain on which the Operator's input Field lives."""
        return self._domain

    def target(self):
        """The domain on which the Operator's output Field lives."""
        return self._target

    def capability(self):
        """int : the supported operation modes"""
        return self.TIMES
        return self.ADJOINT_TIMES

    def apply(self, x, mode):
        # Betrachte nur Hälfte des Signal Felds, ACHTUNG: Richtiger Abschnitt sollte durchgegeben werden!!!!!
        # x_new_domain = ift.RGSpace(
        #    (x.size // 2), distances=x.domain[0].distances[0])
        # x = ift.Field(x_new_domain, val=x.val[x.size//4: x.size//4 * 3])

        if mode == 1:
            R = self._M * ift.GeometryRemover(self._domain)
            return R.times(x)
        elif mode == 2:
            x = self._M.times(x)
            return ift.Field(self._domain, val=x.val)
        else:
            raise NotImplementedError('Mode %d currently not supported.' % mode)


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
# class EnergyTimeResponse(ift.LinearOperator):

#     # Input: 2D Histogram des Signals, domain muss tuple aus Zeit x Energie Räumen sein
#     # Output: 3 x 2D Histogramme der Daten, time_bins x instruments x energy channels
#     # Bilde Signalvektor auf jeweiligen Datenraum ab (Zeit, Energie[verschiedene Binnings für verschiedene instrumente])
#     # Bei feinem Signalvektor weniger Unstimmigkeiten mit Datenbinning!
#     def __init__(self, domain, mask=None):
#         super(EnergyTimeResponse, self).__init__()
#         self._domain = ift.DomainTuple.make(domain)
#         self._target = ift.DomainTuple.make(ift.UnstructuredDomain((domain.shape[0], 3, 256)))
#         self._TimeResponse = TimeResponse(domain[0], mask)
#         self._EnergyReponse = EnergyResponse(domain[1])

#     def domain(self):
#         """DomainTuple : the operator's input domain

#             The domain on which the Operator's input Field lives."""
#         return self._domain

#     def target(self):
#         """DomainTuple : the operator's output domain

#             The domain on which the Operator's output Field lives."""
#         return self._target

#     def capability(self):
#         """int : the supported operation modes

#         Returns the supported subset of :attr:`TIMES`, :attr:`ADJOINT_TIMES`,
#         :attr:`INVERSE_TIMES`, and :attr:`ADJOINT_INVERSE_TIMES`,
#         joined together by the "|" operator.
#         """
#         return self.TIMES
#         return self.ADJOINT_TIMES

#     def apply(self, x, mode):
#         """ Applies the Operator to a given `x`, in a specified `mode`.

#         Parameters
#         ----------
#         s : Field
#             The input Field, living on the Operator's domain or target,
#             depending on mode.

#         mode : int
#             - :attr:`TIMES`: normal application
#             - :attr:`ADJOINT_TIMES`: adjoint application
#             - :attr:`INVERSE_TIMES`: inverse application
#             - :attr:`ADJOINT_INVERSE_TIMES` or
#               :attr:`INVERSE_ADJOINT_TIMES`: adjoint inverse application

#         Returns
#         -------
#         Field
#             The processed Field living on the Operator's target or domain,
#             depending on mode.
#         """
#         if mode == 1:
#             # iterate over energy dimension to get all rows living in time dimension and other way around
#             # Input for Time/Energy-Responses need to be ift Fields!
#             out = ift.Field.zeros(self._target)
#             for j in range(x.shape[1]):
#                 x.val[:, j] = self._TimeResponse.times(ift.Field(self._domain[0], val=x.val[:, j])).val
#             for i in range(x.shape[0]):
#                 out.val[i] = self._EnergyReponse.times(ift.Field(self._domain[1], val=x.val[i])).val
#             return out
#         elif mode == 2:
#             # Instrumenten response adjungiert multiplizieren
#             out = ift.Field.zeros(self._domain)
#             for i in range(x.shape[0]):
#                 out.val[i] = self._EnergyReponse.adjoint_times(ift.Field(self._EnergyReponse._target, val=x.val[i])).val
#             for j in range(x.shape[1]):
#                 out[:, j] = self._TimeResponse.adjoint_times(ift.Field(self.TimeResponse._target, val=out.val[:, j])).val
#             return out
#         else:
#             raise NotImplementedError('Mode %d currently not supported.' % mode)
