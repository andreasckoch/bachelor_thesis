import numpy as np
import nifty4 as ift
import utilities as QPOutils


class EnergyResponse(ift.LinearOperator):

    # Input: 1D Histogram des Signals
    # Output: 1D Histogram der Daten
    # Bilde Signalvektor auf jeweiligen Datenraum ab (Zeit, Energie[verschiedene Binnings für verschiedene instrumente])
    # Bei feinem Signalvektor weniger Unstimmigkeiten mit Datenbinning!
    def __init__(self):
        self._energy_dicts, self._energies = QPOutils.get_dicts(True, True)
        self._factors = QPOutils.effectve_area_and_energy_width()

    def __call__(self, s):
        # Betrachte nur Hälfte des Signal Felds, ACHTUNG: Richtiger Abschnitt sollte durchgegeben werden!!!!!
        s_new_domain = ift.RGSpace(
            (s.size // 2), distances=s.domain[0].distances[0])
        s = ift.Field(s_new_domain, val=s.val[s.size//4: s.size//4 * 3])

        # 4 Schritte Response (Instrumente, Energy Bins, Channels, Effective Area + Energy Bin Breite)
        s = QPOutils.energy_response(s, self._energy_dicts, self._energies)  # dim: 3 x 256
        s *= self._factors

        R = ift.GeometryRemover(s.domain)
        return R.times(s)  # exp passiert nicht in Response


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


if __name__ == "__main__":

    start_time = 845
    end_time = 1200
    time_pix = 2**12
    data = QPOutils.get_data(start_time, end_time, time_pix)

    R_E = Response()
    #time_mask = QPOutils.get_time_mask(data)
    #R_t = Response(time_mask)

    # erzeuge mock signal um Response zu testen
    s = QPOutils.mock_signal()

    lam = R_E(ift.exp(s))


"""
- Wie gebe ich später Daten in das Modell rein? Ich habe meine drei Histogramme in binned_data untergebracht.
- Wie funktioniert nun die Übersetzung von Channel zu Energie?


- Channels in Energie übersetzen und diese Daten einlesen (alle Instrumente summiert, schon geschehen),
   128 energy_bins nutzen
- Response bauen (an NIFTy vorgaben halten, schauen, wie andere Responses implementiert wurden)
   und testen (funktioniert der Übergang vom Datenraum zum Signalraum und zurück)


- Response bauen (maske in zeit+ zeropadding in zeit und energie) 
   maske als vektor und dann mit field und diagonal operator wie in mock data.py
   zeropadding implizit machen (nicht nullen an vektor hängen), sondern nur den 
   mittlere Ausschnitt von s returnen (im signalraum)
- testen
- inference modell bauen (d4po solver)


Total Photon Counts per Instrument (SGR1806 data): {PCU0: 341909, PCU2: 335600, PCU3: 329606}





setze initial guesses
setzte uncertainites
setze initial parameter value

definiere domains
baue Response R

lese data ein

packe alles in Problem P

lasse solver über P laufen




 3.4.2018:
Plan für morgen:
Andi: Mock Signals als Histogram
Marvin: Response fertigbauen (Histogram Mapping)

"""
