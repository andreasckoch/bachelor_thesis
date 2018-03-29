import numpy as np
import nifty4 as ift
import utilities as QPOutils


class Response(object):

    def __init__(self, mask):
        self._mask = mask

    def __call__(self, s):
        return self._mask*s[s.size//4:s.size//4*3]


if __name__ == "__main__":

    start_time = 845
    end_time = 1200
    time_pix = 2**12
    data = QPOutils.get_data(start_time, end_time, time_pix)

    energy_mask = QPOutils.get_energy_sensitivity_mask()
    R_E = Response(energy_mask)
    time_mask = QPOutils.get_time_mask(data)
    R_t = Response(time_mask)


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




"""
