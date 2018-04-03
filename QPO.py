import numpy as np
import nifty4 as ift
import utilities as QPOutils


class Response(object):

    # Input: 1D Histogram des Signals
    # Output: 1D Histogram der Daten
    # Bilde Signalvektor auf jeweiligen Datenraum ab (Zeit, Energie[verschiedene Binnings für verschiedene instrumente])
    # Bei feinem Signalvektor weniger Unstimmigkeiten mit Datenbinning!
    def __init__(self, mask = None):
        self._mask = mask

    def __call__(self, s):

        # Betrachte nur Hälfte des Signal Felds
        s_new_domain = ift.RGSpace((s.size // 2), distances = s.total_volume() / s.size)
        s = ift.Field(s_new_domain, val = s.val[s.size//4 : s.size//4 * 3])
        

        if self._mask != None:
          M = ift.DiagonalOperator(ift.Field(s.domain, val = self._mask))
        else:
          M = ift.Field.ones(s.domain)
        
        R = ift.GeometryRemover(s.domain) * M
        return R.times(s)
        #return self._mask.val*s.val[s.val.shape//4:s.val.shape//4*3]  # implizites Ausschneiden von signal vektor





if __name__ == "__main__":

    start_time = 845
    end_time = 1200
    time_pix = 2**12
    #data = QPOutils.get_data(start_time, end_time, time_pix)

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




"""
