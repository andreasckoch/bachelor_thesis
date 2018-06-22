# Bayesian Spectral and Temporal Signal Map Reconstruction of SGR 1806-20 giant flare

## Dependencies:
Nifty, D4po, Numpy, Scipy, Matplotlib

## How To Use:
1) Get the required data files: SGR1806_time_PCUID_energychannel.txt and
energy_channels.txt [edited version with every entry in its left column being
only one energy channel!].

2) Specify all parameters in main.py [every parameter is editable except for
`e_pix`] and execute it to start the reconstruction. It will initialise the D4PO
problem class building the response as is specified in the file response.py
[using several functions of utilities.py] and the data as is specified in
utilities.py's function get_data().

3) For stronger convergence criteria, adjust `d_h` and `d_h_p` in solver.py's
`make_convergence_criteria()`.

If any questions remain, please contact andreaskoch2222@gmail.com.

NIFTy: `723089ab3b67cd384f2a9153bdf99f7dbd5f4138`.
D4PO: `438adb2765af3034d4012a5fa797c8d9b2d84ec3`.
