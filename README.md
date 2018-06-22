# Bayesian Spectral and Temporal Signal Map Reconstruction of SGR 1806-20

## Dependencies:
Nifty, D4po, Numpy, Scipy, Matplotlib

## How To Use:
1) Get the required data files: SGR1806_time_PCUID_energychannel.txt and energy_channels.txt [edited version with every entry in its left column being only one energy channel!]
2) Create a local file named constants.py with the locations of SGR1806_time_PCUID_energychannel.txt given as variable "data_path" and energy_channels.txt as variable "energy_path"
3) Specify all parameters in main.py [every parameter is editable except for e_pix] and execute it to start the reconstruction. It will initialise the D4PO problem class building the response as is specified in the file response.py [using several functions of utilities.py] and the data as is specified in utilities.py's function get_data().
4) For stronger convergence criteria, adjust d_h and d_h_p in solver.py's make_convergence_criteria().

If any questions remain, please contact andreaskoch2222@gmail.com.