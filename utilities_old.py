import nifty4 as ift
import numpy as np
import matplotlib.pyplot as plt

def find_dead_times(threshold, data):
    #
    # take in some data field and mark all segments that are longer than threshold and have value 0 
    # using a mask with 0. Doing that using a kernel that yields 0 for sought after segments.
    #
    kernel = ift.Field.ones(ift.UnstructuredDomain(shape = threshold), dtype = np.float64)
    data_dim = data.val.shape
    data_mask = ift.Field.ones(ift.UnstructuredDomain(shape = data_dim), dtype = np.float64)
    
    # simple indicator function to count dead segments
    NotData = False
    dead_count = 0

    for i in range(data_dim[0]):
        if i == data_dim[0] - (threshold - 1):
            break
        if np.sum((kernel * data.val[i:i + threshold]).val) == 0:
            data_mask.val[i:i + threshold] = 0
            if NotData == False:
                dead_count += 1
            NotData = True
        else:
            NotData = False


    return data_mask, dead_count






def build_response(signal_domain, data_mask):
    # zero pad so dimensions of data space matches signal space

    padded_data = ift.Field.zeros(signal_domain, dtype = np.float64)
    index = padded_data.val.shape[0] // 2 - data_mask.val.shape[0] // 2
    padded_data.val[index: index + data_mask.val.shape[0]] = data_mask.val
    M = ift.DiagonalOperator(ift.Field(signal_domain, val = padded_data))

    # mock Response
    R = ift.GeometryRemover(signal_domain)*M

    return R







def plot_data():
    # plotte 2D plot (Energy-Zeit) aus SGR1806 Daten mit Photon Counts als Farbverlauf

    data_path = "/home/andi/bachelor/data/energyandtime/SGR1806_time_energykeV.txt"

    data = np.loadtxt(data_path).transpose()
    npix = 2**18
    # tmp = np.digitize(data[0], np.array([data[-1]*i/npix for i in range(npix)]))
    #_, time_bins = np.unique(tmp, return_counts=True)
    # time_bins = np.array([data[-1]*i/npix for i in range(npix)])
    energy_bins = np.insert(np.unique(data[1]), 0, 0)
    binned_data, _, _ = np.histogram2d(data[1], data[0], bins=[energy_bins, npix])

    plt.imshow(binned_data)
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.savefig("SGR1806_t_E_histogram2.png")

    return binned_data, npix


if __name__ == "__main__":

    total_volume = 3000.0
    binned_data, npix = plot_data()
    collapsed_data = np.ma.sum(binned_data, axis = 0)
    ift_data = ift.Field(ift.UnstructuredDomain(shape = collapsed_data.shape), val = collapsed_data)

    # find dead time segments, here: 5 zeros in a row
    data_mask, dead_count = find_dead_times(5, ift_data)
    print(dead_count)

    # define signal space and build response
    # signal space is wrong!!!
    x1 = ift.RGSpace(npix, distances=total_volume / npix)
    R = build_response(x1, data_mask)
    print(R)
    ift.plot(ift.Field(x1, val=ift_data.val), name='./data/data.png')
    ift.plot(ift.Field(x1, val=data_mask.val), name='./data/data_mask.png')

    # Next Step: Plot only a "window" of ift_data, from ca 600s to 1400s