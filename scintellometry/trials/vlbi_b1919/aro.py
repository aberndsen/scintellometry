from __future__ import division, print_function

import numpy as np
import astropy.units as u
from astropy.time import Time

#from scintellometry.folding.fold_aro2 import fold
from scintellometry.folding.fold import Folder
from scintellometry.folding.pmap import pmap
from scintellometry.folding.arofile import multifile

from observations import obsdata

from mpi4py import MPI

MAX_RMS = 4.2


def rfi_filter_raw(raw):
    rawbins = raw.reshape(-1, 1048576)  # note, this is view!
    rawbins *= (rawbins.std(-1, keepdims=True) < MAX_RMS)
    return raw


def rfi_filter_power(power):
    return np.clip(power, 0., MAX_RMS**2 * power.shape[-1])


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    Obs = obsdata()

    # pulsar parameters
    psr = 'B1919+21'
    # psr = 'B2016+28'
    # psr = 'B0329+54'
    # psr = 'B0823+26'
    # psr = 'J1810+1744'
    dm = Obs['psrs'][psr]['dm']
    telescope = 'aro'
    date_dict = {'B0823+26': '2013-07-24T15:06:16',
                 'B1919+21': '2013-07-25T18:14:20',
                 # 'J1810+1744': '2013-07-27T16:55:17'}
                 'J1810+1744': '2013-07-26T16:30:37'}

    # find nearest observation to date_dict[psr]
    # ARO keys are ISOT time strings
    obskey = Obs[telescope].nearest_observation(date_dict[psr])

    seq_file, raw_files = Obs[telescope].aro_seq_raw_files(obskey)

    #***TODO: apply LOFAR polyphase instead
    nchan = 512  # frequency channels to make
    ngate = 512  # number of bins over the pulsar period
    # total_size = sum(os.path.getsize(fil) for fil in raw_files)
    # nt = total_size // recsize
    nt = 18  # each 32MB set has 2*2**25/2e8=0.33554432 s, so 180 -> ~1 min

    ntbin = 12  # number of bins the time series is split into for folding

    fref = 150. * u.MHz  # ref. freq. for dispersion measure

    verbose = 'very'
    do_waterfall = True
    do_foldspec = True
    dedisperse = 'incoherent'

    with multifile(seq_file, raw_files) as fh1:
        time0 = fh1.time0
        phasepol = Obs[telescope][obskey].get_phasepol(time0)

        recsize = fh1.recsize  
        ntint = recsize*2//(2 * nchan)    # number of samples after FFT
        ntw = min(10200, nt*ntint)  # number of samples to combine for waterfall

        samplerate = fh1.samplerate

        # number of records to skip
        nskip = fh1.nskip('2013-07-25T22:15:00')    # number of records to skip
        if verbose and comm.rank == 0:
            print("Using start time {0} and phase polynomial {1}"
                  .format(time0, phasepol))
            print("Skipping {0} records and folding {1} records to cover "
                  "time span {2} to {3}"
                  .format(nskip, nt,
                          time0 + nskip * recsize * 2 / samplerate,
                          time0 + (nskip+nt) * recsize * 2 / samplerate))

        # set the default parameters to fold
        # Note, some parameters may be in fh1's HDUs, or fh1.__getitem__
        # but these are overwritten if explicitly sprecified in Folder
        folder = Folder(
                        fh1, nchan=nchan,
                        nt=nt, ntint=ntint, nskip=nskip, ngate=ngate,
                        ntbin=ntbin, ntw=ntw, dm=dm, fref=fref,
                        phasepol=phasepol,
                        dedisperse=dedisperse, do_waterfall=do_waterfall,
                        do_foldspec=do_foldspec, verbose=verbose, progress_interval=1,
                        rfi_filter_raw=rfi_filter_raw,
                        rfi_filter_power=None)
        myfoldspec, myicount, mywaterfall = folder(fh1, comm=comm)

    if do_waterfall:
        waterfall = np.zeros_like(mywaterfall)
        comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)
        if comm.rank == 0:
            nonzero = waterfall == 0.
            waterfall -= np.where(nonzero,
                                  np.sum(waterfall, 1, keepdims=True) /
                                  np.sum(nonzero, 1, keepdims=True), 0.)
            np.save("aro{0}waterfall_{1}.npy".format(psr, node), waterfall)

    if do_foldspec:
        foldspec = np.zeros_like(myfoldspec)
        icount = np.zeros_like(myicount)
        comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
        comm.Reduce(myicount, icount, op=MPI.SUM, root=0)

        if comm.rank == 0:
            np.save("aro{0}foldspec_{1}".format(psr, node), foldspec)
            np.save("aro{0}icount_{1}".format(psr, node), icount)
            # get normalised flux in each bin (where any were added)
            nonzero = icount > 0
            f2 = np.where(nonzero, foldspec/icount, 0.)
            # subtract phase average and store
            f2 -= np.where(nonzero,
                           np.sum(f2, 1, keepdims=True) /
                           np.sum(nonzero, 1, keepdims=True), 0)
            foldspec1 = f2.sum(axis=2)
            fluxes = foldspec1.sum(axis=0)
            foldspec3 = f2.sum(axis=0)

            with open('aro{0}flux_{1}.dat'.format(psr, node), 'w') as f:
                for i, flux in enumerate(fluxes):
                    f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))

    plots = True
    if plots and comm.rank == 0:
        if do_waterfall:
            w = waterfall.copy()
            pmap('aro{0}waterfall_{1}.pgm'.format(psr, node),
                 w, 1, verbose=True)
        if do_foldspec:
            pmap('aro{0}folded_{1}.pgm'.format(psr, node),
                 foldspec1, 0, verbose)
            pmap('aro{0}foldedbin_{1}.pgm'.format(psr, node),
                 f2.transpose(0,2,1).reshape(nchan,-1), 1, verbose)
            pmap('aro{0}folded3_{1}.pgm'.format(psr, node),
                 foldspec3, 0, verbose)
