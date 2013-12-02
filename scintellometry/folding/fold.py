""" FFT and Fold data """
from __future__ import division, print_function

from inspect import getargspec
import numpy as np
import astropy.units as u

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.scipy_fftpack import rfft, rfftfreq, irfft
    _fftargs = {'threads': 2,
                'planner_effort': 'FFTW_ESTIMATE'}
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # use FFT from scipy, since unlike numpy it does not cast up to complex128
    from scipy.fftpack import rfft, rfftfreq, irfft
    _fftargs = {}

from fromfile import fromfile

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc

# size in bytes of records read from file (simple for ARO: 1 byte/sample)
# double since we need to get ntint samples after FFT
_bytes_per_sample = {np.int8: 2, '4bit': 1}

def fold(fh1, comm, dtype, samplerate, fedge, fedge_at_top, nchan,
         nt, ntint, nskip, ngate, ntbin, ntw, dm, fref, phasepol,
         dedisperse='incoherent',
         do_waterfall=True, do_foldspec=True, verbose=True,
         progress_interval=100, rfi_filter_raw=None, rfi_filter_power=None):
    """
    FFT data, fold by phase/time and make a waterfall series
    
    Parameters
    ----------
    fh1 : file handle
        handle to file holding voltage timeseries
    comm: MPI communicator or None
    dtype : numpy dtype or '4bit' or '1bit'
        way the data are stored in the file
    samplerate : float
        rate at which samples were originally taken and thus double the
        band width (frequency units)
    fedge : float
        edge of the frequency band (frequency units)
    fedge_at_top: bool
        whether edge is at top (True) or bottom (False)
    nchan : int
        number of frequency channels for FFT
    nt, ntint : int
        total number nt of sets, each containing ntint samples in each file
        hence, total # of samples is nt*ntint, with each sample containing
        a single polarisation
    nskip : int
        number of records (ntint * nchan * 2 / 2 bytes) to skip
    ngate, ntbin : int
        number of phase and time bins to use for folded spectrum
        ntbin should be an integer fraction of nt
    ntw : int
        number of time samples to combine for waterfall (does not have to be
        integer fraction of nt)
    dm : float
        dispersion measure of pulsar, used to correct for ism delay
        (column number density)
    fref: float
        reference frequency for dispersion measure
    phasepol : callable
        function that returns the pulsar phase for time in seconds relative to
        start of part of the file that is read (i.e., ignoring nhead)
    dedisperse : None or string
        None, 'incoherent', 'coherent', 'by-channel'
    do_waterfall, do_foldspec : bool
        whether to construct waterfall, folded spectrum (default: True)
    verbose : bool or int
        whether to give some progress information (default: True)
    progress_interval : int
        Ping every progress_interval sets
        
    """
    if comm is None:
        rank = 0
        size = 1
    else:
        rank = comm.rank
        size = comm.size

    # initialize folded spectrum and waterfall
    foldspec = np.zeros((nchan, ngate, ntbin))
    icount = np.zeros((nchan, ngate, ntbin), dtype=np.int64)
    nwsize = nt*ntint//ntw
    waterfall = np.zeros((nchan, nwsize))

    # size in bytes of records read from file (simple for ARO: 1 byte/sample)
    # double since we need to get ntint samples after FFT
    # Marten: double-check this part
    itemsize = _bytes_per_sample.get(dtype, None)
    if itemsize is None:
        itemsize = np.dtype(dtype).itemsize
    recsize = nchan*ntint*itemsize
    if verbose:
        print('Reading from {}'.format(fh1))

    if nskip > 0:
        if verbose and rank == 0:
            print('Skipping {0} records = {1} bytes'
                  .format(nskip, nskip*recsize))
        if size == 1:
            # MPI: only skip here if we are not threaded, otherwise seek in for-loop
            fh1.seek(nskip * recsize)

    dt1 = (1./samplerate).to(u.s)
    # need 2*nchan real-valued samples for each FFT
    dtsample = nchan * 2 * dt1
    tstart = dtsample * ntint * nskip

    # pre-calculate time delay due to dispersion in coarse channels
    freq = (fedge - rfftfreq(nchan*2, dt1.value) * u.Hz
            if fedge_at_top
            else
            fedge + rfftfreq(nchan*2, dt1.value) * u.Hz)
    # [::2] sets frequency channels to numerical recipes ordering
    dt = (dispersion_delay_constant * dm *
          (1./freq[::2]**2 - 1./fref**2)).to(u.s).value
    if dedisperse in {'coherent', 'by-channel'}:
        # pre-calculate required turns due to dispersion
        fcoh = (fedge - rfftfreq(nchan*2*ntint, dt1.value) * u.Hz
                if fedge_at_top
                else
                fedge + rfftfreq(nchan*2*ntint, dt1.value) * u.Hz)
        # set frequency relative to which dispersion is coherently corrected
        if dedisperse == 'coherent':
            _fref = fref
        else:
            # _fref = np.round((fcoh * dtsample).to(1).value) / dtsample
            _fref = np.repeat(freq.value, ntint) * freq.unit
        # (check via eq. 5.21 and following in
        # Lorimer & Kramer, Handbook of Pulsar Astrono
        dang = (dispersion_delay_constant * dm * fcoh *
                (1./_fref-1./fcoh)**2) * 360. * u.deg
        # order of frequencies is r[0], r[1],i[1],...r[n-1],i[n-1],r[n]
        # for 0 and n need only real part, but for 1...n-1 need real, imag
        # so just get shifts for r[1], r[2], ..., r[n-1]
        dang = dang.to(u.rad).value[1:-1:2]
        dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)

    for j in xrange(rank, nt, size):
        if verbose and j % progress_interval == 0:
            print('Doing {:6d}/{:6d}; time={:18.12f}'.format(
                j+1, nt, (tstart+dtsample*j*ntint).value))  # time since start

        # just in case numbers were set wrong -- break if file ends
        # better keep at least the work done
        try:
            # data just a series of bytes, each containing one 8 bit or
            # two 4-bit samples (set by dtype in caller)
            if size > 1:
                fh1.seek(nskip*recsize + j*recsize*itemsize)
            if verbose >= 1:
                print("READ", j, fh1.index, fh1.fh_raw[fh1.sequence['raw'][fh1.index]].Get_position())
            raw = fromfile(fh1, dtype, recsize)
        except(EOFError, IOError) as exc:
            print("Hit {}; writing pgm's".format(exc))
            break
        if verbose >= 2 :
            print("Read {} items".format(raw.size), end="")

        if rfi_filter_raw:
            raw = rfi_filter_raw(raw)
            print("... raw RFI", end="")

        vals = raw.astype(np.float32)
        if dedisperse in {'coherent', 'by-channel'}:
            fine = rfft(vals, axis=0, overwrite_x=True, **_fftargs)
            fine_cmplx = fine[1:-1].view(np.complex64)
            fine_cmplx *= dd_coh  # this overwrites parts of fine, as intended
            vals = irfft(fine, axis=0, overwrite_x=True, **_fftargs)
            if verbose >= 2:
                print("... dedispersed", end="")

        chan2 = rfft(vals.reshape(-1, nchan*2), axis=-1,
                     overwrite_x=True, **_fftargs)**2
        # rfft: Re[0], Re[1], Im[1], ..., Re[n/2-1], Im[n/2-1], Re[n/2]
        # re-order to Num.Rec. format: Re[0], Re[n/2], Re[1], ....
        power = np.hstack((chan2[:,:1]+chan2[:,-1:],
                           chan2[:,1:-1].reshape(-1,nchan-1,2).sum(-1)))

        if verbose >= 2:
            print("... power", end="")

        if rfi_filter_power:
            power = rfi_filter_power(power)
            print("... power RFI", end="")

        # current sample positions in stream
        isr = j*ntint + np.arange(ntint)

        if do_waterfall:
            # loop over corresponding positions in waterfall
            for iw in xrange(isr[0]//ntw, isr[-1]//ntw + 1):
                if iw < nwsize:  # add sum of corresponding samples
                    waterfall[:,iw] += np.sum(power[isr//ntw == iw],
                                              axis=0)
            if verbose >=  2:
                print("... waterfall", end="")

        if do_foldspec:
            tsample = (tstart + isr*dtsample).value  # times since start
            ibin = j*ntbin//nt  # bin in the time series: 0..ntbin-1

            for k in xrange(nchan):
                if dedisperse == 'coherent':
                    t = tsample  # already dedispersed
                else:
                    t = tsample - dt[k]  # dedispersed times

                phase = phasepol(t)  # corresponding PSR phases
                iphase = np.remainder(phase*ngate,
                                      ngate).astype(np.int)
                # sum and count samples by phase bin
                foldspec[k, :, ibin] += np.bincount(iphase, power[:, k], ngate)
                icount[k, :, ibin] += np.bincount(iphase, power[:, k] != 0.,
                                                  ngate)

            if verbose >= 2:
                print("... folded", end="")

        if verbose >= 2:
            print("... done")

    if verbose:
        print('read {0:6d} out of {1:6d}'.format(j+1, nt))

    return foldspec, icount, waterfall


class Folder(dict):
    """
    convenience class to populate many of the 'fold' arguments
    from the psrfits headers of a datafile

    """
    def __init__(self, fh, **kwargs):

        # get the required arguments to 'fold'
        fold_args = getargspec(fold)
        fold_argnames = fold_args.args
        fold_defaults = fold_args.defaults
        # and set the defaults
        Nargs = len(fold_args.args)
        Ndefaults = len(fold_defaults)
        for i, v in enumerate(fold_defaults):
            self[fold_argnames[Nargs - Ndefaults + i]] = v
            
        # get some defaults from fh (may be overwritten by kwargs)
        self['dtype'] = fh.dtype
        self['samplerate'] = (1./fh['SUBINT'].header['TBIN']*u.Hz).to(u.MHz)
        self['fedge'] = fh.fedge
        self['fedge_at_top'] = fh.fedge
        self['nchan'] = fh['SUBINT'].header['NCHAN']
        self['ngate'] = fh['SUBINT'].header['NBIN_PRD']

        # update arguments with passed kwargs
        for k in kwargs:
            if k in fold_argnames:
                self[k] = kwargs[k]
            else:
                print("{} not needed for fold routine".format(k))
        # warn of missing 
        missing = [k for k in fold_argnames[1:] if k not in self]
        if len(missing) > 0:
            print("Missing 'fold' arguments: {}".format(missing))
                
    def __call__(self, fh, comm=None):
        ifold, icount, water = fold(fh, comm=comm, **self)
        return ifold, icount, water

