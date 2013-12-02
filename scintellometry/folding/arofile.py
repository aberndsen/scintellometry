from __future__ import division

import numpy as np
import os
import re
from astropy.table import Table
from astropy import units as u
from astropy.time import Time

from mpi4py import MPI
from psrfits_tools import psrFITS


# ARO defaults for psrfits HDUs
_ARO_defs = {}
_ARO_defs['PRIMARY'] = {'TELESCOP':'Algonquin',
                        'IBEAM':1, 'FD_POLN':'LIN',
                        'OBS_MODE':'SEARCH',
                        'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                        'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                        'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                        'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                        'OBSNCHAN':20, 'CHAN_DM':0,
                        'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                        'SCANLEN':1, 'FA_REQ':0,
                        'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                        'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0}
samplerate = 200. * u.MHz
_ARO_defs['SUBINT']  = {'INT_TYPE': 'TIME',
                        'SCALE': 'FluxDen',
                        'POL_TYPE': 'AABB',
                        'NPOL':2,
                        'TBIN':(1./samplerate).to('s').value,
                        'NBIN':1, 'NBIN_PRD':1,
                        'PHS_OFFS':0,
                        'NBITS':1,
                        'ZERO_OFF':0, 'SIGNINT':0,
                        'NSUBOFFS':0,
                        'NCHAN':1,
                        'CHAN_BW':1,
                        'DM':0, 'RM':0, 'NCHNOFFS':0,
                        'NSBLK':1}

class multifile(psrFITS):
    """ 
    A class for ARO data. We subclass psrFITS, so it acts like 
    a FITS file with a PRIMARY and SUBINT header

    """
    def __init__(self, sequence_file, raw_voltage_files, recsize=2**25,
                 dtype='4bit', comm=None):
        # initialize the HDU's
        psrFITS.__init__(self, hdus=['SUBINT'])
        for hdu, defs in _ARO_defs.iteritems():
            for card, val in defs.iteritems():
                self[hdu].header.update(card, val)
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm
        self.sequence_file = sequence_file
        self.sequence = Table(np.loadtxt(sequence_file, np.int32),
                              names=['seq', 'raw'])
        self.sequence.sort('seq')

        # get start date from sequence filename
        arodate = re.search('\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                            os.path.basename(sequence_file))
        if arodate:
            isot = arodate.group()
            # convert time to UTC; dates given in EDT
            self.time0 = Time(isot, scale='utc') + 4*u.hr
            self['PRIMARY'].header['DATE-OBS'] = self.time0.iso
        else:
            self.time0 = None
        # MPI.File.Open doesn't handle files with ":"
        self.fh_raw = []
        self.fh_links = []
        for raw in raw_voltage_files:
            fname, islnk = good_name(os.path.abspath(raw))
            self.fh_raw.append(MPI.File.Open(self.comm, fname,
                                             amode=MPI.MODE_RDONLY))
            if islnk:
                self.fh_links.append(fname)
        self.recsize = recsize
        self.index = 0
        self.seq0 = self.sequence['seq'][0]

        # other useful ARO defaults
        self.dtype = '4bit'
        self.fedge = 200. * u.MHz
        self.fedge_at_top = True
        self.samplerate = 200. * u.MHz

    def nskip(self, date, time0=None):
        """
        return the number of records needed to skip from start of
        file to iso timestamp 'date'.

        Optionally:
        time0 : use this start time instead of self.time0
                either a astropy.time.Time object or string in 'utc'

        """
        if time0 is None:
            time0 = self.time0
        elif isinstance(time0, str):
            time0 = Time(time0, scale='utc')
        
        if time0 is None:
            print("time0 not defined: %s did not match iso time string" 
                  % (self.sequence_file))
            nskip = None
        else:
            dt = (Time(date, scale='utc')-time0)
            nskip = int(round(
                (dt/(self.recsize*2 / self.samplerate))
                .to(u.dimensionless_unscaled)))
        return nskip

    def seek(self, offset):
        assert offset % self.recsize == 0
        self.index = offset // self.recsize
        for i, fh in enumerate(self.fh_raw):
            fh.Seek(np.count_nonzero(self.sequence['raw'][:self.index] == i) *
                    self.recsize)

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)

    def read(self, size):
        assert size == self.recsize
        if self.index == len(self.sequence):
            raise EOFError
        self.index += 1
        # so far, only works for continuous data, so ensure we're not missing
        # any sequence numbers
        if self.sequence['seq'][self.index-1] != self.index-1 + self.seq0:
            raise IOError("multifile sequence numbers have to be consecutive")
        i = self.sequence['raw'][self.index-1]
        # MPI needs a buffer to read into
        z = np.zeros(size, dtype='i1')
        self.fh_raw[i].Iread(z)
        return z

    def __repr__(self):
        return ("<open multifile raw_voltage_files {} "
                "using sequence file '{}' at index {}>"
                .format(self.fh_raw, self.sequence_file, self.index))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def good_name(f):
    """
    MPI.File.Open can't process files with colons.
    This routine checks for such cases and creates a well-named link to the file.

    Returns (good_name, islink)
    """
    if f is None: return f

    fl = f
    newlink = False
    if ':' in f:
        #fl = tempfile.mktemp(prefix=os.path.basename(f).replace(':','_'), dir='/tmp')
        fl = os.path.join('/tmp', os.path.dirname(f).replace('/','_') + '__' + os.path.basename(f).replace(':','_'))
        if not os.path.exists(fl):
            try:
                os.symlink(f, fl)
            except(OSError):
                pass
            newlink = True
    return fl, newlink
