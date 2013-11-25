from __future__ import division

import numpy as np
import os
import tempfile
from multifile import multifile

from mpi4py import MPI

class mpimultifile(multifile):
    def __init__(self, sequence_file, raw_voltage_files, recsize=2**25, comm=None):
        super(mpimultifile, self).__init__(sequence_file, raw_voltage_files, recsize=2**25)
        self.comm = comm
        self.myindex = 0
        self.fh_raw = []
        # MPI.File.Open doesn't handle files with ":"'s, track tmp files
        self.fh_links = []
        for raw in raw_voltage_files:
             fname, islnk = good_name(raw)
             #self.fh_raw.append(MPI.File.Open(comm, fname, amode=MPI.MODE_RDONLY))
             self.fh_raw.append(MPI.File.Open(MPI.COMM_SELF, fname, amode=MPI.MODE_RDONLY))
             if islnk:
                 self.fh_links.append(fname)
        unique = np.unique(self.sequence['raw'])
        self.offsets = {}
        for k in unique:
            self.offsets[k] = -1

    def close(self):
        for fh in self.fh_raw:
            fh.Close()
        for fh in self.fh_links:
            if os.path.exists(fh):
                os.unlink(fh)    

    def read(self, size):
        """ fold_aro2.py assumed we read sequentially, but with mpi we read
            every comm.size chunk. This read routine adapts the old sequential
            reads to the mpi ones.
        """
        assert size == self.recsize

        preskip = self.sequence['raw'][self.index:self.index+self.comm.rank]
        for k in preskip:
            self.fh_raw[k].Seek(self.fh_raw[k].Get_position() + size)
            self.index += 1

        if self.index >= len(self.sequence):
            raise EOFError
        self.index += 1
        # so far, only works for continuous data, so ensure we're not missing
        # any sequence numbers
        if self.sequence['seq'][self.index-1] != self.index-1 + self.seq0:
            raise IOError("multifile sequence numbers have to be consecutive")
        i = self.sequence['raw'][self.index-1]
        # MPI read requires the buffer
        z = np.zeros(size, dtype='i1')
        self.fh_raw[i].Iread(z)

        postskip = self.sequence['raw'][self.index:self.index+(self.comm.size-self.comm.rank-1)]
        for k in postskip:
            self.fh_raw[k].Seek(self.fh_raw[k].Get_position() + size)
            self.index += 1

        return z

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

