"""
load the observation data, which is stored as a ConfigObj object.

We do some parsing of the data in routine 'obsdata' to 
get the data in a useful format

"""
from configobj import ConfigObj
from numpy.polynomial import Polynomial
from numpy import argmin
import re
from astropy import units as u

from astropy.coordinates import ICRS
from astropy.time import Time

def obsdata(conf='observations.conf'):
    """ load the observation data """
    C = ConfigObj(conf)

    # map things from ConfigObj to dictionary of useful objects
    obs = {}
    for key, val in C.iteritems():
        if key == 'psrs': 
            obs[key] = parse_psrs(val)
        else:
            obs[key] = parse_tel(key,val)
    return obs

class telescope(dict):
    def __init__(self, name):
        self['name'] = name
        self['observations'] = []

    def nearest_observation(self, t):
        """
        return key of nearest observation to (utc) time 't'.
        A warning is raised if the observation > 1s away 

        """
        if isinstance(t, str):
            t = Time(t, scale='utc')
        
        dts = []
        dates = [self[d]['date'] for d in self['observations']]
        for date in dates:
            dts.append(abs( (t - date).sec ))
        dtmin = argmin(dts)
        key = self['observations'][dtmin] 
        if dts[dtmin] > 1.:
            tmplt = ("Warning, observation {0} is more than 1 second away from request time {1}")
            raise Warning(tmplt.format(key, str(t)))
        return key

    def aro_seq_raw_files(self, key):
        """
        return the ARO sequence and raw files for observation 'key'

        """
        obs = self[key]
        fnbase = obs.get('fnbase', self.get('fnbase', None))
        disk_no = obs.get('disk_no', self.get('disk_no', None))
        node = obs.get('node', self.get('node', None))
        dt = key
        seq_file = (self['seq_filetmplt'].format(fnbase, disk_no[0], node, dt))
        raw_files = [self['raw_filestmplt'].format(fnbase, disk_no[i], node, dt, i)
                     for i in range(3)]
        return seq_file, raw_files
        

class observation(dict):
    def __init__(self, date, val):
        self['date'] = date 
        for k, v in val.iteritems():
            if k == 'ppol' and v.startswith('Polynomial'):
                self[k] = eval(v)
            else:
                self[k] = v

    def get_phasepol(self, time0):
        """
        return the phase polynomial at time0
        (calculated if necessary)
        """
        phasepol = self['ppol']
        if phasepol is None:
            subs = [self['src'], str(self['date'])]
            wrn = "{0} is not configured for time {1} \n".format(*subs)
            wrn += "\tPlease update observations.conf "
            raise Warning(wrn)

        elif not isinstance(phasepol, Polynomial):
            subs = [self['src'], self['date']]
            print("Calculating {0} polyco at {1}".format(*subs))
            from astropy.utils.data import get_pkg_data_filename
            from pulsar.predictor import Polyco

            polyco_file = get_pkg_data_filename(phasepol)
            polyco = Polyco(polyco_file)
            phasepol = polyco.phasepol(time0, rphase='fraction', t0=time0,
                                       time_unit=u.second, convert=True)
        return phasepol

def parse_tel(telname, vals):
    tel = telescope(telname)
    for key, val in vals.iteritems():
        try:
            # then this is an observation
            date = Time(key, scale='utc')
            obs = observation(date, val)
            tel.update({key: obs})
            tel['observations'].append(key)
        except ValueError:
            tel.update({key: val})
    return tel

def parse_psrs(psrs):
    for name, vals in psrs.iteritems():
        if 'coords' not in vals:
            # add a coordinate attribute
            match = re.search("\d{4}[+-]\d+", name)
            if match is not None:
                crds = match.group()
                # set *very* rough position (assumes name format [BJ]HHMM[+-]DD*)
                ra = '{0}:{1}'.format(crds[0:2], crds[2:4])
                dec = '{0}:{1}'.format(crds[5:7], crds[7:]).strip(':')
                vals['coords'] = ICRS(coordstr='{0}, {1}'.format(ra,dec),
                                      unit=(u.hour, u.degree))
            else:
                vals['coords'] = ICRS(coordstr='0, 0',
                                      unit=(u.hour, u.degree))
        else:
            coord = vals['coords']
            if coord.startswith("<ICRS RA"):
                # parse the (poor) ICRS print string
                ra = re.search('RA=[+-]?\d+\.\d+ deg', coord).group()
                dec = re.search('Dec=[+-]?\d+\.\d+ deg', coord).group()
                coord = '{0}, {1}'.format(ra[3:], dec[4:])
            vals['coords'] = ICRS(coordstr=coord)
        if 'dm' in vals:
            vals['dm'] = eval(vals['dm'])
    return psrs
        
if __name__ == '__main__':
    obsdata()
