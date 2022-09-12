from ase.io import read
from ase.build import molecule
from dscribe.descriptors import SOAP
from quippy.descriptors import Descriptor

desdictSOAP = {"nmax": 6,
           "lmax": 5,
           "rcut": 3.0,
           "sigma": 3.0/8,
           "periodic": False}
soapD = SOAP(species=[1,3],r_cut=desdictSOAP["rcut"],n_max=desdictSOAP["nmax"],l_max=desdictSOAP["lmax"],sigma=desdictSOAP["sigma"])
