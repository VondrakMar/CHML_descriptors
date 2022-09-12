from quippy.descriptors import Descriptor
from ase.io import read
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


Descriptor_description = f"soap cutoff=4 zeta=2 delta = 1.0 atom_sigma =1.0 l_max=2 n_max=3 n_Z=2 Z={{1 3}} n_species=2 species_Z={{1 3}}"
soap = Descriptor(Descriptor_description)

def _descr_deriv_atoms(molecules):
    soaps = soap.calc(molecules, grad=True)
    soap_atoms=[]
    soap_derivatives = []
    neigh_id = []
    for mol in soaps:
        for at in mol["data"]:
            soap_atoms.append(at)
        for at in mol["grad_data"]:
            soap_derivatives.append(at)
        for at in mol["grad_index_0based"]:    
            neigh_id.append(at)        
    return soap_atoms, soap_derivatives, neigh_id

def num_gradient(mol,h=0.001,direction=0,iatom=0):
    tmpmol = mol.copy()
    pos = tmpmol.get_positions()
    pos[iatom][direction] += h
    tmpmol.set_positions(pos)
    soap_plus = soap.calc(tmpmol)["data"]
    pos[iatom][direction] += -2.0*h
    tmpmol.set_positions(pos)
    soap_minus = soap.calc(tmpmol)["data"]
    pos[iatom][direction] += h
    tmpmol.set_positions(pos)
    return (soap_plus-soap_minus)/(2.0*h)

def reorg(ref_mol):
    desc1,der1, id = _descr_deriv_atoms([ref_mol])
    id = [list(item) for item in id]
    derAll = []
    derOne = [[],[],[]]
    derTwo = []
    lenDer = len(der1[0][0])
    count = 0
    lib_der = {}
    for a in range(len(der1)):
        lib_der[str(id[a])] = der1[a] 
    for a1 in range(len(ref_mol)):
        for a2 in range(len(ref_mol)):
            if [a1,a2] in id:
                derOne[0].append(lib_der[str([a1,a2])][0])
                derOne[1].append(lib_der[str([a1,a2])][1])
                derOne[2].append(lib_der[str([a1,a2])][2])
            else:
                for di in range(3):
                    derOne[di].append(np.zeros(lenDer))
        derAll.append(derOne)
        derOne = [[],[],[]]
    return derAll

all_molecules = read("/datavon1/DatabasesMolecules/LiH/LiH.xyz@:",format="extxyz")
main = all_molecules[100]

#print(len(main))

#Descriptor_description = f"soap cutoff=3 zeta=2 delta = 1.0 atom_sigma = 1.0 l_max=2 n_max=3 add_species=F"

soap_main = soap.calc(main, grad=True)
soap_num = num_gradient(main,direction=0,iatom=3)
soap_reorg = reorg(main)
for id, idx in enumerate(soap_main['grad_index_0based']):
    print(id,idx)


print(soap_main['grad_data'][15][0])
print(num_gradient(main,direction=0,iatom=3)[2])
print(soap_reorg[2][0][3])
'''


print(len(soap_num))
print(len(soap_num[0]))
print(len(soap_reorg))
print(len(soap_reorg[0]))
print(len(soap_reorg[0][0]))
print(len(soap_reorg[0][0][0]))

#print(soap_num)
#print(soap_reorg)


print("prdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
print(soap_reorg)
print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeek')

print(list(soap_main))
print(soap_main['grad_data'])
print(soap_main['grad_index_0based'])

print((soap_main['grad_data'].shape))
print(num_gradient(main,direction=0,iatom=3)[3])
print("######################################################################")
print(soap_main['grad_data'][16][0])
print(soap_main['grad_index_0based'])
'''