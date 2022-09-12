from quippy.descriptors import Descriptor
from ase.io import read
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from funcsDer import *

Descriptor_description = f"soap cutoff=4 zeta=2 delta = 1.0 atom_sigma =1.0 l_max=2 n_max=3 n_Z=2 Z={{1 3}} n_species=2 species_Z={{1 3}}"
soap = Descriptor(Descriptor_description)


all_molecules = read("/datavon1/DatabasesMolecules/LiH/LiH.xyz@:",format="extxyz")
train_mol = all_molecules[:20]
train_desc = descr_atoms(train_mol)
print(len(train_desc))
main = all_molecules[100]

#print(len(main))

#Descriptor_description = f"soap cutoff=3 zeta=2 delta = 1.0 atom_sigma = 1.0 l_max=2 n_max=3 add_species=F"

soap_main = soap.calc(main, grad=True)
soap_num = num_gradient_soap(main,direction=0,iatom=3)
soap_reorg = reorg(main)
for id, idx in enumerate(soap_main['grad_index_0based']):
    print(id,idx)


#print(soap_main['grad_data'][15][0])
#print(num_gradient_soap(main,direction=0,iatom=3)[2])
#print(soap_reorg[2][0][3])


print((num_kernel_gradient(main,train_desc)[0][0][0]))
print((an_kernel_gradient(main,train_desc)[0][0][0]))
'''
print(len(num_kernel_gradient(main,train_desc)))

print(len(num_kernel_gradient(main,train_desc)[0]))

print(len(num_kernel_gradient(main,train_desc)[0][0]))

print(len(num_kernel_gradient(main,train_desc)[0][0][0]))
'''
'''
print(len(an_kernel_gradient(main,train_desc)))

print(len(an_kernel_gradient(main,train_desc)[0]))

print(len(an_kernel_gradient(main,train_desc)[0][0]))

print(len(an_kernel_gradient(main,train_desc)[0][0][0]))
'''

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