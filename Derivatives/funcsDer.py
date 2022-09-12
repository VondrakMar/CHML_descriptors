
from quippy.descriptors import Descriptor
from ase.io import read
import numpy as np
import sys


Descriptor_description = f"soap cutoff=4 zeta=2 delta = 1.0 atom_sigma =1.0 l_max=2 n_max=3 n_Z=2 Z={{1 3}} n_species=2 species_Z={{1 3}}"
soap = Descriptor(Descriptor_description)

def descr_deriv_atoms(molecules):
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

def descr_atoms(molecules):
    soaps = soap.calc(molecules)
    soap_atoms=[]
    for mol in soaps:
        for at in mol["data"]:
            soap_atoms.append(at)        
    return soap_atoms

def num_gradient_soap(mol,h=0.001,direction=0,iatom=0):
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

def num_gradient_kernel(mol,desc_train,h=0.001,direction=0,iatom=0):
    tmpmol = mol.copy()
    pos = tmpmol.get_positions()
    #pos /= Bohr
    pos[iatom][direction] += h
    tmpmol.set_positions(pos)
    Kplus = soap_kernel_descriptors(tmpmol,desc_train=desc_train)
    pos[iatom][direction] += -2.0*h
    tmpmol.set_positions(pos)
    Kminus = soap_kernel_descriptors(tmpmol,desc_train=desc_train)
    pos[iatom][direction] += h
    tmpmol.set_positions(pos)

    return (Kplus-Kminus)/(2.0*h)


def soap_kernel_descriptors(ref_mol,desc_train, zeta=2):
    desc1 = soap.calc(ref_mol)["data"]
    K = np.matmul(desc1,np.transpose(desc_train))**zeta
    return K

def num_kernel_gradient(ref_mol,desc_train,h=0.001):
    dKdr = []
    for i in range(len(ref_mol)):
        dKdri = []
        for direction in range(3):
            dKdri.append(num_gradient_kernel(ref_mol,desc_train=desc_train,h=h,direction=direction,iatom=i))
        dKdr.append(dKdri)
    return dKdr

def reorg(ref_mol):
    desc1,der1, id = descr_deriv_atoms([ref_mol])
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



def an_kernel_gradient(ref_mol,desc_train,zeta=2):
    desc1,der1, id = descr_deriv_atoms([ref_mol])
    id = [list(item) for item in id]
    derAll = []
    derOne = [[],[],[]]
    lenDer = len(der1[0][0])
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
    dKdr = []

    for iatom in range(len(ref_mol)):
        dKdri = []
        for direction in range(3):
            curDer = (derAll[iatom][direction])
            t1 = (np.matmul(desc1,np.transpose(desc_train))**(zeta-1))
            t2 = np.matmul(curDer,np.transpose(desc_train))
            K = zeta*np.multiply(t1,t2)
            dKdri.append(K)
        dKdr.append(dKdri)
    return dKdr