import tempfile
import numpy as np
import scipy.linalg
from pyscf import fci
from scipy.special import binom  #Binom formula
from pyscf import gto, ao2mo, scf, mcscf
from pyscf import lib

mol = gto.Mole()
mol.atom=[
       ['H',( 1.09061 ,  0.0000, 0.)],
       ['H',(-0.545307,  0.9445, 0.)],
       ['H',(-0.545307, -0.9445, 0.)],
        ]
mol.basis='sto-3g'
mol.spin=1
mol.unit = 'B'
mol.verbose=0
mol.build()

mol.symmetry = False

hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
nao = hcore_ao.shape[0]
np.save('Nbas.npy',nao)
np.save('NE.npy',mol.nelectron)

h=np.block([[hcore_ao, np.zeros((nao,nao))],[np.zeros((nao,nao)),hcore_ao]])
np.save('h.npy',h)

Sovlp = mol.intor('int1e_ovlp')
S=np.block([[Sovlp, np.zeros((nao,nao))],[np.zeros((nao,nao)),Sovlp]])
np.save('S.npy',S)

g = mol.intor('int2e_sph', aosym=8)
np.save('g.npy',g)

bta=np.array([mol.bas_atom(0)+1])
for idx in range(1,nao):
    bta=np.append(bta,mol.bas_atom(int(idx))+1)
    
np.save('bta.npy',bta)

NR_energy=gto.mole.energy_nuc(mol, charges=None, coords=None)

np.save('NR_energy.npy',NR_energy)
print('NR_energy: ',NR_energy)






