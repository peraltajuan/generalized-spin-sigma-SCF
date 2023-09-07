import numpy as np
import sys
from scipy.linalg import sqrtm
from numpy.linalg import inv
from scipy.linalg import eigh
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
from timeit import default_timer as timer
from functions_sigma import *

np.set_printoptions(threshold=sys.maxsize)
#*************************************************************************

def main():
 
  fname = sys.argv[1]

  Nbas = np.load('Nbas.npy') 
  sys.stdout = open( 'Pop.'+fname+'.out'  , 'w')

  print("Python version: ",sys.version )
  print("NumPy  version: ",np.__version__)
  width = 73
  prt_sep(width)
  print( "*"*28 +  " S I G M A G H F " + "*"*28 )  
  prt_sep(width)
  print(f"Nbas = {Nbas:g}")
  NE = np.load('NE.npy')
  print(f"NE   = {NE:g}")
  print("Using files with name: "+fname)

  g =np.load('g.npy')
  S = np.load('S.npy')
  h = np.load('h.npy')


  Cp= np.zeros((2*Nbas,2*Nbas))
  C= np.zeros((2*Nbas,2*Nbas))
  A = S
  S12 = sqrtm(A.real)
  Sm12 = inv(S12)
  
  for i in range(Nbas):
      for j in range(Nbas):
        Cp[i,j] = Sm12[i,j]
  for i in range(Nbas,2*Nbas):
      for j in range(Nbas,2*Nbas):
        Cp[i,j] = Sm12[i-Nbas,j-Nbas]
  C=Cp.T.conj()

  NR_energy = 0.0

  bta = np.load('bta.npy')
# atom to basis map
  atb = np.array((at2bas(bta)),dtype=object )

  Sinv = inv(S).astype(complex)


# Arrange 2EI in a GHF format  
  G = sq2ei_G(g,Nbas)

# This is the initial density matrix to be used. It is the occupation number matrix but it can be changed.
# The transformation to an orthonormal basis is done using the MO coefficients in C.
  P_MO = np.zeros((2*Nbas,2*Nbas),dtype=complex)
#  for i in range(NE): 
#        P_MO[i,i] = 1.00
  P_MO[0,0] =1.
  P_MO[1,1] =1.
  P_MO[3,3] =1.

  start = timer()
 
  # Notes:
  #  MaxSCF = 0    :do only simulated annealing
  #  Iter_SA = 0   :do only SCF
  #  MaxSCF and Iter_SA non-zero to do a sequence of simulated annealing plus SCF
  #  Only the omegas that complete successfully the prescribed sequence should get printed
  #  At the end of each omega, a population analysis is printed in a .out file
  #  If the SCF fails, the SCF energy and density are taken from 
  #  the simulated annealing step. This is handled with the Exception.


  MaxSCF = 1000             # Maximum number of SCF iteration
  Iter_SA = 1000            # Number of Simulated Annealing steps
  method = dispersor        # fock or dispersor
  just_var = False          # just_var=True works with the variance instead of the full dispersor

  # The array 'omegas' contains all the omega values to scan
  omegas = np.linspace(-3.00,-1.50, 300 , endpoint=True)

  if just_var  and method.__name__ == 'dispersor' : str_target = 'variance'    # these lines are for printing only
  if not just_var and method.__name__ == 'dispersor': str_target = 'dispersor'
  if method.__name__ == 'fock' : str_target = 'fock'


  data_to_save =[]  


  file_dat = open( 'sigma.'+fname+'.dat'  , 'w' )


  #  Start main loop
 
  for io, omega in enumerate(omegas):

     prt_sep(width) 
     if Iter_SA > 0:
         print(f'Starting {Iter_SA:g} iterations simulated annealing with {str_target:s} and omega = {omega:7.4f}')


         P_SA, target_SA = sim(h,G,C,P_MO,S,Sinv,omega,NE,Iter_SA,method,just_var=just_var)
         En_SA  = fock(P_SA,h,G,0,0,just_en=True)


         print(f'Energy SA       = {En_SA+NR_energy:18.11f}')
         print(f'Target SA       = {target_SA:18.11f}')
         success = False

     else:
         target_SA = dispersion(P_SA,h,G,Sinv,omega,just_en=True,just_var=just_var)
         En_SA  = fock(P_SA,h,G,0,0,just_en=True) 
         print('--- SA    Skipped ---')
         success = False
     # Attempt and SCF after the SA. If successful, replace P and the target quantity.   

     if False:
         print ('Check Gradient: ' , check_grad(dispersion, dispersor, complex_to_real( P_SA.ravel() )  ,h,G,Sinv,omega   ) )
         fprime =  approx_fprime(complex_to_real( P_SA.ravel() )  ,dispersion ,1.e-5, h,G,Sinv,omega     )
         print('Effective Fock from finite differences')
         FPrime =  real_to_complex(fprime).reshape(2*Nbas,2*Nbas)
         printM(FPrime)

         prt_sep(30)
         print('Effective Fock from equation')
         FEff, disp = dispersor(P_SA,h,G,Sinv,omega)
         printM( FEff )
         quit()


     try:
         print(f'Starting {MaxSCF:3g} SCF iterations with {str_target:s} and omega = {omega:7.4f}')
         P_scf,  target_scf  = simple_scf(h,G,S,P_SA,Sinv,omega,NE,NR_energy,MaxSCF,method,just_var=just_var)
         En_scf  = fock(P_scf,h,G,0,0,just_en=True)

  
         print(f'SCF Energy      = {En_scf+NR_energy:18.11f}')
         print(f'Target SCF      = {target_scf:18.11f}')
         success = True
     except Exception:
         if MaxSCF != 0 : print('--- SCF Failed ---')
         if MaxSCF == 0 : print('--- SCF Skipped ---')
         En_scf = En_SA
         P_scf = P_SA
         target_scf = target_SA
         success = False
         pass

#    possibly keep the same density

     if Iter_SA == 0: P_SA = P_scf

     if success or Iter_SA > 0: 
           np.savetxt(file_dat , [omega,En_scf+NR_energy,target_scf], newline=' '  )
           file_dat.write('\n')
           file_dat.flush()
           # population. 
           prt_sep(width) 
           get_pop(P_scf, atb, S, fname)
           prt_sep(width) 

     



# 
  file_dat.close()
  end = timer()
  print(f'Time for {len(omegas):g} iterations: ', end-start )
  sys.stdout.close()

  
###########################################################################
if __name__=='__main__':
    main()


