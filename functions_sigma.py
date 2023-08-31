#!/usr/bin/env python3
#
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import scipy
import sys
import random
import cmath
#from einsumt import einsumt as einsum
from numpy import einsum
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)


# print pretty separation
def prt_sep(i):
  print("*"*i,flush=True)


# print pretty Matrix
def printM(a):
    for row in a:
        for col in row:
            print("{:>+,.5f}".format(col), end="  ")
        print("")


def reorder_C(A):
    """
    Reorders a GHF Coefficient from 
    1a 1b 2a 2b ...  to
    1a 2a 3a .... 1b 2b 3b
    only columns are reordered
    """

    L = len(A)
    N = int(L/2)    
    B = np.zeros((L,L),dtype=complex)
    for i in range(L):
      ii = 2*np.mod(i,N) + int(i/N)
      B[:,i] = A[:,ii]
    return B




def reorder(A):
    """
    Reorders a GHF matrix from 
    1a 1b 2a 2b ...  to
    1a 2a 3a .... 1b 2b 3b
    """

    L = len(A)
    N = int(L/2)  
    B = np.zeros((L,L),dtype=complex)
    for i in range(L):
      ii = 2*np.mod(i,N) + int(i/N)
      for j in range(L):
        jj = 2*np.mod(j,N) + int(j/N) 
        B[i,j] = A[ii,jj]
    return B




def unfold(A):
    ''' unfolds LT to square format REAL
    this is faster than using the .expand method
    '''
    L = len(A)
    N = int((np.sqrt(1+8*L) - 1)/2 )
    B = np.empty((N,N),dtype=float)
    ij = 0
    for i in range(N):
      for j in range(i+1):
        B[i,j] = A[ij]
        B[j,i] = B[i,j]
        ij = ij + 1
    return B


def unfoldRS(A):
    ''' unfolds LT to square format COMPLEX
    this is faster than using the .expand method
    '''
    L = len(A)
    N = int((np.sqrt(1+8*L) - 1)/2 )
    B = np.empty((N,N),dtype=complex)
    ij = 0
    for i in range(N):
      for j in range(i+1):
        B[i,j] = A[ij]
        B[j,i] = B[i,j].conjugate()
        ij = ij + 1
    return B

def at2bas(z):
    '''atom to basis map
    '''
    nat = int(max(z))
    atmap =  [  [] for i in range(nat) ]
    n = 0
    for i in z:
#      atmap[i-1] = atmap[i-1] + 1
      atmap[i-1] = np.append(atmap[i-1], int(n) )
      n = n  + 1
    for ii, i in enumerate(atmap):
       atmap[ii] = atmap[ii].astype(int)
    return atmap

def real_to_complex(z):     
    return (z[:len(z)//2] + 1j * z[len(z)//2:]).ravel()

def complex_to_real(z):    
    return np.concatenate((np.real(z), np.imag(z))).ravel()



def lind(i,j):
    '''
    Used to convert lower triangular to square storage
    '''
    return  int( max(i+1,j+1)*(max(i+1,j+1)-1)/2 + min(i+1,j+1)-1 ) 




def int_spn(i,j,Nbas):
    '''
    Silly spin integration for the reordering functions
    '''
    i_spn = mod(i,Nbas)
    j_spn = mod(j,Nbas)
    return np.kron(i_spn,j_spn)




def sq2ei(g,Nbas):
    '''
    This funcion creates the FULL 2EI in AOs (space part) 
    '''
    G = np.zeros((Nbas,Nbas,Nbas,Nbas),dtype=float)
    for i in range(Nbas):
      for j in range(Nbas):
        ij =    lind(i,j)
        for k in range(Nbas):
          for l in range(Nbas):
             kl =  lind(k,l)
             ijkl = lind(ij,kl)
             G[i,j,k,l] = g[ijkl]
    return G 


def sq2ei_G(g,Nbas):
    '''
    This funcion creates the FULL 2EI in spin-AOs (space x spin). The order is in the block form 
    aa  ab
    ba  bb
    for two indices.
    Note: this is memory hungry as it uses (2*Nbas)^4 words ~ 12 Gb for Nbas = 100
    '''
    G = np.zeros((2*Nbas,2*Nbas,2*Nbas,2*Nbas),dtype=float)
    for i in range(Nbas):
      for j in range(Nbas):
        ij =    lind(i,j)
        for k in range(Nbas):
          for l in range(Nbas):
             kl =  lind(k,l)
             ijkl = lind(ij,kl)
             G[i,j,k,l] = g[ijkl]
             G[i+Nbas,j+Nbas,k,l] = g[ijkl]
             G[i,j,k+Nbas,l+Nbas] = g[ijkl]
             G[i+Nbas,j+Nbas,k+Nbas,l+Nbas] = g[ijkl]
    return G 



def fock(P,h,g,X,z,**kwargs):
    '''
    Evaluates the HF energy and possibly the Fock matrix
       P: density matrix in AO basis
       h: core hamiltonian
       g: two-electron integrals
       X: not used (but kept for compatibility with the dispersor function
       z: same as X
       kwargs optional keywords:
       just_en=True or False for just evaluating the energy. Default: False
       just_var=True or False for just evaluating the variance (not used here). Default: False
    '''
    kwargs_default = {'just_en': False, 'just_var':False}
    kwargs_default.update(kwargs)
    just_en = kwargs_default['just_en']
    just_var= kwargs_default['just_var']
    J = einsum("ijkl, lk -> ij", g, P,optimize=True)
    K = einsum("ilkj, lk -> ij", g, P,optimize=True)
    if not just_en :
          return h+J-K,  np.trace(( h +  .5*J - .5*K) @ P).real
    else: 
          return  np.trace(( h +  .5*J - .5*K) @ P).real





def dispersion(Pi,h,G,Sinv,omega,**kwargs):
       '''
       Evaluates the dispersor or the variance
       P: density matrix in AO basis
       h: core hamiltonian
       G: two-electron integrals
       Sinv: inverse of the overlap matrix
       omega: omega value 
       kwargs optional keywords:
       just_en=True or False for just evaluating the target quantity. Default: False
       just_var=True or False for just evaluating the variance (instead of the full dispersor). 
                Default: False
       '''
       kwargs_default = {'just_en': False, 'just_var':False}
       kwargs_default.update(kwargs)
       just_en = kwargs_default['just_en']
       just_var= kwargs_default['just_var']
       factor = 1.0                # 1.0 to add the last term in the dispersor
       if just_var: factor = 0.0   # 0.0 to just calculate the variance
       if(len(Pi) == len(h)): 
          P = Pi
       else:
          P = real_to_complex(Pi).reshape(len(h),len(h))
       F,E = fock(P,h,G,0,0)
       Q = (Sinv - P)
       v1 = np.trace(F @ P @ F @ Q)
       v2 =     -.5*einsum("pqrs, ijkl, pi, lq, rk, js ->", G, G, P, Q, P, Q, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (1, 2), (0, 1)]) + \
                +.5*einsum("pqrs, ijkl, pi, jq, rk, ls ->", G, G, P, Q, P, Q, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (1, 2), (0, 1)])
       return (v1 + v2 + factor*(E-omega)**2 ).real


def d_dispersion(Pi,h,G,Sinv,omega,**kwargs):
       '''
       Evaluates the dispersor or the variance
       P: density matrix in AO basis
       h: core hamiltonian
       G: two-electron integrals
       Sinv: inverse of the overlap matrix
       omega: omega value 
       kwargs optional keywords:
       just_en=True or False for just evaluating the target quantity. Default: False
       just_var=True or False for just evaluating the variance (instead of the full dispersor). 
                Default: False
       '''
       kwargs_default = {'just_en': False, 'just_var':False}
       kwargs_default.update(kwargs)
       just_en = kwargs_default['just_en']
       just_var= kwargs_default['just_var']
       factor = 1.0                # 1.0 to add the last term in the dispersor
       if just_var: factor = 0.0   # 0.0 to just calculate the variance
       if(len(Pi) == len(h)): 
          P = Pi
       else:
          P = real_to_complex(Pi).reshape(len(h),len(h))
       F,E = fock(P,h,G,0,0)
       Q = (Sinv - P)
       v1 = np.trace(F @ P @ F @ Q)
       v2 =     -.5*einsum("pqrs, ijkl, pi, lq, rk, js ->", G, G, P, Q, P, Q, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (1, 2), (0, 1)]) + \
                +.5*einsum("pqrs, ijkl, pi, jq, rk, ls ->", G, G, P, Q, P, Q, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (1, 2), (0, 1)])
       # do not use this with just_var = True
       return (v1 + v2 + factor*(E-omega)**2 ).real , 2.*omega -2.*E



def dispersion_grad(omega,Pi,h,G,S,Sinv,NE,NR_energy,What):
       just_en = False
       just_var= False
       MaxSCF = 3000
       method = dispersor
       P_scf,  target_scf  = simple_scf(h,G,S,Pi,Sinv,omega,NE,NR_energy,MaxSCF,method,just_var=just_var)
       Pi = P_scf
       f,g =  d_dispersion(Pi,h,G,Sinv,omega)
       if What == 0: return f
       if What == 1: return g
       if What == 2: return f,g
       return f,g



def dispersor(Pi,h,G,Sinv,omega,**kwargs):
       '''
       Evaluates the dispersor or the variance  and possibly the effective Fock matrix
       P: density matrix in AO basis
       h: core hamiltonian
       G: two-electron integrals
       Sinv: inverse of the overlap matrix
       omega: omega value 
       kwargs optional keywords:
       just_en=True or False for just evaluating the target quantity. Default: False
       just_var=True or False for just evaluating the variance (instead of the full dispersor). 
              Default: False
       '''
       kwargs_default = {'just_en': False, 'just_var':False}
       kwargs_default.update(kwargs)
       just_en = kwargs_default['just_en']
       just_var= kwargs_default['just_var']
       factor = 1.0                # 1.0 to add the last term in the dispersor
       if just_var: factor = 0.0   # 0.0 to just calculate the variance
       if(len(Pi) == len(h)): 
           P = Pi
       else:
           P = real_to_complex(Pi).reshape(len(h),len(h)) 
        
       if not just_en:
           F, E = fock(P,h,G,Sinv,omega,just_en=False)
           Q = (Sinv - P)

           F_eff = (F @ (Q - P) @ F)    +                                                            \
                   (                                                                              \
                   +einsum("pqrs, qi, ij, jp -> sr",  G, Q, F, P, optimize=True)                \
                   -einsum("prqs, qi, ij, jp -> sr",  G, Q, F, P, optimize=True)                \
                   +einsum("pqrs, qi, ij, jp -> sr",  G, P, F, Q, optimize=True)                 \
                   -einsum("prqs, qi, ij, jp -> sr",  G, P, F, Q, optimize=True)                \
                   +einsum("rk,ls,ji,rspi,lkjq -> qp",P, Q, Q, G, G , optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])     \
                   -einsum("rk,ls,ji,rspi,jklq -> qp",P, Q, Q, G, G , optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])     \
                   -einsum("rk,ls,ij,rsiq,lkpj -> qp",P, Q, P, G, G , optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])     \
                   +einsum("rk,ls,ij,rsiq,pklj -> qp",P, Q, P, G, G , optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])     \
                   )                                                                               \
                   + (2.*(E - omega) * F )*factor
           if (len(Pi) != len(h)):
                return  complex_to_real( F_eff.ravel() )  
           else:
                return F_eff,  dispersion(P,h,G,Sinv,omega,just_en=just_en,just_var=just_var) 
       else: 
           return dispersion(P,h,G,Sinv,omega,just_en=just_en,just_var=just_var) 
          






def rotate(P_MO,NE,amax):
    '''
    Picks one random 'o' and one random but different 'v' and performs 
    a complex unitary rotation of P_MO. 'o' and 'v' are ANY two orbitals.
    NOTE: Check that this rotation is OK.
    '''
    Nsize = len(P_MO)
    o = random.randint(0,Nsize-1)
    lv = (list(range(Nsize))) 
    lv.pop(o)
    v = random.choice(lv)
    U = np.eye(Nsize,dtype=complex)
    a =  amax*random.uniform(-1.,1.)
    a1 = amax*random.uniform(-1.,1.)
    a2 = amax*random.uniform(-1.,1.)
    t  = amax*random.uniform(-1.,1.)
    expa = cmath.exp(1j*a/2)
    U[o,v] = np.sin(t)*cmath.exp( 1j*a2)*expa
    U[o,o] = np.cos(t)*cmath.exp( 1j*a1)*expa
    U[v,o] =-np.sin(t)*cmath.exp(-1j*a2)*expa
    U[v,v] = np.cos(t)*cmath.exp(-1j*a1)*expa
    
    return  U.T.conj() @ P_MO @ U 

def urotate(P_MO,NE,amax):
    '''
    Picks one random 'o' and one random but different 'v' and performs
    a complex unitary rotation of P_MO. 'o' and 'v' are ANY two orbitals.
    NOTE: Check that this rotation is OK.
    '''
    Nsize = len(P_MO)/2
    o = random.randint(0,int(Nsize)-1)
    lv = (list(range(int(Nsize))))
    lv.pop(o)
    v = random.choice(lv)
    U = np.eye(2*int(Nsize),dtype=complex)
    a =  amax*random.uniform(-1.,1.)
    a1 = amax*random.uniform(-1.,1.)
    a2 = amax*random.uniform(-1.,1.)
    t  = amax*random.uniform(-1.,1.)
    expa = cmath.exp(1j*a/2)
    U[o,v] = np.sin(t)*cmath.exp( 1j*a2)*expa
    U[o,o] = np.cos(t)*cmath.exp( 1j*a1)*expa
    U[v,o] =-np.sin(t)*cmath.exp(-1j*a2)*expa
    U[v,v] = np.cos(t)*cmath.exp(-1j*a1)*expa

    o = random.randint(0,int(Nsize)-1)
    lv = (list(range(int(Nsize))))
    lv.pop(o)
    v = random.choice(lv)
    a =  amax*random.uniform(-1.,1.)
    a1 = amax*random.uniform(-1.,1.)
    a2 = amax*random.uniform(-1.,1.)
    t  = amax*random.uniform(-1.,1.)
    expa = cmath.exp(1j*a/2)
    U[o+int(Nsize),v+int(Nsize)] = np.sin(t)*cmath.exp( 1j*a2)*expa
    U[o+int(Nsize),o+int(Nsize)] = np.cos(t)*cmath.exp( 1j*a1)*expa
    U[v+int(Nsize),o+int(Nsize)] =-np.sin(t)*cmath.exp(-1j*a2)*expa
    U[v+int(Nsize),v+int(Nsize)] = np.cos(t)*cmath.exp(-1j*a1)*expa


    return  U.T.conj() @ P_MO @ U

def rorotate(P_MO,NE,amax):
    '''
    Picks one random 'o' and one random but different 'v' and performs
    a complex unitary rotation of P_MO. 'o' and 'v' are ANY two orbitals.
    NOTE: Check that this rotation is OK.
    '''

    Nsize = len(P_MO)/2
    U = np.eye(2*int(Nsize),dtype=complex)

    o = random.randint(0,int(Nsize)-1)
    lv = (list(range(int(Nsize))))
    lv.pop(o)
    v = random.choice(lv)
    a =  amax*random.uniform(-1.,1.)
    a1 = amax*random.uniform(-1.,1.)
    a2 = amax*random.uniform(-1.,1.)
    t  = amax*random.uniform(-1.,1.)
    expa = cmath.exp(1j*a/2)
    U[o,v] = np.sin(t)*cmath.exp( 1j*a2)*expa
    U[o,o] = np.cos(t)*cmath.exp( 1j*a1)*expa
    U[v,o] =-np.sin(t)*cmath.exp(-1j*a2)*expa
    U[v,v] = np.cos(t)*cmath.exp(-1j*a1)*expa

    U[o+int(Nsize),v+int(Nsize)] = np.sin(t)*cmath.exp( 1j*a2)*expa
    U[o+int(Nsize),o+int(Nsize)] = np.cos(t)*cmath.exp( 1j*a1)*expa
    U[v+int(Nsize),o+int(Nsize)] =-np.sin(t)*cmath.exp(-1j*a2)*expa
    U[v+int(Nsize),v+int(Nsize)] = np.cos(t)*cmath.exp(-1j*a1)*expa

    return  U.T.conj() @ P_MO @ U

def accept(Enew,Eold,T):
    '''
    Decides if the new energy is accepted or not
    '''
    if Enew < Eold:
#       Acceps always if Enew < Eold
#       Ex = np.exp((Enew - Eold)/T)
#       return  random.uniform(0.,1.)*(1.+Ex) > Ex 
        return True
    else:
       try: 
           Ex = np.exp((Eold - Enew)/T)  
       except:                          
         #Avoid underflow error ???    
           Ex = 0.                        
       return  random.uniform(0.,1.)*(1.+Ex) < Ex

def sigma(h,G,C,P_MO,S,Sinv,omega,NE,Niter,method,**kwargs):
    '''
    Main loop 
    h: core hamiltonian
    G: two-electron integrals
    C: MO coefficients
    P_MO: density matrix in MO basis
    S: overlap matrix
    Sinv: inverse of the overlap matrix
    omega: omega value 
    NE: number of electrons (not needed)
    Niter: number of iterations
    method: dispersion if method=dispersor, or energy if method=fock
    kwargs optional keywords:
    just_en=True or False for just evaluating the target quantity. Default: False
    just_var=True or False for just evaluating the variance (instead of the full dispersor). 
         Default: False
    '''
    kwargs_default = {'just_en': False, 'just_var':False}
    kwargs_default.update(kwargs)
    just_en = kwargs_default['just_en']
    just_var= kwargs_default['just_var']
    # parametrs
    a0 = 1.4
    T0 = 0.001
#    Niter=20000
    CT = 0.9995
    Ca = 0.9995
    error_thres = 1.E-5
 
    P_MO_old = P_MO

    E_old = method(C.T.conj() @ P_MO @ C,h,G,Sinv,omega, just_en=True,just_var=just_var )
    a = a0
    T = T0
    for i in range(Niter):
          P_MO_new = rotate(P_MO_old,NE,a)
          P_new = C.T.conj() @ P_MO_new @ C
          E_new = method(P_new,h,G,Sinv,omega,just_en=True,just_var=just_var )

          if accept(E_new, E_old, T):
             P_MO_old = P_MO_new
             E_old = E_new
             a = a0*Ca**i
             T = T0*CT**i
    Pf = C.T.conj() @ P_MO_old @ C
    Ff, Ef  = dispersor(Pf,h,G,Sinv,omega,just_en=False,just_var=just_var)
    error =np.linalg.norm(  Ff @ Pf @ S - S @ Pf @ Ff )
    print(f'SA   Error = {error:8.4e}')
    return Pf , Ef


def usigma(h,G,C,P_MO,S,Sinv,omega,NE,Niter,method,**kwargs):
    '''
    Main loop 
    h: core hamiltonian
    G: two-electron integrals
    C: MO coefficients
    P_MO: density matrix in MO basis
    S: overlap matrix
    Sinv: inverse of the overlap matrix
    omega: omega value
    NE: number of electrons (not needed)
    Niter: number of iterations
    method: dispersion if method=dispersor, or energy if method=fock
    kwargs optional keywords:
    just_en=True or False for just evaluating the target quantity. Default: False
    just_var=True or False for just evaluating the variance (instead of the full dispersor).
         Default: False
    '''
    kwargs_default = {'just_en': False, 'just_var':False}
    kwargs_default.update(kwargs)
    just_en = kwargs_default['just_en']
    just_var= kwargs_default['just_var']
    # parametrs
    a0 = 1.4
    T0 = 0.01
#    Niter=20000
    CT = 0.9995
    Ca = 0.9995
    error_thres = 1.E-5

    P_MO_old = P_MO

    E_old = method(C.T.conj() @ P_MO @ C,h,G,Sinv,omega, just_en=True,just_var=just_var )
    a = a0
    T = T0
    for i in range(Niter):
          P_MO_new = urotate(P_MO_old,NE,a)
          P_new = C.T.conj() @ P_MO_new @ C
          E_new = method(P_new,h,G,Sinv,omega,just_en=True,just_var=just_var )

          if accept(E_new, E_old, T):
             P_MO_old = P_MO_new
             E_old = E_new
             a = a0*Ca**i
             T = T0*CT**i
    Pf = C.T.conj() @ P_MO_old @ C
    Ff, Ef  = dispersor(Pf,h,G,Sinv,omega,just_en=False,just_var=just_var)
    error =np.linalg.norm(  Ff @ Pf @ S - S @ Pf @ Ff )
    print(f'SA  Error = {error:8.4e}')
    return Pf , Ef


def rosigma(h,G,C,P_MO,S,Sinv,omega,NE,Niter,method,**kwargs):
    '''
    Main loop 
    h: core hamiltonian
    G: two-electron integrals
    C: MO coefficients
    P_MO: density matrix in MO basis
    S: overlap matrix
    Sinv: inverse of the overlap matrix
    omega: omega value
    NE: number of electrons (not needed)
    Niter: number of iterations
    method: dispersion if method=dispersor, or energy if method=fock
    kwargs optional keywords:
    just_en=True or False for just evaluating the target quantity. Default: False
    just_var=True or False for just evaluating the variance (instead of the full dispersor).
         Default: False
    '''
    kwargs_default = {'just_en': False, 'just_var':False}
    kwargs_default.update(kwargs)
    just_en = kwargs_default['just_en']
    just_var= kwargs_default['just_var']
    # parametrs
    a0 = 1.4
    T0 = 0.001
#    Niter=20000
    CT = 0.9995
    Ca = 0.9995
    error_thres = 1.E-5

    P_MO_old = P_MO

    E_old = method(C.T.conj() @ P_MO @ C,h,G,Sinv,omega, just_en=True,just_var=just_var )
    a = a0
    T = T0
    for i in range(Niter):
          P_MO_new = rorotate(P_MO_old,NE,a)
          P_new = C.T.conj() @ P_MO_new @ C
          E_new = method(P_new,h,G,Sinv,omega,just_en=True,just_var=just_var )

          if accept(E_new, E_old, T):
             P_MO_old = P_MO_new
             E_old = E_new
             a = a0*Ca**i
             T = T0*CT**i
    Pf = C.T.conj() @ P_MO_old @ C
    Ff, Ef  = dispersor(Pf,h,G,Sinv,omega,just_en=False,just_var=just_var)
    error =np.linalg.norm(  Ff @ Pf @ S - S @ Pf @ Ff )
    print(f'SA  Error = {error:8.4e}')
    return Pf , Ef


def stop_scf(E,e,i):
    print(f'SCF completed in {i:g} iterations') 
    print(f'DIIS error = {e:10.5e}') 






class subspace(list):
    '''
    This is a class extension of simple python lists
    It has only one function thay adds an element, and keeps
    it to a given length by removing the head. Used for DIIS
    '''
    def add(self, item):
        list.append(self, item)
        if len(self) > 10 : del self[0]





def extrapolate_F(fmats, coeff,N):
    '''
    Used for DIIS
    '''
    extrapolated_F = np.zeros((N,N),dtype=complex)
    for i in range(len(fmats)):
        extrapolated_F += fmats[i]*coeff[i]
    return extrapolated_F





def eval_diis_c(errors):
    '''
    Evaluates DIIS coefficients from the B matrix
    '''
    b = np.zeros((len(errors)+1,len(errors)+1),dtype=float)
    b[-1,:] = -1.
    b[:,-1] = -1. 
    b[-1,-1]= 0.
    r = np.zeros((len(errors)+1),dtype=float)
    r[-1] = -1.

    for i in range(len(errors)):
        for j in range(i+1):
            # use the Frobenius inner product
            error =  np.trace( errors[i].T.conj()  @  errors[j] ).real
            b[i,j] = error
            b[j,i] = error
    *coeff, _ = np.linalg.solve(b,r)
#    print(len(errors),coeff) 
    return np.array(coeff).flatten()


def find_max(e):
    coef = eval_diis_c(e)
    return max(np.abs(coef))



# not used 
def reduce_diis(e, F):
    '''
    Reduces the list of Fock and errors if a coefficnet is too large
    '''
    success = False
    while not success:
       c = find_max(e)
       if c > 5.1:
         del e[0]
         del F[0]
       else:
         success = True  
    


def simple_scf(h,G,S,P,Sinv,omega,NE,NR,MaxSCF,method,**kwargs):
    '''
    Performs a simple SCF calculation
    h: core hamiltonian
    G: two-electron integrals
    S: overlap matrix
    P: density matrix in AO basis
    Sinv: inverse of the overlap matrix
    omega: omega value 
    NE: number of electrons 
    MaxSCF: maximum number of iterations
    method: dispersion if method=dispersor, or energy if method=fock
    kwargs optional keywords:
    just_en=True or False for just evaluating the target quantity. Default: False
    just_var=True or False for just evaluating the variance (instead of the full dispersor). 
         Default: False
    'method' is 'fock' or 'dispersor'
    '''
    kwargs_default = {'just_en': False, 'just_var':False}
    kwargs_default.update(kwargs)
    just_en = kwargs_default['just_en']
    just_var= kwargs_default['just_var']
    Nbas = len(S)
#    MaxSCF = 300
    SCF_thresh = 1.e-13
    occ = np.zeros((Nbas,Nbas),dtype=complex)
    for i in range(NE): 
       occ[i,i] = 1.0
    Pnew = P

    F_mats = subspace()
    e_mats = subspace()


    for i in range(MaxSCF):

       #      Evaluate Fock and Energy
       #      If methods = dispersor, E is the dispersor

       F, E = method(Pnew,h,G,Sinv,omega,just_en=False,just_var=just_var)

       #      Add matrices to the subspace lists

       F_mats.add(F)

       #      Calculate DIIS commutator error

       diis_error = F @ Pnew @ S - S @ Pnew @ F

       #      Add error to the subspace
 
       e_mats.add(diis_error)
       norm_error =np.linalg.norm(diis_error)
        
       #      check for convergence

       if norm_error < SCF_thresh:
            stop_scf(E+NR,norm_error,i)
            return Pnew, E
            break

       #      First iteration only: diagonalize and form new density matrix
       

       if len(F_mats) < 8:
            if len(F_mats) == 1:  
                _, C = scipy.linalg.eigh(F, b=S)
            else:  
                Fd = 0.8*F_mats[ -2 ] + 0.2*F_mats[ -1 ]
                _, C = scipy.linalg.eigh(Fd, b=S)
            Pnew = C @ occ @ C.T.conj()

       #      If there are enough error vectors: DIIS-extrapolate F and calculate new density matrix

       else:
#            reduce_diis(e_mats, F_mats)
            diis_coef = eval_diis_c(e_mats)
            extrap_F  = extrapolate_F(F_mats,diis_coef,Nbas)
            _, C = scipy.linalg.eigh(extrap_F, b=S)
            Pnew = C @ occ @ C.T.conj()
            
    raise Exception('Max SCF iteretions reached - No Luck')



# population and local spin analysis



def pop(label,atmap,Pt,Px,Py,Pz):
    """ 
    This routine performs a standard population analysis and prints it.
    label: A label for printing
    atmap: atom to basis map. This is a list where
         atb[0] is list of basis for atom 0, atb[0] = [0,1,2,3,4] for example.
         atb[1] is the list of basis for atom 1
         etc.
    Pt:  total charge density matrix
    Px, Py, Pz: spin density matrices
    To do: write data using  fname to be used in plots
    """
    np.seterr(all='raise')
    print(label)
    print(f'             Charge        Sx       Sy       Sz     ')

    N = len(atmap)
    vect = []
    for  iat, lstat in enumerate(atmap):
      Q, Sx,Sy,Sz = 0.0, 0.0, 0.0, 0.0 
      for bas in lstat:
        Q  += Pt[bas,bas]  
        Sx += Px[bas,bas] 
        Sy += Py[bas,bas] 
        Sz += Pz[bas,bas] 
      Q  =  Q.real
      Sx = Sx.real
      Sy = Sy.real
      Sz = Sz.real
      vect.append( [Sx,Sy,Sz] )
#     Here we should verify that S and Q end up being real numbers as a consistency check
      print(f'Atom {iat:2g}    {Q:8.5f}    {Sx:8.5f} {Sy:8.5f} {Sz:8.5f}')
#    prt_sep(73)
    
    cols = rows = np.arange(1,N+1)
    angle_mat = np.zeros((N,N),dtype=float)

    for i in range(N):
      for j in range(N):
        try: 
          v = np.dot( vect[i], vect[j] )/np.linalg.norm(vect[i])/np.linalg.norm(vect[j])
          angle_mat[i,j] =  np.arccos(v  )*180/np.pi
        except:
          angle_mat[i,j] = 999

    df_ang = pd.DataFrame( angle_mat , index=rows, columns=cols)
    df_ang = df_ang.replace({999:"--"})
    pd.options.display.float_format = '{:8.2f}'.format
    print("Local Spin Angles (deg)" )
    print(df_ang, flush=True)









def s2pop(atb,Pt,Px,Py,Pz):
  """ 
  This routine performs the S**2 local spin population based on
  J. Chem. Theory Comput. 2017, 13, 6101−6107 (Equation 11)
  DOI: 10.1021/acs.jctc.7b01022 and prints it.
  atb: atom to basis map. This is a list where
       atb[0] is list of basis for atom 0, atb[0] = [0,1,2,3,4] for example.
       atb[1] is the list of basis for atom 1
       etc.
  Pt:  total charge density matrix
  Px, Py, Pz: spin density matrices
  To do: write data using  fname to be used in plots
  """

  N = len(atb)
  S2mat = np.zeros((N,N),dtype=float)
  Stot = 0.0 

  for at1, bas1 in enumerate(atb):
   for at2, bas2 in enumerate(atb):
    S2 = 0.0 
    for A in bas1:
        if at1 == at2:   S2 += 0.75*Pt[A,A]
        for B in bas2:
            S2 += 0.75*(-0.5*Pt[A,B]*Pt[B,A]) +  \
                  0.25*(Px[A,A]*Px[B,B] + Py[A,A]*Py[B,B] + Pz[A,A]*Pz[B,B] + \
                  0.50*(Px[A,B]*Px[B,A] + Py[A,B]*Py[B,A] + Pz[A,B]*Pz[B,A]  ) ) 
    S2mat[at1,at2] = S2.real
    Stot += S2
  # Here we should verify that Stot ends up being real numbers as a consistency check
  Stot = Stot.real
  cols = rows = np.arange(1,N+1)
  df_S2 = pd.DataFrame(S2mat, index=rows, columns=cols)
  pd.options.display.float_format = '{:8.5f}'.format
  print("<S**2> Local Spin Population" )
  print(df_S2)
  print(f'total <S**2>:       {Stot:8.5f}',flush=True)
#  prt_sep(9*N+2)








def get_pop(P, atb, S, fname):
  '''
  Main driver for the population analysis
  To do: write data using  fname to be used in plots
  '''

  Nbas = len(S)
  dim = int(Nbas/2)
  Ss =S[dim:,dim:].real
  S12 = sqrtm(Ss)
  Sm12 = inv(S12)


  Paa = P[:dim,:dim]
  Pbb = P[dim:2*dim,dim:2*dim]
  Pab = P[dim:2*dim,:dim]
  Pba = P[:dim,dim:2*dim]

  Pt = (Paa + Pbb)
  Px = (Pab + Pba)
  Py = 1j*(Pab - Pba)
  Pz = (Paa - Pbb)


  # Lowdin 

  P_t = S12 @ Pt @ S12 
  P_x = S12 @ Px @ S12
  P_y = S12 @ Py @ S12
  P_z = S12 @ Pz @ S12

  pop('Lowdin Population', atb,P_t,P_x,P_y,P_z)
#  pop_ang('Angles', atb,P_t,P_x,P_y,P_z)

  # Local Spin

  s2pop(atb,P_t,P_x,P_y,P_z)



