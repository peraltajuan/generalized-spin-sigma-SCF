# Generalized spin $\sigma$-SCF
This Python 3 code performs generalized spin $\sigma$-SCF calculations with 2-component (complex) spinors. 
The code can minimize the Hamiltonian dispersion using simulated annealing and/or self-consistent field iterations.
Please see these publications for details:
+ O. B. Oña, G. E. Massaccesi, J. I. Melo, A. Torre, L. Lain, D. R. Alcoba, and
  J. E. Peralta, Generalized Spin $\sigma$-SCF Method (this work; to be submitted).
+ H.-Z. Ye, M. Welborn, N. D. Ricke, and T. Van Voorhis, $\sigma$-SCF: A direct energy-targeting
method to mean-field excited states, J. Chem. Phys. 147, 214104 (2017)
+ G. David, T. J. P. Irons, A. E. A. Fouda, J. W. Furness, and A. M. Teale, Self-consistent
field methods for excited states in strong magnetic fields: a comparison between energy-
and variance-based approaches, J. Chem. Theory Comput. 17, 5492–5508 (2021).


### Instructions  
1) Install [PySCF](https://pyscf.org). Thsi is needed to obtain one- and two-electron integrals.

2) Use
```
python integrals.py
```
to generate the integrals and store them. You can edit the `integrals.py` file as needed to change the system at hand.

ℹ️ A version of this code that utilizes [Gaussian 16](https://gaussian.com/gaussian16/) matrix files can be provided upon request.<br>

3) Run in the command line
```
python sigma.py filename > sigma.out
```
After a successful termination, the output files `Pop.filename.out`  and `Sigma.filename.dat` should be created with the population analysis and the summary of the results, respectively. Edit the file `sigma.py` as needed for the purpose of the calculation. More information can be found as comments in the different files.

⚠️ The current version of the code is not optimized for speed.<br>
