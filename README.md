# generalized spin $\sigma$-SCF
This Python 3 code performs generalized spin $\sigma$-SCF with 2-component spinors. 
Please see these publications for details
+ H.-Z. Ye, M. Welborn, N. D. Ricke, and T. Van Voorhis, $\sigma$-SCF: A direct energy-targeting
method to mean-field excited states, J. Chem. Phys. 147, 214104 (2017)
+ G. David, T. J. P. Irons, A. E. A. Fouda, J. W. Furness, and A. M. Teale, Self-consistent
field methods for excited states in strong magnetic fields: a comparison between energy-
and variance-based approaches, J. Chem. Theory Comput. 17, 5492–5508 (2021).
+ O. B. Oña, G. E. Massacces, J. I. Melo, A. Torre, L. Lain, D. R. Alcoba, and
  J. E. Peralta, Generalized Spin $\sigma$-SCF Method (to be submitted).

### Instructions  
1) Install [PySCF](https://pyscf.org), needed to obtain one- and two-electron integrals.
Then use
```
python integrals.py
```
to generate the integrals and store them. You can edit the `integrals.py` file as needed to change the system at hand.

⚠️ The current version of the code is not optimized for speed.<br>
ℹ️ A version of this code that utilizes Gaussian matrix files can be provided upon request.<br>

2) Run in the command line
```
python sigma.py filename > sigma.out
```
After a successful termination, the output files `Pop.filename.out`  and `Sigma.filename.dat` should be created with the population analysis and the summary of the results, respectively. Edit the file `sigma.py` as needed for the purpose of the calculation. More information can be found as comments in the different files. 

