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
1) Install [PySCF](https://pyscf.org) to obtain one- and two-electron integrals.
Then use
```
python integrals.py
```
to generate the integrals. You can edit the `integrals.py` file as needed.
[!WARNING]
The current version of the code is not optimized for speed
[!NOTE] A version of this code that utilizes Gaussian matrix files can be provided upon request.

3) Run in the command line
   ```
   python sigma.py filename > sigma.out
   ```
The output files `Pop.filename.out`  and `Reco.filename.dat`

