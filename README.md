# sGB ADM simulation
- Author:       Vít Beneš
- Institution:  [FNSPE CTU](https://fjfi.cvut.cz/cz/)
- Email:        vit.benes@outlook.cz
- GitHub:       [benesvit](https://github.com/benesvit)

This project is a part of a master's thesis named Scalarization of Black Holes in Gauss–Bonnet Gravity at Czech Technical University. It aims at attempting a dynamical simulation of the scalar Gauss–Bonnet gravity (sGB). The ADM formulation for the sGB equation used in this project was derived by [Witek et al.](https://arxiv.org/abs/2004.00009) The code for the simulation is generated using the open-source [NRPy+ package](https://nrpyplus.net/).

### Important project files and structure:
 1. The NRPy+ library modules are located in the [nrpy/](nrpy/) directory. The version of NRPy+ used was accessed on 11. 5. 2021 and is different from the [live version](https://github.com/nrpy/nrpy). We therefore recommend using the NRPy+ library as given here. 
    - Some examples from the authors of NRPy+ are also provided in this directory such as the [Tutorial-Start_to_Finish-BSSNCurvilinear-Two_BHs_Collide.ipynb](nrpy\Tutorial-Start_to_Finish-BSSNCurvilinear-Two_BHs_Collide.ipynb), giving examples how to perform a dynamical simulation with NRPy+ of a collision of two BHs (using the BSSN equations).

2. <a name="item2"></a>The Python code file where the sGB ADM quantities are setup and the equations are constructed in terms of NRPy+ indexed expression is [sGB_ADM_quantities.py](sGB_ADM_quantities.py). This module can:
    - Setup sGB ADM quantities in a general case.
    - Setup sGB ADM quantities in spherical symmetry.
    - Setup the sGB ADM quantities using the initial data:
        1) Schwarzschild black hole in [trumpet coordinates](https://iopscience.iop.org/article/10.1088/0264-9381/31/11/117001), inspired by NRPy+ initial data for the BSSN equations given in [nrpy/BSSN/StaticTrumpet.py](nrpy\BSSN\StaticTrumpet.py).
    - Find the sGB ADM equations in terms of the setup tensors, using the process described in the thesis.
3. **Notebook constructing the C code and performing the simulation [sGB_ADM_simulation.ipynb](sGB_ADM_simulation.ipynb)** for the sGB ADM equations with given initial data. The main results of the thesis are obtained using this notebook.

4. Python code for setting up the standard ADM quantities and equations [ADM_quantities.py](ADM_quantities.py) and notebook performing a numerical simulation using these quantities [ADM_simulation.ipynb](ADM_simulation.ipynb). This was performed in order to see how the sGB ADM formulation for zero coupling constant $\alpha_\text{GB} = 0$ differs from the standard ADM formulation.

5. Code validating the points made in the chapter Numerical Setup - Spherical symmetry about the dependency of sGB ADM equations in spherical symmetry [matrix_validation.ipynb](matrix_validation.ipynb). 
