import sys
import os
import sympy as sp

sys.path.append('nrpy')

# Import necessary NRPy+ modules:
import indexedexp as ixp
import reference_metric as rfm
import NRPy_param_funcs as par

DIM = 3

# Indices that are going to be kept in the M matrix.
indices = [(0,0), (1,1), (2,2)] 

# Set simplification behavior. Simplifying the expressions in
# SymPy here is computationally expensive and NRPy+ performs CSE
# during code generation, so this remained unused.
def set_simplify(simplify = False) -> None:
    """
    Function to set the simplification behavior.
    """
    global SIMPLIFY, my_simplify
    SIMPLIFY = simplify
    my_simplify = sp.simplify if SIMPLIFY else lambda x: x

set_simplify()

# Set up parameters
f = None
def set_f (f_ = sp.Function('f')(sp.Symbol('Phi'))) -> None:
    """
    Function to set f, just in case it needs to be changed outside the module. 
    Args:
        f_ (sp.Function): sympy function of sp.Symbol('Phi')
    """
    global f
    f = f_

set_f()
alphaGB = sp.Symbol('alphaGB')

def set_up_initial_quantities_general() -> None:
    """
    Set up general quantitites
    """
    # Here the generic tensors without any symmetries could be assigned. Maybe not 
    # necessary for our case. 
    return None

# Function to setup to setup the tensors in terms of the initial data, can be used to extract ADD, gammaDD, etc.
def set_up_initial_quantitites_from_ID() -> None:
    """
    Set up initial quantities from initial data (ID) in spherical coordinates.
    This function initializes the metric, extrinsic curvature, lapse, and shift
    in spherical coordinates and then transforms them to the chosen coordinate system.
    """
    global Phi, Ktr, KPhi, ADD, gammaDD, betaU, alpha
    gammaDD = ixp.zerorank2()
    ADD = ixp.zerorank2()
    Phi = sp.sympify(0)
    Ktr = sp.sympify(0)
    KPhi = sp.sympify(0)
    betaU = ixp.zerorank1()
    alpha = sp.sympify(0)

    rthph = ixp.declarerank1("rthph")  # Array representing r, θ, and φ is created.

    # The initial data are setup
    M = sp.symbols('M')
    psi0 = sp.sqrt(1 + M / rthph[0])

    gammaSphDD = ixp.zerorank2()
    gammaSphDD[0][0] = psi0**4
    gammaSphDD[1][1] = psi0**4 * rthph[0]**2
    gammaSphDD[2][2] = psi0**4 * rthph[0]**2 * sp.sin(rthph[1])**2

    KSphDD = ixp.zerorank2()
    KSphDD[0][0] = -M / rthph[0]**2
    KSphDD[1][1] = M
    KSphDD[2][2] = M * sp.sin(rthph[1])**2

    alphaSph = rthph[0] / (rthph[0] + M)
    betaSphU = ixp.zerorank1()

    # Reference metric must be called before setting up the ID.
    if not rfm.have_already_called_reference_metric_function:
        print("Error. Reference metric must be created before setting up the initial data.")
        sys.exit(1)

    # Function for substitution is defined.
    def sympify_integers__replace_rthph(obj, rthph, rthph_of_xx):
        if isinstance(obj, int):
            return sp.sympify(obj)
        return obj.subs(rthph[0], rthph_of_xx[0]).\
            subs(rthph[1], rthph_of_xx[1]).\
            subs(rthph[2], rthph_of_xx[2])

    r_th_ph_of_xx = rfm.xxSph  # rfm.xxSph contains spherical coordinates r, θ, φ written in terms of xx0, xx1, xx2.

    # Substitution is performed.
    alphaSph = sympify_integers__replace_rthph(alphaSph, rthph, r_th_ph_of_xx)
    for i in range(DIM):
        betaSphU[i] = sympify_integers__replace_rthph(betaSphU[i], rthph, r_th_ph_of_xx)
        for j in range(DIM):
            gammaSphDD[i][j] = sympify_integers__replace_rthph(gammaSphDD[i][j], rthph, r_th_ph_of_xx)
            KSphDD[i][j] = sympify_integers__replace_rthph(KSphDD[i][j], rthph, r_th_ph_of_xx)

    # Step 3: Jacobian transformation needs to be performed on the non-scalar quantities.

    # The Jacobian is found by taking the derivative of r, θ, φ wrt. to xx0, xx1, xx2.
    Jac_dUSph_dDrfmUD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            Jac_dUSph_dDrfmUD[i][j] = sp.diff(r_th_ph_of_xx[i], rfm.xx[j])

    # Inverse Jacobian is found using ixp function for matrix inversion.
    Jac_dUrfm_dDSphUD, _ = ixp.generic_matrix_inverter3x3(Jac_dUSph_dDrfmUD)

    # The unrescaled ADM tensors are assigned.
    alpha = alphaSph
    KDD = ixp.zerorank2()
    Phi = sp.sympify(0)
    KPhi = sp.sympify(0)
    for i in range(DIM):
        for j in range(DIM):
            betaU[i] += Jac_dUrfm_dDSphUD[i][j] * betaSphU[j]
            for k in range(DIM):
                for l in range(DIM):
                    gammaDD[i][j] += Jac_dUSph_dDrfmUD[k][i]*Jac_dUSph_dDrfmUD[l][j]*gammaSphDD[k][l]
                    KDD[i][j]     += Jac_dUSph_dDrfmUD[k][i]*Jac_dUSph_dDrfmUD[l][j]*KSphDD[k][l]

    gammaUU, _ = ixp.symm_matrix_inverter3x3(gammaDD)

    for i in range(DIM):
        for j in range(DIM):
            Ktr += KDD[i][j] * gammaUU[i][j]
    for i in range(DIM):
        for j in range(DIM):
            ADD[i][j] = KDD[i][j] - sp.Rational(1,3) * gammaDD[i][j] * Ktr


def set_up_initial_quantitites_sphsym() -> None:
    global Phi, Ktr, KPhi, ADD, gammaDD, betaU, alpha

    if not rfm.have_already_called_reference_metric_function:
        print("Error. Reference metric must be created before setting up the initial data.")
        sys.exit(1)
    
    Phi = sp.Function('Phi')(rfm.xx[0])
    Ktr = sp.Function('Ktr')(rfm.xx[0])
    KPhi = sp.Function('KPhi')(rfm.xx[0])
    alpha = sp.Function('alpha')(rfm.xx[0])

    gammaDD = ixp.zerorank2()
    ADD = ixp.zerorank2()
    betaU = ixp.zerorank1()
    
    b = sp.sin(rfm.xx[1])**2

    for i in range(DIM):
        gammaDD[i][i] = sp.Function(f"gammaDD{i}{i}")(rfm.xx[0])

    gammaDD[2][2] = gammaDD[1][1] * b

    gammaUU,_ = ixp.symm_matrix_inverter3x3(gammaDD)

    # betaU = ixp.zerorank1()
    betaU[0] = sp.Function('betaU0')(rfm.xx[0])

    # ADD = ixp.zerorank2()
    for i in range(DIM):
        ADD[i][i] = sp.Function(f"ADD{i}{i}")(rfm.xx[0])

    ADD[1][1] = - gammaUU[0][0] * ADD[0][0] / (2 * gammaUU[1][1])
    ADD[2][2] = b * ADD[1][1]

def generate_subs(name, rank) -> dict:
    """
    Generates substitution that substitutes functions for sympy symbols
    for compatibility of taking derivatives and using numpy.
    Args:
        name (str): name of the tensor
        rank (int): rank of the tensor
    """
    symmetry = "sym01" if rank == 2 else None
    subs_dict = {}
    if rank == 0:
        subs_dict.update({sp.Function(name)(rfm.xx[0]):sp.Symbol(name)})
    if rank > 0:
        indexed_expr = ixp.declare_indexedexp(rank=rank, symbol=name, symmetry=symmetry, dimension=3)
        if rank == 1:
            for i in range(3):
                subs_dict.update({sp.Function(indexed_expr[i].name)(rfm.xx[0]):sp.Symbol(indexed_expr[i].name)})
        elif rank == 2:
            for i in range(3):
                for j in range(3):
                    subs_dict.update({sp.Function(indexed_expr[i][j].name)(rfm.xx[0]):sp.Symbol(indexed_expr[i][j].name)})
        else:
            raise ValueError("Ranks > 2 are not supported.")
    return subs_dict

def generate_subs_from_list(tensor_names,tensor_ranks) -> dict:
    """
    Generates substitution of functions to sp.Symbols from list of tensor names
    supplemented with a list of corresponding ranks of these tensors.
    Args:
        tensor_names (list): list of tensor names
        tensor_ranks (list): list of tensor ranks
    """
    subs_dict = {}
    if len(tensor_names) == len(tensor_ranks):
        for tensor, rank in zip(tensor_names,tensor_ranks):
            subs_dict.update(generate_subs(tensor,rank))
    else:
        raise ValueError("Both arguments of generate_subs_from_list must be lists of the same size.")
    return subs_dict

def generate_first_derivative_subs(name, rank) -> dict:
    """
    Generates substitution for first derivative of a tensor
    Args:
        name (str): name of the tensor
        rank (int): rank of the tensor
    """
    subs_dict = {}
    symmetry = "sym01" if rank == 2 else None
    indexed_expr = ixp.declare_indexedexp(rank=rank+1, symbol=name+"_dD", symmetry=symmetry, dimension=3)
    var = rfm.xx[0]
    if rank == 0:
        der = sp.diff(sp.Function(name)(var),var)
        subs_dict.update({der:indexed_expr[0]})
    elif rank == 1:
        der = sp.Derivative(sp.Function(name+str(0))(var),var)
        subs_dict.update({der:indexed_expr[0][0]})
    elif rank == 2:
        for i in range(3):
            der = sp.Derivative(sp.Function(name+str(i)+str(i))(var),var)
            subs_dict.update({der:indexed_expr[i][i][0]})
    else:
        raise ValueError("Ranks > 2 are not supported.")
    return(subs_dict)

def generate_first_derivative_subs_from_list(tensor_names,tensor_ranks) -> dict:
    """
    Generates substitution of functions to sp.Symbols from list of tensor names
    supplemented with a list of corresponding ranks of these tensors.
    Args:
        tensor_names (list): list of tensor names
        tensor_ranks (list): list of corresponding ranks of the tensors
    """
    subs_dict = {}
    if len(tensor_names) == len(tensor_ranks):
        for tensor, rank in zip(tensor_names,tensor_ranks):
            subs_dict.update(generate_first_derivative_subs(tensor,rank))
    else:
        raise ValueError("Both arguments of generate_subs_from_list must be lists of the same size.")
    return subs_dict

def generate_second_derivative_subs(name,rank) -> dict:
    """
    Generates second derivative substitution for a tensor.
    Args:
        name (str): name of the tensor
        rank (int): rank of the tensor
    """
    subs_dict = {}
    var = rfm.xx[0]
    if rank == 0:
       subs_dict.update({sp.Derivative(sp.Function(name)(var),var,2):sp.Symbol(name+"_dDD00")})
    elif rank == 2:
        for i in range(3):
            der = sp.Derivative(sp.Function(name+str(i)+str(i))(var),var,2)
            subs_dict.update({der:sp.Symbol(name + '_dDD'+str(i)+str(i)+'00')})
    else:
        raise ValueError("Ranks > 2 are not supported.")
    return(subs_dict)   

def generate_second_derivative_subs_from_list(tensor_names,tensor_ranks) -> dict:
    """
    Generates substitution of functions to sp.Symbols from list of tensor names
    supplemented with a list of corresponding ranks of these tensors.
    Args:
        tensor_names (list): list of tensor names
        tensor_ranks (list): list of corresponding ranks of the tensors
    """
    subs_dict = {}
    if len(tensor_names) == len(tensor_ranks):
        for tensor, rank in zip(tensor_names,tensor_ranks):
            subs_dict.update(generate_second_derivative_subs(tensor,rank))
    else:
        raise ValueError("Both arguments of generate_subs_from_list must be lists of the same size.")
    return subs_dict

def setup_derivatives() -> None:
    """
    Sets up necessary first and second derivatives of tensors.
    """
    global gammaDD_dD, gammaDD_dDD, ADD_dD, betaU_dD, alpha_dD, alpha_dDD, Phi_dD, Phi_dDD, KPhi_dD, Ktr_dD

    KPhi_dD = ixp.zerorank1()
    Ktr_dD = ixp.zerorank1()
    alpha_dD  = ixp.zerorank1()
    Phi_dD  = ixp.zerorank1()

    gammaDD_dD = ixp.zerorank3() 
    ADD_dD = ixp.zerorank3()
    betaU_dD = ixp.zerorank2() 

    gammaDD_dDD = ixp.zerorank4()
    alpha_dDD = ixp.zerorank2()
    Phi_dDD = ixp.zerorank2()

    d_r = lambda element: sp.diff(element, rfm.xx[0])
    d_th = lambda element: sp.diff(element, rfm.xx[1])
    d_p = lambda element: sp.diff(element, rfm.xx[2])

    derivatives = (d_r, d_th, d_p)

    for i, der in enumerate(derivatives):
        alpha_dD[i] = der(alpha)
        Phi_dD[i] = der(Phi)
        KPhi_dD[i] = der(KPhi)
        Ktr_dD[i] = der(Ktr)
    
    for i in range(DIM):
        for j,der in enumerate(derivatives):
            betaU_dD[i][j] = der(betaU[i])

    for i, der_i in enumerate(derivatives):
        for j, der_j in enumerate(derivatives):
            alpha_dDD[i][j] = der_j(der_i(alpha))
            Phi_dDD[i][j] = der_j(der_i(Phi))

    for i in range(DIM):
        for j in range(DIM):
            for k, der in enumerate(derivatives):
                gammaDD_dD[i][j][k] = der(gammaDD[i][j])
                ADD_dD[i][j][k] = der(ADD[i][j])

    for i in range(DIM):
        for j in range(DIM):
            for l,der_l in enumerate(derivatives):
                for k, der_k in enumerate(derivatives):
                    gammaDD_dDD[i][j][l][k] = der_k(der_l(gammaDD[i][j]))

def substitute_quantities() -> None:
    """
    Performs substitution of functions to sympy symbols compatible with NRPy+ in the expressions.
    """
    global alpha, Phi, KPhi, Ktr, gammaDD, ADD, betaU, alpha_dD, Phi_dD, KPhi_dD, Ktr_dD, betaU_dD, alpha_dDD, Phi_dDD, gammaDD_dD, ADD_dD, gammaDD_dDD

    tensors = ['alpha','Phi','KPhi','Ktr','gammaDD','ADD','betaU']
    ranks = [0,0,0,0,2,2,1]

    tensor_sub = generate_subs_from_list(tensors,ranks)
    der_sub = generate_first_derivative_subs_from_list(tensors,ranks)
    der2_sub = generate_second_derivative_subs_from_list(['gammaDD', 'alpha', 'Phi'],[2,0,0])

    alpha = alpha.subs(tensor_sub)
    KPhi = KPhi.subs(tensor_sub)
    Ktr = Ktr.subs(tensor_sub)
    Phi = Phi.subs(tensor_sub)
    
    for i in range(DIM):
        betaU[i] = betaU[i].subs(tensor_sub)
        alpha_dD[i] = alpha_dD[i].subs(der_sub)
        Phi_dD[i] = Phi_dD[i].subs(der_sub)
        KPhi_dD[i] = KPhi_dD[i].subs(der_sub)
        Ktr_dD[i] = Ktr_dD[i].subs(der_sub)
        for j in range(DIM):
            betaU_dD[i][j] = betaU_dD[i][j].subs(der_sub)
            alpha_dDD[i][j] = alpha_dDD[i][j].subs(der2_sub)
            Phi_dDD[i][j] = Phi_dDD[i][j].subs(der2_sub)

    for i in range(DIM):
        for j in range(DIM):
            ADD[i][j] = ADD[i][j].subs(tensor_sub)
            for k in range(DIM):
                ADD_dD[i][j][k] = ADD_dD[i][j][k].subs(der2_sub).subs(der_sub).subs(tensor_sub)
    for i in range(DIM):
        for j in range(DIM):
            gammaDD[i][j] = gammaDD[i][j].subs(tensor_sub)
            for k in range(DIM):
                gammaDD_dD[i][j][k] = (gammaDD_dD[i][j][k].subs(der_sub).subs(tensor_sub))
                for l in range(DIM):
                    gammaDD_dDD[i][j][k][l] = gammaDD_dDD[i][j][k][l].subs(der2_sub).subs(der_sub).subs(tensor_sub)

def generate_substitution_unrescaled_to_rescaled() -> None:
    hDD = ixp.declarerank2('hDD', symmetry= 'sym01')
    aDD = ixp.declarerank2('aDD', symmetry = 'sym01')
    vetU = ixp.declarerank1('vetU')

    betaU_rescaled = ixp.zerorank1()
    gammaDD_rescaled = ixp.zerorank2()
    ADD_rescaled = ixp.zerorank2()

    betaU_dummy = ixp.declarerank1('betaU')
    gammaDD_dummy = ixp.declarerank2('gammaDD',symmetry='sym01')
    ADD_dummy = ixp.declarerank2('ADD',symmetry='sym01')

    for i in range(DIM):
        betaU_rescaled[i] += vetU[i]*rfm.ReU[i]
        for j in range(DIM):
            gammaDD_rescaled[i][j]+= hDD[i][j]*rfm.ReDD[i][j] + rfm.ghatDD[i][j]
            ADD_rescaled[i][j] += aDD[i][j]*rfm.ReDD[i][j]
    
    subs_dict = {}
    subs_dict.update({betaU_dummy[i]:betaU_rescaled[i] for i in range(DIM)})
    subs_dict.update({gammaDD_dummy[i][j]:gammaDD_rescaled[i][j] for i in range(DIM) for j in range(DIM)})
    subs_dict.update({ADD_dummy[i][j]:ADD_rescaled[i][j] for i in range(DIM) for j in range(DIM)})

    # Derivatives here
    hDD_dD = ixp.declarerank3('hDD_dD', symmetry = 'sym01') # \partial_k h_{ij}
    hDD_dupD = ixp.declarerank3('hDD_dupD', symmetry = 'sym01') # \partial_k h_{ij} upwinded FD 

    gammaDD_dD_rescaled = ixp.zerorank3() # \partial_k \gamma_{ij}
    gammaDD_dupD_rescaled = ixp.zerorank3() # \partial_k \gamma_{ij} upwinded FD
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                gammaDD_dD_rescaled[i][j][k] += rfm.ghatDDdD[i][j][k] + hDD_dD[i][j][k]*rfm.ReDD[i][j] + hDD[i][j]*rfm.ReDDdD[i][j][k]
                gammaDD_dupD_rescaled[i][j][k] += rfm.ghatDDdD[i][j][k] + hDD_dupD[i][j][k]*rfm.ReDD[i][j] + hDD[i][j]*rfm.ReDDdD[i][j][k]
    
    gammaDD_dD_dummy = ixp.declarerank3('gammaDD_dD', symmetry = 'sym01')
    gammaDD_dupD_dummy = ixp.declarerank3('gammaDD_dupD', symmetry = 'sym01')

    subs_dict.update({gammaDD_dD_dummy[i][j][k]:gammaDD_dD_rescaled[i][j][k] for i in range(DIM) for j in range(DIM) for k in range(DIM)})
    subs_dict.update({gammaDD_dupD_dummy[i][j][k]:gammaDD_dupD_rescaled[i][j][k] for i in range(DIM) for j in range(DIM) for k in range(DIM)})

    hDD_dDD = ixp.declarerank4('hDD_dDD', symmetry = 'sym01_sym23') # \partial_k \partial_l h_{ij}
    gammaDD_dDD_rescaled = ixp.zerorank4() # \partial_k \partial_l \gamma_{ij}
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    gammaDD_dDD_rescaled[i][j][k][l]  = rfm.ghatDDdDD[i][j][k][l]
                    gammaDD_dDD_rescaled[i][j][k][l] += hDD_dDD[i][j][k][l]*rfm.ReDD[i][j]
                    gammaDD_dDD_rescaled[i][j][k][l] += hDD_dD[i][j][k]*rfm.ReDDdD[i][j][l] + \
                                                hDD_dD[i][j][l]*rfm.ReDDdD[i][j][k]
                    gammaDD_dDD_rescaled[i][j][k][l] += hDD[i][j]*rfm.ReDDdDD[i][j][k][l]
    gammaDD_dDD_dummy = ixp.declarerank4('gammaDD_dDD', symmetry = 'sym01_sym23')
    subs_dict.update({gammaDD_dDD_dummy[i][j][k][l]:gammaDD_dDD_rescaled[i][j][k][l] for i in range(DIM) for j in range(DIM) for k in range(DIM) for l in range(DIM)})

    vetU_dD = ixp.declarerank2("vetU_dD", "nosym") # \partial_j\mathcal{V}^i
    betaU_dD_rescaled = ixp.zerorank2() # \partial_j \beta^i
    for i in range(DIM):
        for j in range(DIM):
            betaU_dD_rescaled[i][j] = vetU_dD[i][j]*rfm.ReU[i] + vetU[i]*rfm.ReUdD[i][j]
    betaU_dD_dummy = ixp.declarerank2('betaU_dD',symmetry = 'nosym')
    subs_dict.update({betaU_dD_dummy[i][j]:betaU_dD_rescaled[i][j] for i in range(DIM) for j in range(DIM)})

            
    aDD_dD = ixp.declarerank3('aDD_dD',symmetry = 'sym01') # \partial_k a_{ij}
    aDD_dupD = ixp.declarerank3('aDD_dupD',symmetry = 'sym01') # \partial_k a_{ij} upwinded FD

    ADD_dD_rescaled = ixp.zerorank3() # \partial_k A_{ij}
    ADD_dupD_rescaled = ixp.zerorank3() # \partial_k A_{ij} upwinded FD

    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                ADD_dD_rescaled[i][j][k] += aDD_dD[i][j][k]*rfm.ReDD[i][j] + aDD[i][j]*rfm.ReDDdD[i][j][k]
                ADD_dupD_rescaled[i][j][k] += aDD_dupD[i][j][k]*rfm.ReDD[i][j] + aDD[i][j]*rfm.ReDDdD[i][j][k]

    ADD_dD_dummy = ixp.declarerank3('ADD_dD',symmetry = 'sym01')
    ADD_dupD_dummy = ixp.declarerank3('ADD_dupD',symmetry = 'sym01')

    subs_dict.update({ADD_dD_dummy[i][j][k]:ADD_dD_rescaled[i][j][k] for i in range(DIM) for j in range(DIM) for k in range(DIM)})
    subs_dict.update({ADD_dupD_dummy[i][j][k]:ADD_dupD_rescaled[i][j][k] for i in range(DIM) for j in range(DIM) for k in range(DIM)})
    return subs_dict

def perform_subs_to_rescaled() -> None:
    """
    Performs substitution of unrescaled quantities to rescaled quantities.
    """
    global betaU, gammaDD, ADD, betaU_dD, gammaDD_dD, ADD_dD

    subs_dict = generate_substitution_unrescaled_to_rescaled()

    for i in range(DIM):
        betaU[i] = betaU[i].subs(subs_dict)
        for j in range(DIM):
            gammaDD[i][j] = gammaDD[i][j].subs(subs_dict)
            ADD[i][j] = ADD[i][j].subs(subs_dict)
            betaU_dD[i][j] = betaU_dD[i][j].subs(subs_dict)
            for k in range(DIM):
                gammaDD_dD[i][j][k] = gammaDD_dD[i][j][k].subs(subs_dict)
                ADD_dD[i][j][k] = ADD_dD[i][j][k].subs(subs_dict)
                for l in range(DIM):
                    gammaDD_dDD[i][j][k][l] = gammaDD_dDD[i][j][k][l].subs(subs_dict)


def setup_sGB_ADM_quantities(get_data_from = 'sphsym', rescaled = True) -> None:
    if get_data_from == 'sphsym':
        set_up_initial_quantitites_sphsym()
    elif get_data_from == 'ID':
        set_up_initial_quantitites_from_ID()
    else:
        raise ValueError("get_data_from must be either 'sphsym' or 'ID'")
    
    # Calculate derivatives from the initial quantities.
    setup_derivatives()
    if get_data_from == 'sphsym':
        substitute_quantities()
    if rescaled:
        perform_subs_to_rescaled()
    global gammaUU, gammadet
    gammaUU, gammadet = ixp.symm_matrix_inverter3x3(gammaDD) # these are redefined here after the substitution


def calculate_mred_S() -> tuple[list,list]: 
    """
    Calculates reduced M based on the index set. Returns M as indexed expression. 
    """
    gammaUU, gammadet = ixp.symm_matrix_inverter3x3(gammaDD)

    # ------- First, basic quantities appearing in sGB ADM equations are defined. -------
    GammaUDD = ixp.zerorank3() # Christoffel symbols \Gamma^i_{jk}
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    GammaUDD[i][j][k] += sp.Rational(1,2) * gammaUU[i][l] * (gammaDD_dD[l][j][k] + gammaDD_dD[l][k][j] - gammaDD_dD[j][k][l])
    Phi_dummy = sp.Symbol('Phi')
    fD = sp.diff(f, Phi_dummy).subs(Phi_dummy, Phi)
    fDD = sp.diff(f, Phi_dummy, 2).subs(Phi_dummy, Phi)

    CD = ixp.zerorank1() # C_i
    for i in range(DIM):
        CD[i] += fDD * KPhi * Phi_dD[i]
        CD[i] += fD*(KPhi_dD[i] - sp.Rational(1, 3)*Ktr*Phi_dD[i])
        for j in range(DIM):
            for k in range(DIM):
                CD[i] += - fD * ADD[i][k] * gammaUU[k][j] * Phi_dD[j]

    CU = ixp.zerorank1() # C^i 
    for i in range(DIM):
        for j in range(DIM):
            CU[i] += CD[j] * gammaUU[j][i]

    Phi_cdDD = ixp.zerorank2() # D_i D_j \Phi
    for i in range(DIM):
        for j in range(DIM):
            Phi_cdDD[i][j] += Phi_dDD[i][j]
            for k in range(DIM):
                Phi_cdDD[i][j] -= GammaUDD[k][i][j] * Phi_dD[k]

    CDD = ixp.zerorank2() # C_{ij}
    for i in range(DIM):
        for j in range(DIM):
            CDD[i][j] += fD * (Phi_cdDD[i][j] - KPhi * (ADD[i][j] + sp.Rational(1, 3) * gammaDD[i][j]*Ktr))
            CDD[i][j] += fDD * Phi_dD[i] * Phi_dD[j]


    Ctr = sp.sympify(0) # C
    for i in range(DIM):
        for j in range(DIM):
            Ctr += gammaUU[i][j] * CDD[i][j]


    CtfDD = ixp.zerorank2() # C_{ij}^{tf}
    for i in range(DIM):
        for j in range(DIM):
            CtfDD[i][j] += CDD[i][j] - sp.Rational(1, 3) * Ctr * gammaDD[i][j]

    CtfUU = ixp.zerorank2() # C^{ij}^{tf}
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    CtfUU[i][j] += CtfDD[k][l] * gammaUU[k][i] * gammaUU[l][j]
    
    gammaUU_dD = ixp.zerorank3() #\partial_k \gamma^{ij}
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for m in range(DIM):
                    for n in range(DIM):
                        gammaUU_dD[i][j][k] -= gammaUU[i][m] * gammaUU[j][n] * gammaDD_dD[m][n][k]


    GammaUDD_dD = ixp.zerorank4() # \partial_m \Gamma^k_{ij}
    for k in range(DIM):
        for i in range (DIM):
            for j in range (DIM):
                for m in range(DIM):
                    for l in range(DIM):
                        GammaUDD_dD[k][i][j][m] += sp.Rational(1,2)*(gammaUU_dD[k][l][m] * (gammaDD_dD[j][l][i] + gammaDD_dD[i][l][j] - gammaDD_dD[i][j][l]) +\
                                                                    gammaUU[k][l] * (gammaDD_dDD[j][l][i][m] + gammaDD_dDD[i][l][j][m] - gammaDD_dDD[i][j][l][m]))

    RDD = ixp.zerorank2()
    for j in range(DIM):
        for k in range(DIM):
            for i in range(DIM):
                RDD[j][k] += GammaUDD_dD[i][j][k][i] - GammaUDD_dD[i][k][i][j]
                for p in range(DIM):
                    RDD[j][k] += GammaUDD[i][i][p] * GammaUDD[p][j][k] - GammaUDD[i][j][p] * GammaUDD[p][i][k]
    for i in range(DIM):
        for j in range(DIM):
            RDD[i][j] = sp.simplify(RDD[i][j])
    
    R = sp.sympify(0)
    for i in range(DIM):
        for j in range(DIM):
            R += gammaUU[i][j]*RDD[i][j]
    R = my_simplify(R)

    RtfDD = ixp.zerorank2(DIM)
    for i in range(DIM):
        for j in range(DIM):
            RtfDD[i][j] = RDD[i][j] - sp.Rational(1,3)*gammaDD[i][j]*R

    
    # ------- Next, variables for subexpressions appearing in M_{red}^D are defined. -------
    AA = sp.sympify(0) # A^{kl}A_{kl}
    for k in range(DIM):
        for l in range(DIM):
            for m in range(DIM):
                for n in range(DIM):
                    AA += gammaUU[m][k] * gammaUU[n][l] * ADD[m][n] * ADD[k][l]
    global Hgr
    Hgr = my_simplify(R - AA + sp.Rational(2,3) * Ktr * Ktr) # \mathcal{H}^{GR}

    MgrD = ixp.zerorank1() # \mathcal{M}^{GR}_i 
    for i in range(DIM):
        MgrD[i] += - sp.Rational(2,3) * Ktr_dD[i]
        for k in range(DIM):
            for l in range(DIM):
                MgrD[i] += gammaUU[l][k] * ADD_dD[i][k][l]
                for a in range(DIM):
                    MgrD[i] += -gammaUU[l][k] * ADD[a][k] * GammaUDD[a][i][l] - gammaUU[l][k] * ADD[i][a] * GammaUDD[a][k][l]

    for i in range(DIM):
        MgrD[i] = my_simplify(MgrD[i])

    EgrDD = ixp.zerorank2() # E^{GR}_{ij}
    for i in range(DIM):
        for j in range(DIM):        
            EgrDD[i][j] += RtfDD[i][j] + sp.Rational(1,3) * Ktr * ADD[i][j] + sp.Rational(1,3) * gammaDD[i][j] * AA
            for k in range(DIM):
                for l in range(DIM):
                    EgrDD[i][j] -= ADD[i][k] * ADD[l][j] * gammaUU[l][k]

    for i in range(DIM):
        for j in range(DIM):
            EgrDD[i][j] = my_simplify(EgrDD[i][j])

    EgrUU = ixp.zerorank2() # E^{GR}^{ij}
    for i in range(DIM):
        for j in range(DIM):
            for m in range(DIM):
                for n in range(DIM):
                    EgrUU[i][j] += EgrDD[m][n] * gammaUU[m][i] * gammaUU[n][j]

    for i in range(DIM):
        for j in range(DIM):
            EgrUU[i][j] = my_simplify(EgrUU[i][j])

    LeviCivitaTensorDDD = ixp.LeviCivitaTensorDDD_dim3_rank3(sp.sqrt(gammadet))
    LeviCivitaTensorDUU = ixp.zerorank3()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for m in range(DIM):
                    for n in range(DIM):
                        LeviCivitaTensorDUU[i][j][k] += gammaUU[k][n] * gammaUU[j][m] * LeviCivitaTensorDDD[i][m][n]

    ADD_cdD = ixp.zerorank3() # \nabla_k A_{ij}
    for j in range(DIM):
        for l in range(DIM):
            for k in range(DIM):
                ADD_cdD[j][l][k] += ADD_dD[j][l][k]
                for m in range(DIM):
                    ADD_cdD[j][l][k] -= GammaUDD[m][j][k] * ADD[m][l] + GammaUDD[m][l][k] * ADD[j][m]

    BDD = ixp.zerorank2() # B_{ij}
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    BDD[i][j] += sp.Rational(1,2)*(LeviCivitaTensorDUU[i][k][l] * ADD_cdD[j][l][k] +\
                                                LeviCivitaTensorDUU[j][k][l] * ADD_cdD[i][l][k])
                    
    FDD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            FDD[i][j] += my_simplify((1 - sp.Rational(1,3) * alphaGB) * gammaDD[i][j] + 2 * alphaGB * CtfDD[i][j])

    HtDDDD = ixp.zerorank4()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    HtDDDD[i][j][k][l] += sp.Rational(1,2) * (gammaDD[k][i] * FDD[j][l] + gammaDD[k][j] * FDD[i][l]) -\
                                            sp.Rational(1,3) * gammaDD[i][j] * FDD[k][l]

    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    HtDDDD[i][j][k][l] = my_simplify(HtDDDD[i][j][k][l])

    HtDDUU = ixp.zerorank4()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    for m in range(DIM):
                        for n in range(DIM):
                            HtDDUU[i][j][k][l] += HtDDDD[i][j][m][n] * gammaUU[m][k] * gammaUU[n][l]

    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    HtDDUU[i][j][k][l] = my_simplify(HtDDUU[i][j][k][l])

    # ------- Now, we find the elements for matrix M_{red}^{D} given in the thesis in eq. (3.2)-(3.7). -------
    # M_{\Phi K}
    MPhiK = - sp.Rational(1,3) * alphaGB * fD * Hgr

    # M_{\Phi A}^{ij}
    MPhiAUU = ixp.zerorank2()
    for k in range(DIM):
        for l in range(DIM):
            MPhiAUU[k][l] = 2 * alphaGB * fD * EgrUU[k][l]
    # M_{KK}
    MKK = 1 - sp.Rational(1,3) * alphaGB * Ctr + sp.Rational(1,12) * alphaGB**2 * fD**2 * Hgr**2
    
    # M_{\Phi A}^{ij}
    MKAUU = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            MKAUU[i][j] += sp.Rational(1,2) * alphaGB * CtfUU[i][j] - sp.Rational(1,2) * alphaGB**2 * fD**2 * Hgr * EgrUU[i][j]
    
    # M_{AK ij}
    MAKDD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            MAKDD[i][j] += - alphaGB * sp.Rational(1,3) * CtfDD[i][j] + alphaGB**2 * fD**2 * sp.Rational(1,3) * Hgr * EgrDD[i][j]

    # M_{AA ij}^{kl}
    MAADDUU = ixp.zerorank4()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    MAADDUU[i][j][k][l] += HtDDUU[i][j][k][l] - 2 * alphaGB**2 * fD**2 * EgrDD[i][j] * EgrUU[k][l]
    
    HtsymDDUU = ixp.zerorank4()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM): 
                    HtsymDDUU[i][j][k][l] += sp.Rational(1,2) * (HtDDUU[i][j][k][l] + HtDDUU[i][j][l][k])

    Mred = ixp.zerorank2(DIM = 2 + len(indices))

    eps = lambda x, y: 1 if x == y else 2 # 1 for the same index and 2 for two different indices
    # Function eps was defined with expr. (3.14) in mind, where there are i,j, i \neq j and for these
    # matrix entries there factors 2. This does not happen in sph. symmetry since we keep only diagonal terms. 
    # This part of the code could be used for finding matrix in (3.14), however the matrix will not be invertible due to A_{ij} \gamma^{ij} = 0.

    # ------- Finally, we define the matrix M_{red}^{D} given in eq (3.25) from the subexpressions defined above. -------
    Mred[0][0] = sp.sympify(1)

    Mred[0][1] = my_simplify(MPhiK)
    
    for i in range(len(indices)):
        row, col = indices[i]
        Mred[0][2+i] = my_simplify(sp.sympify(eps(row,col)) * MPhiAUU[row][col])

    Mred[1][0] = sp.sympify(0)

    Mred[1][1] = my_simplify(MKK)

    for i in range(len(indices)):
        row, col = indices[i]
        Mred[1][2+i] = sp.sympify(eps(row,col)) * MKAUU[row][col]

    for i in range(len(indices)):
        rowi, coli = indices[i]
        Mred[2+i][0] = sp.sympify(0)
        Mred[2+i][1] = MAKDD[rowi][coli]
        for j in range(len(indices)):
            rowj, colj = indices[j]
            Mred[2+i][2+j] = my_simplify(eps(rowj,colj) * MAADDUU[rowi][coli][rowj][colj])

    # ------- Next we will find expressions for source terms S^{\Phi}, S^K, S^{A}. Starting with defining subexpressions appearing in the equations. -------
    alpha_cdDD = ixp.zerorank2() # D_i D_j \alpha
    for i in range(DIM):
        for j in range(DIM):
            alpha_cdDD[i][j] += alpha_dDD[i][j]
            for a in range(DIM):
                alpha_cdDD[i][j] -= GammaUDD[a][i][j] * alpha_dD[a]

    alpha_cdDDtr = sp.sympify(0) #D_i D^i \alpha
    for i in range(DIM):
        for j in range(DIM):
            alpha_cdDDtr += gammaUU[i][j] * alpha_cdDD[i][j]

    alpha_cdDDtf = ixp.zerorank2() #[D_i D_j \alpha]^{tf}
    for i in range(DIM):
        for j in range(DIM):
            alpha_cdDDtf[i][j] += alpha_cdDD[i][j] - sp.Rational(1,3) * alpha_cdDDtr * gammaDD[i][j]

    DDaAAKK = sp.sympify(0) # \frac{1}{\alpha}D^i D_i \alpha - A^2 - \frac{1}{3} K^2
    DDaAAKK = alpha_cdDDtr/alpha - AA - sp.Rational(1,3) * Ktr**2

    BBctr = sp.sympify(0) # double contraction B_{ij} B^{ij}
    for k in range(DIM):
        for j in range(DIM):
            for i in range(DIM):
                for l in range(DIM):
                    BBctr += BDD[k][j] * BDD[i][l] * gammaUU[i][k] * gammaUU[l][j]

    AActrDD = ixp.zerorank2() # contraction A_{km} A^m_j
    for k in range(DIM):
        for j in range(DIM):
            for m in range(DIM):
                for l in range(DIM):
                    AActrDD[k][j] += ADD[k][m] * ADD[l][j] * gammaUU[l][m]

    MgrMgrctr = sp.sympify(0) # M^{GR}_i contraction with itself
    for i in range(DIM):
        for j in range(DIM):
            MgrMgrctr += MgrD[i] * MgrD[j] * gammaUU[i][j]

    KPhi2DPhiDPhi = sp.sympify(0) # K_Phi^2 - D^k\Phi D_k\Phi
    KPhi2DPhiDPhi += KPhi**2
    for i in range(DIM):
        for j in range(DIM):
            KPhi2DPhiDPhi -= gammaUU[i][j] * Phi_dD[i] * Phi_dD[j]

    EgrDDalphaAActr = sp.sympify(0) # Egr^{kl} (\frac{1}{\alpha}[D_k D_l \alpha]^{tf}+A_{km}A^m_l)
    for i in range(DIM):
        for j in range(DIM):
            EgrDDalphaAActr += EgrUU[i][j] * (alpha_cdDDtf[i][j] / alpha + AActrDD[i][j])
    
    # ------- Expressions for source terms S^{\Phi}, S^{K}, S^A are now found, as given in eq. (2.51) - (2.53) in the thesis. --------
    
    # S^{\Phi}
    SPhi = sp.sympify(0)
    for i in range(DIM):
        for j in range(DIM):
            SPhi -=  gammaUU[i][j] * Phi_cdDD[i][j] + gammaUU[i][j] * alpha_dD[i] * Phi_dD[j] / alpha
    SPhi += Ktr * KPhi + sp.Rational(1,3) * alphaGB * fD * Hgr * DDaAAKK -\
            2 * alphaGB * fD * EgrDDalphaAActr + alphaGB * fD * (2 * BBctr - MgrMgrctr)
    
    # S^K
    SK = sp.sympify(0)
    SK += - (1 - sp.Rational(1,3) * alphaGB * Ctr +\
            sp.Rational(1,12) * alphaGB * alphaGB * fD * fD * Hgr * Hgr) * DDaAAKK
    SK += sp.Rational(1,2) * KPhi * KPhi + alphaGB * sp.Rational(1,4) * fDD * Hgr * KPhi2DPhiDPhi
    for i in range(DIM):
        for j in range(DIM):
            SK -= sp.Rational(1,2) * alphaGB * CtfUU[i][j] * (alpha_cdDDtf[i][j] / alpha + AActrDD[i][j])
    SK -= sp.Rational(1,2) * alphaGB * alphaGB * fD * fD * Hgr * EgrDDalphaAActr
    SK -= sp.Rational(1,4) * alphaGB * alphaGB * fD * fD * Hgr * (2 * BBctr - MgrMgrctr)
    SK += alphaGB * sp.Rational(1,3) * Ctr * Hgr
    for i in range(DIM):
        SK -= alphaGB * CU[i] * MgrD[i]
        for j in range(DIM):
            SK -= alphaGB * sp.Rational(1,2) * CtfUU[i][j] * EgrDD[i][j]

    SADD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            SADD[i][j] += (sp.Rational(1,3) * alphaGB * CtfDD[i][j] -\
                        sp.Rational(1,3) * alphaGB * alphaGB * fD * fD * Hgr * EgrDD[i][j]) * DDaAAKK
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    SADD[i][j] += alphaGB * (LeviCivitaTensorDUU[i][k][l] * BDD[j][k] * CD[l] +\
                                        LeviCivitaTensorDUU[j][k][l] * BDD[i][k] * CD[l])

    MgrCctr = sp.sympify(0) # M^{GR}_i C^i
    for i in range(DIM):
        for j in range(DIM):
            MgrCctr += MgrD[i] * CD[j] * gammaUU[i][j]

    for i in range(DIM):
        for j in range(DIM):
            SADD[i][j] -= alphaGB * (sp.Rational(1,2) * (MgrD[i] * CD[j] + MgrD[j] * CD[i]) -\
                                sp.Rational(1,3) * gammaDD[i][j] * MgrCctr)

    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    SADD[i][j] += - (HtDDUU[i][j][k][l] - 2 * alphaGB * alphaGB * fD * fD * EgrDD[i][j] * EgrUU[k][l]) * \
                    (alpha_cdDDtf[k][l] / alpha + AActrDD[k][l] + sp.Rational(1,3) * gammaDD[k][l] * AA)

    Phi_cdDDtr = sp.sympify(0) # D_i D^i \Phi
    for i in range(DIM):
        for j in range(DIM):
            Phi_cdDDtr += gammaUU[i][j] * Phi_cdDD[i][j]

    DPhiDPhictr = sp.sympify(0) # D^i \Phi D_i \Phi
    for i in range(DIM):
        for j in range(DIM):
            DPhiDPhictr += gammaUU[i][j] * Phi_dD[i] * Phi_dD[j]
        
    for i in range(DIM):
        for j in range(DIM):
            SADD[i][j] -= sp.Rational(1,2) * (Phi_cdDD[i][j] - sp.Rational(1,3) * gammaDD[i][j] * Phi_cdDDtr)
            SADD[i][j] += EgrDD[i][j]*(1 + alphaGB * Ctr + alphaGB * fDD * KPhi2DPhiDPhi)
            SADD[i][j] -= alphaGB * alphaGB * fD * fD * EgrDD[i][j] * (2 * BBctr - MgrMgrctr)
    
    # ------- We put the expressions for S into a vector. -------
    S = ixp.zerorank1(DIM = 2 + len(indices))
    S[0] += my_simplify(SPhi)
    S[1] += my_simplify(SK)
    for i in range(len(indices)):
        row, col = indices[i]
        S[2+i] += my_simplify(SADD[row][col])

    # ------- We find expression for H in sGB gravity. -------
    # TODO: Create Hamiltonian constraint elsewhere than here. 
    global H
    H = sp.sympify(0)
    H += Hgr * (1 - alphaGB * sp.Rational(1,3) * Ctr) - sp.Rational(1,2) * (KPhi * KPhi)
    for i in range(DIM):
        for j in range(DIM):
            H -=  sp.Rational(1,2) * Phi_dD[i] * gammaUU[i][j] * Phi_dD[j]
            H += 2 * alphaGB * EgrDD[i][j] * CtfUU[i][j]

    return Mred, S


def calculate_reduced(Mred) -> tuple[list,list]:
    """
    Calculates reduced M as given in eq. (3.45) in the thesis. Returns first 3 rows of M and first 3 rows of
    the vector containing \\tilde{M} expressions.
    """
    spMred = sp.Matrix(Mred)

    b = sp.sin(rfm.xx[1])**2

    Mredsym = []
    for i in range(5):
        row = []
        for j in range(4):
            if j < 3:
                row.append(spMred[i,j])
            elif j == 3:
                row.append(spMred[i,j] + b * spMred[i,j+1])
        Mredsym.append(row)

    reduced = sp.Matrix(Mredsym)

    Mredsymsym = []
    for i in range(5):
        row = []
        for j in range(3):
            if j < 2:
                row.append(reduced[i,j])
            elif j == 2:
                row.append(reduced[i,j] - gammaUU[0][0] * reduced[i,j+1] / (2 * gammaUU[1][1]))
        Mredsymsym.append(row)
    rereduced = sp.Matrix(Mredsymsym)

    return rereduced[:3,:].tolist(), (reduced[:3,-1]).tolist()

def invert_reduced(input_array) -> list:
    """
    Calculates inversion of a matrix of the form [[1, a, b],[0, c, d],[0, e, f]]
    """
    denom = input_array[1][1] * input_array[2][2] - input_array[1][2] * input_array[2][1]
    
    # Map to inversion
    output_array = [
        [1, (-input_array[0][1] * input_array[2][2] + input_array[0][2] * input_array[2][1]) / denom, 
            (input_array[0][1] * input_array[1][2] - input_array[0][2] * input_array[1][1]) / denom],
        [0, input_array[2][2] / denom, -input_array[1][2] / denom],
        [0, -input_array[2][1] / denom, input_array[1][1] / denom]
    ]
    
    return output_array

def calculate_lie_derivatives():
    """
    Calculates the vector of Lie derivatives appearing in eq. (3.56).
    """
    li_rhs = ixp.zerorank1(DIM = 3)
    li_rhs[0] = betaU[0] * sp.Symbol('KPhi_dupD0')
    li_rhs[1] = betaU[0] * sp.Symbol('Ktr_dupD0')
    # Following term is coming from the part \beta^r \partial_r A_{rr} - 2 \partial_r \beta^r A_{rr},
    # where after plugging the rescalin A_{rr} = ReDD[0][0] a_{rr} we get the following expression
    li_rhs[2] = rfm.ReDD[0][0] * betaU[0] * sp.Symbol('aDD_dupD000') + betaU[0] * sp.Symbol('aDD00') * rfm.ReDDdD[0][0][0] \
                - 2 * betaU_dD[0][0] * sp.Symbol('aDD00') * rfm.ReDD[0][0]
    return li_rhs

def calculate_M_rhs():
    """
    Calculates the RHS for KPhi, Ktr and A_{rr} evolution equations in spherical symmetry.
    """
    global KPhi_rhs, Ktr_rhs, aDD_rhs
    # Calculate matrix M and the matter vector S
    Mred, S = calculate_mred_S()

    # Find the reduced matrix  due to symmetries of A_{ij}
    # and the vector that goes to the RHS of the equations (from trace of \mathcal{L}_n A_{ij})
    reduced,M_rhs_vec = calculate_reduced(Mred)

    # Invert the reduced matrix
    Mredinv = invert_reduced(reduced)

    # Calculate factor A_{ij}A^{ij}/\gamma^{\theta\theta}
    factor = 0
    for i in range(DIM):
        factor += ADD[i][i] * ADD[i][i] * gammaUU[i][i] * gammaUU[i][i] / gammaUU[1][1]

    # Assemble the RHS of the equations 
    spM_rhs_vec = sp.Matrix(M_rhs_vec)
    spM_rhs_vec = factor * spM_rhs_vec
    RHS_vec = spM_rhs_vec + sp.Matrix(S[:3])
    spMredinv = sp.Matrix(Mredinv)
    RHS = alpha * spMredinv * RHS_vec - sp.Matrix(calculate_lie_derivatives())
    RHS[2] = RHS[2] / rfm.ReDD[0][0] # Rescale A_{rr} RHS so it becomes a_{rr} RHS

    KPhi_rhs = RHS[0]
    Ktr_rhs = RHS[1]
    # All other components of the aDD rhs can be set to zero, as we have
    # already rewritten the equation in terms of a_{rr}. This makes it easier
    # to implement in NRPy+. 
    aDD_rhs = ixp.zerorank2()
    aDD_rhs[0][0] = RHS[2]

def calculate_Phi_rhs():
    global Phi_rhs
    Phi_dupD = ixp.declarerank1('Phi_dupD')
    Phi_rhs = -alpha*KPhi
    for i in range(DIM):
        Phi_rhs += betaU[i] * Phi_dupD[i]

def calculate_hDD_rhs():
    global hDD_rhs
    gammDD_dupD_subs = {sp.Symbol(f'gammaDD_dD{ind}'): sp.Symbol(f'gammaDD_dupD{ind}') for ind in ['000','110']}
    hDD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            hDD[i][j] = sp.simplify((gammaDD[i][j] - rfm.ghatDD[i][j])/rfm.ReDD[i][j])
    aDD = ixp.zerorank2()

    for i in range(DIM):
        for j in range(DIM):
            aDD[i][j] = ADD[i][j]/rfm.ReDD[i][j]

    ind = [(0,0),(1,1)]
    hrr_hthth = ixp.zerorank1(DIM = len(ind))
    for i in range(len(ind)):
        row, col = ind[i]
        hrr_hthth[i] = -2 * alpha * (aDD[row][col] + sp.Rational(1,3) * Ktr * (hDD[row][col] + (rfm.ghatDD[row][col]) / rfm.ReDD[row][col]))
        for k in range(DIM):
            hrr_hthth[i] += (betaU[k] * gammaDD_dD[row][col][k].subs(gammDD_dupD_subs) + betaU_dD[k][row] * gammaDD[k][col] + betaU_dD[k][col] * gammaDD[k][col]) / rfm.ReDD[row][col]
    
    hDD_rhs = ixp.zerorank2(DIM)
    hDD_rhs[0][0] = hrr_hthth[0]
    hDD_rhs[1][1] = hrr_hthth[1]

# Functions setting up the gauges are defined. 
def calculate_alpha_rhs(alpha_gauge='log-slicing'):
    global alpha_rhs
    if alpha_gauge == 'log-slicing':
        alpha_rhs = -alpha * 2 * Ktr
        alpha_dupD = ixp.zerorank1()
        alpha_dupD[0] = sp.Symbol('alpha_dupD0')
        for i in range(DIM):
            alpha_rhs += betaU[i] * alpha_dupD[i]
    else:
        raise ValueError("Only log-slicing gauge is implemented.")

def calculate_vetU_rhs(beta_gauge='frozen'):
    global vetU_rhs
    vetU_rhs = ixp.zerorank1()
    if beta_gauge == 'frozen':
        pass
    else:
        raise ValueError("Only frozen gauge is implemented.")

def calculate_rhs():
    calculate_M_rhs()
    calculate_Phi_rhs()
    calculate_hDD_rhs()
    calculate_alpha_rhs()
    calculate_vetU_rhs()
