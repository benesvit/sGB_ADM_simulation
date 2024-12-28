import sys
import os
import sympy as sp

sys.path.append('nrpy')

# Import necessary NRPy+ modules:
import indexedexp as ixp
import reference_metric as rfm
import NRPy_param_funcs as par

DIM = 3

# Set coordinates and reference metric, default on Spherical
par.set_parval_from_str("reference_metric::CoordSystem", "Spherical")

if not rfm.have_already_called_reference_metric_function:
    rfm.reference_metric()

def set_KO(enable_KO = False):
    global KO_enabled
    KO_enabled = enable_KO

set_KO()

def setup_ADM_quantities():
    global gammaDD, ADD, Ktr, alpha, betaU, hDD, aDD, vetU

    gammaDD = ixp.zerorank2()
    ADD = ixp.zerorank2()
    Ktr = sp.symbols('Ktr')
    alpha = sp.symbols('alpha')
    betaU = ixp.zerorank1()

    hDD = ixp.declarerank2("hDD", "sym01")
    aDD = ixp.declarerank2("aDD", "sym01")
    vetU = ixp.declarerank1("vetU")

    for i in range(DIM):
        betaU[i] = rfm.ReU[i]*vetU[i]
        for j in range(DIM):
            gammaDD[i][j] = hDD[i][j]*rfm.ReDD[i][j] + rfm.ghatDD[i][j]
            ADD[i][j] = aDD[i][j]*rfm.ReDD[i][j]
    
def setup_ADM_equations():
    global gammaUU, gammadet, hDD_dD, hDD_dupD, gammaDD_dD, gammaDD_dupD, hDD_dDD, gammaDD_dDD, vetU_dD, betaU_dD, aDD_dD, aDD_dupD, ADD_dD, ADD_dupD
    gammaUU,gammadet = ixp.symm_matrix_inverter3x3(gammaDD)

    hDD_dD = ixp.declarerank3('hDD_dD', symmetry = 'sym01') # \partial_k h_{ij}
    hDD_dupD = ixp.declarerank3('hDD_dupD', symmetry = 'sym01') # \partial_k h_{ij} upwinded FD 

    gammaDD_dD = ixp.zerorank3() # \partial_k \gamma_{ij}
    gammaDD_dupD = ixp.zerorank3() # \partial_k \gamma_{ij} upwinded FD
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                gammaDD_dD[i][j][k] += rfm.ghatDDdD[i][j][k] + hDD_dD[i][j][k]*rfm.ReDD[i][j] + hDD[i][j]*rfm.ReDDdD[i][j][k]
                gammaDD_dupD[i][j][k] += rfm.ghatDDdD[i][j][k] + hDD_dupD[i][j][k]*rfm.ReDD[i][j] + hDD[i][j]*rfm.ReDDdD[i][j][k]


    hDD_dDD = ixp.declarerank4('hDD_dDD', symmetry = 'sym01_sym23') # \partial_k \partial_l h_{ij}
    gammaDD_dDD = ixp.zerorank4() # \partial_k \partial_l \gamma_{ij}
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    gammaDD_dDD[i][j][k][l]  = rfm.ghatDDdDD[i][j][k][l]
                    gammaDD_dDD[i][j][k][l] += hDD_dDD[i][j][k][l]*rfm.ReDD[i][j]
                    gammaDD_dDD[i][j][k][l] += hDD_dD[i][j][k]*rfm.ReDDdD[i][j][l] + \
                                                hDD_dD[i][j][l]*rfm.ReDDdD[i][j][k]
                    gammaDD_dDD[i][j][k][l] += hDD[i][j]*rfm.ReDDdDD[i][j][k][l]

    vetU_dD = ixp.declarerank2("vetU_dD", "nosym") # \partial_j\mathcal{V}^i
    betaU_dD = ixp.zerorank2() # \partial_j \beta^i
    for i in range(DIM):
        for j in range(DIM):
            betaU_dD[i][j] = vetU_dD[i][j]*rfm.ReU[i] + vetU[i]*rfm.ReUdD[i][j]

    alpha_dD = ixp.declarerank1('alpha_dD')
    alpha_dDD = ixp.declarerank2('alpha_dDD',symmetry = 'sym01')
            
    aDD_dD = ixp.declarerank3('aDD_dD',symmetry = 'sym01') # \partial_k a_{ij}
    aDD_dupD = ixp.declarerank3('aDD_dupD',symmetry = 'sym01') # \partial_k a_{ij} upwinded FD

    ADD_dD = ixp.zerorank3() # \partial_k A_{ij}
    ADD_dupD = ixp.zerorank3() # \partial_k A_{ij} upwinded FD


    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                ADD_dD[i][j][k] += aDD_dD[i][j][k]*rfm.ReDD[i][j] + aDD[i][j]*rfm.ReDDdD[i][j][k]
                ADD_dupD[i][j][k] += aDD_dupD[i][j][k]*rfm.ReDD[i][j] + aDD[i][j]*rfm.ReDDdD[i][j][k]

    # \Gamma^i_{jk}
    GammaUDD = ixp.zerorank3() 
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    GammaUDD[i][j][k] += sp.Rational(1,2)*gammaUU[i][l]*(gammaDD_dD[l][j][k] + gammaDD_dD[l][k][j] - gammaDD_dD[j][k][l])
    
    #\partial_k \gamma^{ij}
    gammaUU_dD = ixp.zerorank3() 
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for m in range(DIM):
                    for n in range(DIM):
                        gammaUU_dD[i][j][k] -= gammaUU[i][m]*gammaUU[j][n]*gammaDD_dD[m][n][k]

    # \partial_m \Gamma^k_{ij}
    GammaUDD_dD = ixp.zerorank4() 
    for k in range(DIM):
        for i in range (DIM):
            for j in range (DIM):
                for m in range(DIM):
                    for l in range(DIM):
                        GammaUDD_dD[k][i][j][m] += sp.Rational(1,2)*(gammaUU_dD[k][l][m]*(gammaDD_dD[j][l][i] + gammaDD_dD[i][l][j] - gammaDD_dD[i][j][l]) +\
                                                                    gammaUU[k][l]*(gammaDD_dDD[j][l][i][m] + gammaDD_dDD[i][l][j][m] - gammaDD_dDD[i][j][l][m]))
    # Ricci tensor
    RDD = ixp.zerorank2()
    for j in range(DIM):
        for k in range(DIM):
            for i in range(DIM):
                RDD[j][k] += GammaUDD_dD[i][j][k][i] - GammaUDD_dD[i][k][i][j]
                for p in range(DIM):
                    RDD[j][k] += GammaUDD[i][i][p]*GammaUDD[p][j][k] - GammaUDD[i][j][p]*GammaUDD[p][i][k]

    # Ricci scalar
    R = sp.sympify(0)
    for i in range(DIM):
        for j in range(DIM):
            R += gammaUU[i][j]*RDD[i][j]

    # R^{tf}_{ij}
    RDDtf = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            RDDtf[i][j] = RDD[i][j] - sp.Rational(1,3)*gammaDD[i][j]*R
    
    # D_i D_j \alpha
    alpha_cdDD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            alpha_cdDD[i][j] += alpha_dDD[i][j]
            for k in range(DIM):
                alpha_cdDD[i][j] -= GammaUDD[k][i][j]*alpha_dD[k]

    # D^i D_i \alpha
    alpha_cdDDtr = sp.sympify(0)
    for i in range(DIM):
        for j in range(DIM):
            alpha_cdDDtr += gammaUU[i][j]*alpha_cdDD[i][j]

    alpha_cdDDtf = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            alpha_cdDDtf[i][j] = alpha_cdDD[i][j] - sp.Rational(1,3)*gammaDD[i][j]*alpha_cdDDtr

    # A_{ij}A^{ij}
    ADDAUU = 0
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                for l in range(DIM):
                    ADDAUU += ADD[i][j]*gammaUU[i][k]*gammaUU[j][l]*ADD[k][l]

    # A^i_j
    AUD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                AUD[i][j] += gammaUU[i][k]*ADD[k][j]
    
    # A^k_i A_{kj}
    AUDADD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                AUDADD[i][j] += AUD[i][k]*ADD[k][j]

    # \mathcal{L}_{\beta} \gamma_{ij}
    lieD_gammaDD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                lieD_gammaDD[i][j] +=  gammaDD_dD[i][j][k]*betaU[k] - betaU_dD[k][i]*gammaDD[k][j] - betaU_dD[k][j]*gammaDD[i][k]

    # \mathcal{L}_{\beta} A_{ij}
    lieD_ADD = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                lieD_ADD[i][j] +=  ADD_dD[i][j][k]*betaU[k] - betaU_dD[k][i]*ADD[k][j] - betaU_dD[k][j]*ADD[i][k]
    
    global hDD_rhs, aDD_rhs,Ktr_rhs, alpha_rhs, vetU_rhs, Hconstr

    hDD_rhs = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            hDD_rhs[i][j] += -2*alpha*(ADD[i][j]+ sp.Rational(1,3)*Ktr*gammaDD[i][j])/rfm.ReDD[i][j] + lieD_gammaDD[i][j]

    Ktr_rhs = -alpha_cdDDtr + alpha*(ADDAUU + sp.Rational(1,3)*Ktr**2)

    aDD_rhs = ixp.zerorank2()
    for i in range(DIM):
        for j in range(DIM):
            aDD_rhs[i][j] += (-alpha_cdDDtf[i][j] + alpha*(RDDtf[i][j] - 2*AUDADD[i][j] + sp.Rational(1,3)*ADD[i][j]*Ktr))/rfm.ReDD[i][j] + lieD_ADD[i][j]

    alpha_rhs = -alpha*2*Ktr
    alpha_dupD = ixp.declarerank1('alpha_dupD')
    for i in range(DIM):
        alpha_rhs += betaU[i]*alpha_dupD[i]

    vetU_rhs = ixp.zerorank1()

    if KO_enabled:
        eps_0 = par.Cparameters("REAL", "KO_Dissipation", "eps_0", 0.99)
        r_ko = par.Cparameters("REAL", "KO_Dissipation", "r_ko", 2)
        w_ko = par.Cparameters("REAL", "KO_Dissipation", "w_ko", 0.17)
        # diss_strength = eps_0*sp.erf((rfm.xx[0] -  r_ko) / w_ko)/2
        diss_strength  = w_ko

        alpha_dKOD = ixp.declarerank1("alpha_dKOD")
        vetU_dKOD = ixp.declarerank2("vetU_dKOD", "nosym")
        kDD_dKOD = ixp.declarerank3("kDD_dKOD", "sym01")
        hDD_dKOD = ixp.declarerank3("hDD_dKOD", "sym01")
        for k in range(3):
            alpha_rhs += diss_strength * alpha_dKOD[k] * rfm.ReU[k]  # ReU[k] = 1/scalefactor_orthog_funcform[k]
            for i in range(3):
                vetU_rhs[i] += diss_strength * vetU_dKOD[i][k] * rfm.ReU[k]  # ReU[k] = 1/scalefactor_orthog_funcform[k]
                for j in range(3):
                    kDD_rhs[i][j] += diss_strength * kDD_dKOD[i][j][k] * rfm.ReU[k]  # ReU[k] = 1/scalefactor_orthog_funcform[k]
                    hDD_rhs[i][j] += diss_strength * hDD_dKOD[i][j][k] * rfm.ReU[k]  # ReU[k] = 1/scalefactor_orthog_funcform[k]

    # K_{ij}K^{ij} = A_{ij}A^{ij} + 1/3 K^2
    KK = ADDAUU + sp.Rational(1,3)*Ktr**2
    Hconstr = R + Ktr*Ktr - KK
# setup_ADM_quantities()
# setup_ADM_equations()


