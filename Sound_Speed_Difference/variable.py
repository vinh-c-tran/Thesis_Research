""" Variable class declaration
    and helper functions for calculating partial derivatives of chemical potential
    numerically via finite differences 

    Author: Vinh Tran 20220223
    Author's Thoughts: Maya just got into Cornell. Insane!! March meeting stuff. 
"""

import numpy as np 
import sympy as sym
from scipy import optimize 

from solver_helper import *


""" Defining Variable class
    A variable object defines one of the variables used when we differentiate the chemical potential.
    Includes: baryon number density nB, baryon fractions and lepton fractions
    Can include other combinations when we include muons as well in the future 
"""

class variable:
    def __init__(self, sym_value, kind = 'independent', fixed = 'fixed'):
        # stores numerical value for the variable 
        self.value = 0.0 
        # tells us if it's independent/dependent
        # currently this functionality is not used 
        self.kind = kind
        # tells us if it's the differentiating variable or not 
        self.fixed = fixed
        # stores symbolic expression for variable 
        self.sym_value = sym_value


""" Helper Functions """

""" Variable Dictionary Generator 
    1. We use a dictionary to link baryons to their respective fraction (likewise for leptons)
    This allows for the ability to generalize this to an arbitrary system with arbitrary particles 
"""
def variable_gen(baryon_list, lepton_list, varying_variable):
    variable_dict = {}
    
    if ('nb' == varying_variable):
        variable_dict['nb'] = variable(sym.symbols('n_B'), 'independent', 'vary')
    else:
        variable_dict['nb'] = variable(sym.symbols('n_B'), 'dependent', 'fixed')
        
        
    for baryon in baryon_list:
        if (baryon == varying_variable and baryon != Proton and baryon != Neutron):
            variable_dict[baryon] = variable(baryon.sym_frac, 'dependent', 'vary')
        elif (baryon == varying_variable and baryon == Proton or baryon == Neutron):
            variable_dict[baryon] = variable(baryon.sym_frac, 'independent', 'vary')
        else:
            variable_dict[baryon] = variable(baryon.sym_frac, 'independent', 'fixed')
            
            
    for lepton in lepton_list:
        if (lepton == varying_variable):
            variable_dict[lepton] = variable(lepton.sym_frac, 'independent', 'vary')
        else: 
            variable_dict[lepton] = variable(lepton.sym_frac, 'independent', 'fixed')
        
    return variable_dict


""" Variable Fill Function 
    1. We connect our variable dictionary to values stored in data arrays indexed by integer i
    and fill in numerical values 
    2. IMPORTANT NOTE
     - As the code is right now, we need to define a frac_dict that links to the data
     - at run time which makes this a little inconvenient. Can try to think of a way to 
     - automate this in the future 
"""

#frac_dict = {} 
#frac_dict[Proton] = proton_frac 
#frac_dict[] 

def variable_fill(variable_dict, frac_dict, baryon_list, lepton_list, integer):
    # fills the variable value object with particle fraction value determined by integer position
    # in the array... 
    variable_dict['nb'].value = hc**3 * n0 * frac_dict['nb'][integer] 
    
    for baryon in baryon_list:
        variable_dict[baryon].value = frac_dict[baryon][integer] 
        
    for lepton in lepton_list: 
        variable_dict[lepton].value = frac_dict[lepton][integer]



""" Generate Sigma Equation of Motion """

""" Helper functions to get the fermi momentum, among other things """

def kf_thing(nb, frac):
    return float((3 * np.pi**2 * nb * frac)**(1/3))

def mass_eff_thing(Baryon, sigma_field):
    return float((Baryon.num_mass - Baryon.num_g_sigma * sigma_field))

def ef_thing(nb, frac, sigma_field, Baryon):
    return np.sqrt(kf_thing(nb, frac)**2 + mass_eff_thing(Baryon, sigma_field)**2)

def log_factor_thing(nb, frac, sigma_field, Baryon):
    numerator = kf_thing(nb, frac) + ef_thing(nb, frac, sigma_field, Baryon) 
    denominator = mass_eff_thing(Baryon, sigma_field) 
    
    return np.log(np.sqrt(numerator**2/denominator**2))

def dU_dsigma(sigma_field):
    term1 = sigma.num_b * Neutron.num_mass * Neutron.num_g_sigma**3 * sigma_field**2
    term2 = sigma.num_c * Neutron.num_g_sigma**4 * sigma_field**3
    
    return term1 + term2 

""" Scalar Density Generator """

def scalar_density_gen(Baryon, sigma_field, variable_dict):
    
    # assuming that variable_dict[baryon].value has already been 
    # populated with fraction value 
    frac = variable_dict[Baryon].value 
    nb = variable_dict['nb'].value  
    
    fermi = kf_thing(nb, frac)
    EF = ef_thing(nb, frac, sigma_field, Baryon)
    m = mass_eff_thing(Baryon, sigma_field)
    log_fac = log_factor_thing(nb, frac, sigma_field, Baryon) 
    
    term1 = fermi * EF
    term2 = - m**2 * log_fac 
    
    return m/(2*np.pi**2) * (term1 + term2) 


""" Sigma equation of motion generator """
def sigma_eom_gen(sigma_field, baryon_list, variable_dict):
    result = sigma.num_mass**2 * sigma_field + dU_dsigma(sigma_field) 
    
    for baryon in baryon_list:
        result -= baryon.num_g_sigma * scalar_density_gen(baryon, sigma_field, variable_dict)
    
    return result 


""" Sigma equation solver """
def sigma_val(sig_guess, baryon_list, var_dict):
    return optimize.fsolve(sigma_eom_gen, sig_guess, args =(baryon_list, var_dict))


""" Helper functions for updating the proton and neutron fraction 
    1. The way that our system is set up, we will have the proton and neutron fractions: x_p, x_n 
    be dependent variables. All other particle fractions will be independent variables. 
    That is, we will need to update them as well whenever 
"""

def baryon_num_conservation_current(var_dict):
    # generate an equation for baryon number conservation 
    lhs = 1.0 
    rhs = 0.0 
    
    for var in var_dict:
        if (var == 'nb'):
            continue
        elif (var.type == 'Baryon'):
            rhs += var.sym_frac
    
    return lhs - rhs


def charge_neutrality_current(var_dict):
    # generate an equation for charge neutrality 
    lhs = 0.0 
    rhs = 0.0 
    
    for var in var_dict:
        if (var == 'nb'):
            continue
        elif (var.charge > 0.0):
            lhs += var.sym_frac
        elif (var.charge < 0.0):
            rhs += var.sym_frac 
    
    return lhs - rhs 


def proton_neutron_frac(var_dict):
    # generate equations 
    chg_neu = charge_neutrality_current(var_dict)
    byn_cons = baryon_num_conservation_current(var_dict)
    
    # update with numerical values 
    for var in var_dict:
        if (var != 'nb' and var != Proton and var != Neutron):
            chg_neu = chg_neu.subs(var.sym_frac, var_dict[var].value)
            byn_cons = byn_cons.subs(var.sym_frac, var_dict[var].value) 
    
    # define system 
    system = [chg_neu, byn_cons]
    symbols = [Neutron.sym_frac, Proton.sym_frac]
    
    # solves for proton and neutron 
    solutions = sym.linsolve(system, symbols) 
    
    # substitute in new values
    var_dict[Neutron].value = solutions.args[0][0]
    var_dict[Proton].value = solutions.args[0][1]


""" Other functions used in calculating the E_F_i """
def fermi_momentum(baryon, var_dict):
    nb_val = var_dict['nb'].value
    baryon_frac = var_dict[baryon].value 
    
    return (3 * np.pi**2 * nb_val * baryon_frac)**(1/3)

def effective_energy(baryon, kf, sigma_field):
    term1 = float(kf**2)
    term2 = (baryon.num_mass - baryon.num_g_sigma * sigma_field)**2
    return np.sqrt(term1 + term2) 


""" Function that takes a step size h and calculates E_F_i(x_0 + h) """
def effective_energy_h_calc(baryon, baryon_list, var_dict, integer, h, sigma_array):
    for var in var_dict:
        if (var_dict[var].fixed == 'vary'):
            var_dict[var].value += h 
    
    # update the proton and neutron fractions 
    proton_neutron_frac(var_dict)
    
    # calculate new fermi momentum 
    kf_current = fermi_momentum(baryon, var_dict) 
    
    # calculate sigma field 
    sig = sigma_val(sigma_array[integer], baryon_list, var_dict)[0]
    
    # calculate effective energy
    ef = effective_energy(baryon, kf_current, sig)
    
    return ef 


""" Putting everything back together """
def E_eff_part_deriv(baryon, baryon_list, lepton_list, integer, h, ind_var, frac_dict):
    
    # generate the dictionary
    var_dict = variable_gen(baryon_list, lepton_list, ind_var)
    
    # fill in values according to specified integer
    # integer specifies a unique point in the phase space as we may recall 
    variable_fill(var_dict, frac_dict, baryon_list, lepton_list, integer)
    
    E_eff_array = np.zeros(4)
    h_array = np.array([-2*h, -h, h, 2*h])
    
    for i in range(4):
        E_eff_array[i] = effective_energy_h_calc(baryon, baryon_list, var_dict, integer, h_array[i])
        # reset values 
        variable_fill(var_dict, baryon_list, lepton_list, integer)
        
    # apply finite differences 
    deriv = 1/(12*h)*(E_eff_array[0] - 8*E_eff_array[1] + 8*E_eff_array[2] - E_eff_array[3])
    
    return deriv








""" Mesonic Contribution """

""" Generating omega, rho, phi mesons 
    1. Generate equation of motion
    2. Differentiate 
"""

def neutron_proton_frac_sym(symbolic_expression, var_dict):
    # generate equations
    chg_neu = charge_neutrality_current(var_dict)
    byn_cons = baryon_num_conservation_current(var_dict)
    
    # define system
    system = [chg_neu, byn_cons]
    symbols = [Neutron.sym_frac, Proton.sym_frac]
    
    # solves for proton and neutron
    solutions = sym.linsolve(system, symbols) 
    neutron_sym = solutions.args[0][0]
    proton_sym = solutions.args[0][1]
    
    # substitute in new values
    symbolic_expression = symbolic_expression.subs([(Proton.sym_frac, proton_sym), (Neutron.sym_frac, neutron_sym)])
    return symbolic_expression 


def omega_gen(baryon_list, var_dict):
    nb = sym.symbols('n_B')
    result = 0
    for baryon in baryon_list:
        result += baryon.sym_g_omega * nb * baryon.sym_frac
    
    return neutron_proton_frac_sym(result/omega.sym_mass**2, var_dict) 


def partial_omega(baryon_list, var_dict, var):
    return sym.diff(omega_gen(baryon_list, var_dict), var)


def rho_gen(baryon_list, var_dict):
    nb = sym.symbols('n_B')
    result = 0
    for baryon in baryon_list:
        result += baryon.isospin * baryon.sym_g_rho * nb * baryon.sym_frac
    
    return neutron_proton_frac_sym(result/rho.sym_mass**2, var_dict)


def partial_rho(baryon_list, var_dict, var):
    return sym.diff(rho_gen(baryon_list, var_dict), var)


def phi_gen(baryon_list, var_dict):
    nb = sym.symbols('n_B')
    result = 0
    for baryon in baryon_list:
        result += baryon.sym_g_phi * nb * baryon.sym_frac
    
    return neutron_proton_frac_sym(result/phi.sym_mass**2, var_dict) 


def partial_phi(baryon_list, var_dict, var):
    return sym.diff(phi_gen(baryon_list, var_dict), var)


""" Mesonic Partial Derivative """
#def partial_mu_meson_sym(baryon, baryon_list, var):
    # calculates the partial derivative of the mesonic contribution to the chemical potential
#    term1 = baryon.sym_g_omega * partial_omega(baryon_list, var)
#    term2 = baryon.sym_g_rho * baryon.isospin * partial_rho(baryon_list, var)
#    term3 = baryon.sym_g_phi * partial_phi(baryon_list, var)
#    
#    return term1 + term2 + term3

def partial_mu_meson(baryon, baryon_list, var_dict, var):
    ind_var = var_dict[var].sym_value 
    
    # calculates the partial derivative of the mesonic contribution to the chemical potential
    term1 = baryon.sym_g_omega * partial_omega(baryon_list, var_dict, ind_var)
    term2 = baryon.sym_g_rho * baryon.isospin * partial_rho(baryon_list, var_dict, ind_var)
    term3 = baryon.sym_g_phi * partial_phi(baryon_list, var_dict, ind_var)
    
    return term1 + term2 + term3


def sym_to_num(sym_expression, baryon_list, meson_list, lepton_list, var_dict):
    for baryon in baryon_list:
        # need a better way of doing this one
        # sum over mesons... 
        sym_expression = sym_expression.subs([(baryon.sym_g_sigma, baryon.num_g_sigma),\
                                              (baryon.sym_g_omega, baryon.num_g_omega),\
                                              (baryon.sym_g_rho, baryon.num_g_rho),\
                                              (baryon.sym_g_phi, baryon.num_g_phi)])
        
        sym_expression = sym_expression.subs([(baryon.sym_frac, var_dict[baryon].value)])
        
    for meson in meson_list:
        sym_expression = sym_expression.subs(meson.sym_mass, meson.num_mass)
        
    for lepton in lepton_list:
        sym_expression = sym_expression.subs(lepton.sym_frac, var_dict[lepton].value)
        
    sym_expression = sym_expression.subs(sym.symbols('n_B'), var_dict['nb'].value)
    
    return float(sym_expression)
    #return sym_expression


def mesonic_contribution(baryon, baryon_list, meson_list, lepton_list, var_dict, var):
    expression = partial_mu_meson(baryon, baryon_list, var_dict, var)
    expression = sym_to_num(expression, baryon_list, meson_list, lepton_list, var_dict)
    
    return expression 


""" Baryon Chemical Potential Partial Derivative """

def chem_pot_partial_deriv(baryon, baryon_list, meson_list, lepton_list, integer, h, ind_var, frac_dict):
    var_dict = variable_gen(baryon_list, lepton_list, ind_var)
    variable_fill(var_dict, frac_dict, baryon_list, lepton_list, integer)
    
    meson_part = mesonic_contribution(baryon, baryon_list, meson_list, lepton_list, var_dict, ind_var) 

    E_ef_contribution = E_eff_part_deriv(baryon, baryon_list, lepton_list, integer, h, ind_var, frac_dict)
    
    return meson_part + E_ef_contribution



""" Leptons """

""" Future function here
    INSERT A FUNCTION THAT UPDATES THE ELECTRON FRACTION WHEN OTHER FRACTIONS ARE CHANGED
    THIS WOULD BE RELEVANT IN THE CASE WHERE WE INCLUDE MUONS AS WELL 
    HAVE TO FIGURE OUT HOW TO DO THIS; WOULD HAVE TO MODIFY HOW WE INITIALIZE THE 
    VARIABLE DICTIONARY AND ACTUALLY MAKE USE OF THE INDEPENDENT/DEPENDENT ATTRIBUTE RATHER
    THAN JUST ASSIGN IT TO PROTONS AND NEUTRONS... 

    UGH WILL TAKE A DAY OR TWO TO FIGURE THIS OUT... 

    ADDING FUNCTIONALITY TAKES TIME
"""


""" 
    The approach here is to generate a symbolic expression for the lepton chemical potential
    Then, we symbolically differentiate to get an analytic expression for the partial derivative
    Then, we convert the symbolic symbols to numerical values to get an actual result 

"""
def k_fermi_sym(lepton):
    pii = sym.symbols('pi')
    nb = sym.symbols('n_B')
    return (3*pii**2 * nb * lepton.sym_frac)**(sym.S(1)/3)


def lepton_chemical_potential_sym(lepton):
    return sym.sqrt(k_fermi_sym(lepton)**2 + lepton.sym_mass**2) 


def lepton_chem_pot_derivative(lepton, var_dict, var):
    ind_var = var_dict[var].sym_value
    return sym.diff(lepton_chemical_potential_sym(lepton), ind_var) 


def sym_to_num_lepton(sym_expression, lepton_list, var_dict):
    for lepton in lepton_list:
        sym_expression = sym_expression.subs(lepton.sym_mass, lepton.num_mass)
        sym_expression = sym_expression.subs(lepton.sym_frac, var_dict[lepton].value)
    
    sym_expression = sym_expression.subs(sym.symbols('n_B'), var_dict['nb'].value)
    sym_expression = sym_expression.subs(sym.symbols('pi'), np.pi)
    
    return float(sym_expression)


def lepton_contribution(lepton, var_dict, frac_dict, baryon_list, lepton_list, integer, var):
    variable_fill(var_dict, frac_dict, baryon_list, lepton_list, integer) 
    expression = lepton_chem_pot_derivative(lepton, var_dict, var)
    return sym_to_num_lepton(expression, lepton_list, var_dict) 
