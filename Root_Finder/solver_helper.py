""""""

import numpy as np
import pandas as pd
import sympy as sym
from broyden import *


# important constants
hc = 197.33
#hc = 197.32698
#n0 = 0.153

# common symbols
Pi = sym.symbols('pi')

# coupling constant symbols 
g_sigma_N_sym, g_omega_N_sym, g_rho_N_sym, g_phi_N_sym = sym.symbols('g_sigma_N, g_omega_N, g_rho_N, g_phi_N')
b_sym, c_sym = sym.symbols('b, c')
g_sigma_H_sym, g_omega_H_sym, g_rho_H_sym, g_phi_H_sym = sym.symbols('g_sigma_H, g_omega_H, g_rho_H, g_phi_H')

# Class declaration 
class eos:
    # equation of state coupling constants 
    def __init__(self, n0 = 0.153,
                g_sigma_N = 0.0, g_omega_N = 0.0, g_rho_N = 0.0, g_phi_N = 0.0, b = 0.0, c = 0.0,\
                g_sigma_H = 0.0, g_omega_H = 0.0, g_rho_H = 0.0, g_phi_H = 0.0,\
                g_sigma_l = 0.0, g_omega_l = 0.0, g_rho_l = 0.0, g_phi_l = 0.0,\
                g_sigma_sig = 0.0, g_omega_sig = 0.0, g_rho_sig = 0.0, g_phi_sig= 0.0,\
                g_sigma_xi = 0.0, g_omega_xi = 0.0, g_rho_xi = 0.0, g_phi_xi = 0.0):
        
        # saturation density 
        self.n0 = n0

        # nucleon couplings 
        self.g_sigma_N = g_sigma_N
        self.g_omega_N = g_omega_N
        self.g_rho_N = g_rho_N
        self.g_phi_N = g_phi_N

        # sigma self coupling 
        self.b = b
        self.c = c

        # universal hyperon coupling
        self.g_sigma_H = g_sigma_H
        self.g_omega_H = g_omega_H
        self.g_rho_H = g_rho_H
        self.g_phi_H = g_phi_H

        # Lambda baryon couplings 
        self.g_sigma_l = g_sigma_l
        self.g_omega_l = g_omega_l
        self.g_rho_l = g_rho_l
        self.g_phi_l = g_phi_l

        # Sigma baryon couplings
        self.g_sigma_sig = g_sigma_sig
        self.g_omega_sig = g_omega_sig
        self.g_rho_sig = g_rho_sig
        self.g_phi_sig = g_phi_sig

        # Xi baryon couplings 
        self.g_sigma_xi = g_sigma_xi
        self.g_omega_xi = g_omega_xi 
        self.g_rho_xi = g_rho_xi 
        self.g_phi_xi = g_phi_xi

class baryon:
    # baryon class, stores both symbolic and numeric values in a single class 
    # baryon particle 
    def __init__(self, name, spin, isospin, charge, kind, var_type,\
                sym_mass, sym_mass_eff, sym_density, sym_frac, sym_kf, sym_ef, sym_chem_pot,\
                num_mass = 0.0, num_mass_eff = 0.0, num_density = 0.0, num_frac = 0.0, num_kf = 0.0, num_ef = 0.0, num_chem_pot = 0.0):

        # variables common to both classes
        self.name = name
        self.kind = kind
        self.var_type = var_type 
        self.charge = charge 
        self.spin = spin 
        self.isospin = isospin 
        self.type = 'Baryon'    

        # variables to be established at baryon declaration
        self.sym_mass = sym_mass
        self.num_mass = num_mass

        # variables to be stored later
        self.sym_mass_eff = sym_mass_eff
        self.sym_num_density = sym_density
        self.sym_frac = sym_frac
        self.sym_kf = sym_kf
        self.sym_ef = sym_ef
        self.sym_chem_pot = sym_chem_pot

        self.num_mass_eff = num_mass_eff
        self.num_num_density = num_density
        self.num_frac = num_frac
        self.num_kf = num_kf
        self.num_ef = num_ef
        self.num_chem_pot = num_chem_pot
        

        # coupling constants 
        self.sym_g_sigma = 0.0
        self.sym_g_omega = 0.0 
        self.sym_g_rho = 0.0 
        self.sym_g_phi = 0.0 

        self.num_g_sigma = 0.0
        self.num_g_omega = 0.0 
        self.num_g_rho = 0.0 
        self.num_g_phi = 0.0


class lepton:
    # lepton particle 
    def __init__(self, name, charge, var_type,\
                sym_mass, sym_density, sym_frac, sym_kf, sym_chem_pot,\
                num_mass = 0.0, num_density = 0.0, num_frac = 0.0, num_kf = 0.0, num_chem_pot = 0.0):
        self.name = name
        self.charge = charge
        self.var_type = var_type
        self.type = 'Lepton'


        self.sym_mass = sym_mass
        self.sym_density = sym_density
        self.sym_frac = sym_frac
        self.sym_kf = sym_kf
        self.sym_chem_pot = sym_chem_pot

        self.num_mass = num_mass
        self.num_density = num_density
        self.num_frac = num_frac
        self.num_kf = num_kf
        self.num_chem_pot = num_chem_pot



class meson:
    def __init__(self, name, sym_mass, sym_field, num_mass, num_field = 0.0, g_N = 0.0, g_H = 0.0):
        self.name = name 
        self.sym_mass = sym_mass # in MeV
        self.sym_field = sym_field
        self.type = 'Meson'

        self.num_mass = num_mass
        self.num_field = num_field

        # coupling constants 
        # could update this in the future to store coupling constants here
        # but for now only makes sense for the sigma meson to store self coupling 

        self.sym_b = sym.symbols('b')
        self.sym_c = sym.symbols('c')
        self.num_b = 0.0 
        self.num_c = 0.0 

        # eos coupling
        self.g_N = g_N
        self.g_H = g_H 


class independent_variable:
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol 



# Pre-initialize our objects so that at run time all we need to do is
# call the lists!! 
""" Baryons """

# symbolic baryon objects
Proton = baryon(name = 'Proton', spin = 1/2, isospin = 1/2, charge = 1, kind = 'Nucleon', var_type = 'Dependent',\
                sym_mass = sym.symbols('m_p'), sym_mass_eff = sym.symbols('m_p^*'), sym_density = sym.symbols('n_p'),\
                sym_frac = sym.symbols('x_p'), sym_kf = sym.symbols('k_p'), sym_ef = sym.symbols('E^*_p'), sym_chem_pot = sym.symbols('mu_p'),\
                num_mass = 939.0)

Neutron = baryon(name = 'Neutron', spin = 1/2, isospin = -1/2, charge = 0.0, kind = 'Nucleon', var_type = 'Dependent',\
                sym_mass = sym.symbols('m_n'), sym_mass_eff = sym.symbols('m_n^*'), sym_density = sym.symbols('n_n'),\
                sym_frac = sym.symbols('x_n'), sym_kf = sym.symbols('k_n'), sym_ef = sym.symbols('E^*_n'), sym_chem_pot = sym.symbols('mu_n'),\
                num_mass = 939.0)

Lambda = baryon(name = 'Lambda', spin = 1/2, isospin = 0, charge = 0, kind = 'Hyperon', var_type = 'Indepdent',\
                sym_mass = sym.symbols('m_Lambda'), sym_mass_eff =  sym.symbols('m_Lambda^*'), sym_density = sym.symbols('n_Lambda'),\
                sym_frac =  sym.symbols('x_Lambda'), sym_kf = sym.symbols('k_Lambda'), sym_ef = sym.symbols('E^*_Lambda'), sym_chem_pot = sym.symbols('mu_Lambda'),\
                num_mass = 1116.0)

Sigma_neu = baryon(name = 'Sigma 0', spin = 1/2, isospin = 0.0, charge = 0.0, kind = 'Hyperon', var_type = '',\
                    sym_mass = sym.symbols('m_Sigma_0'), sym_mass_eff = sym.symbols('m_Sigma_0^*'),\
                    sym_density = sym.symbols('n_Sigma_0'), sym_frac = sym.symbols('x_Sigma_0'), sym_kf = sym.symbols('k_Sigma'),\
                    sym_ef = sym.symbols('E^*_Sigma'), sym_chem_pot = sym.symbols('mu_Sigma'),\
                    num_mass = 1192.642)

Sigma_plus = baryon(name = 'Sigma +', spin = 1/2, isospin = 1.0, charge = 1.0, kind = 'Hyperon', var_type = '',\
                    sym_mass = sym.symbols('m_Sigma_+'), sym_mass_eff = sym.symbols('m_Sigma_+^*'),\
                    sym_density = sym.symbols('n_Sigma_+'), sym_frac = sym.symbols('x_Sigma_+'), sym_kf = sym.symbols('k_Sigma_+'),\
                    sym_ef = sym.symbols('E^*_Sigma_+'), sym_chem_pot = sym.symbols('mu_Sigma_+'),\
                    num_mass = 1189.37)

Sigma_min = baryon(name = 'Sigma -', spin = 1/2, isospin = -1.0, charge = -1.0, kind = 'Hyperon', var_type = '',\
                    sym_mass = sym.symbols('m_Sigma_-'), sym_mass_eff = sym.symbols('m_Sigma_-^*'),\
                    sym_density = sym.symbols('n_Sigma_-'), sym_frac = sym.symbols('x_Sigma_-'), sym_kf = sym.symbols('k_Sigma_-'),\
                    sym_ef = sym.symbols('E^*_Sigma_-'), sym_chem_pot = sym.symbols('mu_Sigma_-'),\
                    num_mass = 1197.5)

xi_neu_sym = baryon(name = 'Xi 0', spin = 1/2, isospin = 1/2, charge = 0.0, kind = 'Hyperon', var_type = '',\
                    sym_mass = sym.symbols('m_Xi_0'), sym_mass_eff =  sym.symbols('m_Xi_0^*'),\
                    sym_density = sym.symbols('n_Xi_0'), sym_frac = sym.symbols('x_Xi_0'), sym_kf =  sym.symbols('k_Xi_0'),\
                    sym_ef = sym.symbols('E^*_Xi_0'), sym_chem_pot = sym.symbols('mu_Xi_0'),\
                    num_mass = 1314.86)

xi_min_sym = baryon(name = 'Xi -', spin = 1/2, isospin = -1/2, charge = -1.0, kind = 'Hyperon', var_type = '',\
                    sym_mass = sym.symbols('m_Xi_-'), sym_mass_eff = sym.symbols('m_Xi_-^*'),\
                    sym_density = sym.symbols('n_Xi_-'), sym_frac = sym.symbols('x_Xi_-'), sym_kf = sym.symbols('k_Xi_-'),\
                    sym_ef = sym.symbols('E^*_Xi_-'), sym_chem_pot = sym.symbols('mu_Xi_-'),\
                    num_mass = 1321.72)


""" Leptons """

# symbolic lepton objects 
electron = lepton(name = 'electron', charge = -1.0, var_type = 'Independent',\
                sym_mass = sym.symbols('m_e'), sym_density = sym.symbols('n_e'), sym_frac = sym.symbols('x_e'),\
                sym_kf = sym.symbols('k_e'), sym_chem_pot = sym.symbols('\mu_e'),\
                num_mass = 0.510)

muon = lepton(name = 'muon', charge = -1.0, var_type = 'Dependent',\
            sym_mass = sym.symbols('m_mu'), sym_density = sym.symbols('n_mu'), sym_frac = sym.symbols('x_\mu'),\
            sym_kf = sym.symbols('k_mu'), sym_chem_pot =  sym.symbols('\mu_mu'),\
            num_mass = 105.65) 




""" Mesons """

# symbolic meson objects 
sigma = meson(name = 'sigma', sym_mass = sym.symbols('m_sigma'), sym_field = sym.symbols('sigma'), num_mass = 550.0)
omega = meson(name = 'omega', sym_mass = sym.symbols('m_omega'), sym_field = sym.symbols('omega'), num_mass = 783.0)
rho = meson(name = 'rho', sym_mass = sym.symbols('m_rho'), sym_field = sym.symbols('rho'), num_mass = 770.0)
phi = meson(name = 'phi', sym_mass = sym.symbols('m_phi'), sym_field = sym.symbols('phi'), num_mass = 1020.0)



#  Load in baryon objects their coupling constants
def baryon_coupling(baryon, eos):
    """ Takes coupling constants from eos object and stores in baryon object """
    
    if (baryon.kind == 'Nucleon'):
        baryon.sym_g_sigma = g_sigma_N_sym
        baryon.sym_g_omega = g_omega_N_sym
        baryon.sym_g_rho = g_rho_N_sym
        baryon.sym_g_phi = g_phi_N_sym

        baryon.num_g_sigma = eos.g_sigma_N
        baryon.num_g_omega = eos.g_omega_N
        baryon.num_g_rho = eos.g_rho_N
        baryon.num_g_phi = eos.g_phi_N

    elif (baryon.kind == 'Hyperon'):
        baryon.sym_g_sigma = g_sigma_H_sym
        baryon.sym_g_omega = g_omega_H_sym
        baryon.sym_g_rho = g_rho_H_sym
        baryon.sym_g_phi = g_phi_H_sym

        # baryon.num_g_sigma = eos.g_sigma_H
        # baryon.num_g_omega = eos.g_omega_H
        # baryon.num_g_rho = eos.g_rho_H
        # baryon.num_g_phi = eos.g_phi_H
    
    elif (baryon.name.split()[0] == 'Lambda'):
        baryon.num_g_sigma = eos.g_sigma_l
        baryon.num_g_omega = eos.g_omega_l
        baryon.num_g_rho = eos.g_rho_l
        baryon.num_g_phi = eos.g_phi_l
    
    elif (baryon.name.split()[0] == 'Sigma'):
        baryon.num_g_sigma = eos.g_sigma_sig
        baryon.num_g_omega = eos.g_omega_sig
        baryon.num_g_rho = eos.g_rho_sig
        baryon.num_g_phi = eos.g_phi_sig
    
    elif (baryon.name.split()[0] == 'Xi'):
        baryon.num_g_sigma = eos.g_sigma_xi
        baryon.num_g_omega = eos.g_omega_xi
        baryon.num_g_rho = eos.g_rho_xi
        baryon.num_g_phi = eos.g_phi_xi
    
    


def sigma_coupling(eos):
    """ Take coupling constant from eos object and store in sigma meson object """
    sigma.num_b = eos.b
    sigma.num_c = eos.c 


def meson_coupling(meson, eos):
    """ Take coupling constants from eos object and store in meson objects """
    if (meson == sigma):
        meson.g_N = eos.g_sigma_N
        meson.g_H = eos.g_sigma_H
    elif (meson == omega):
        meson.g_N = eos.g_omega_N
        meson.g_H = eos.g_omega_H
    elif (meson == rho):
        meson.g_N = eos.g_rho_N
        meson.g_H = eos.g_rho_H
    elif (meson == phi):
        meson.g_N = eos.g_phi_N
        meson.g_H = eos.g_phi_H

# write an initialization function 

def init(eos, baryon_list, meson_list, lepton_list):
    """ Takes eos object and loads in all coupling constants into mesons, baryons """

    # baryons 
    for baryon in baryon_list:
        # load coupling constants into baryon objects
        baryon_coupling(baryon, eos)

        # resets fermi momenta values
        baryon.num_kf = 0.0 

    # load in sigma self coupling 
    sigma_coupling(eos)

    # mesons
    for meson in meson_list:
        # meson coupling 
        meson_coupling(meson, eos)
        # reset meson field values
        meson.num_field = 0.0 

    # ordering meson list
    meson_list.sort(key = mass)


    # lepton list stuff 
    for lepton in lepton_list:
        # reset fermi momenta 
        lepton.num_kf = 0.0



""" When initializing a system, would just need to
1. Initialize the independent variables. 
2. Declare the baryon,lepton, meson, and independent variable lists 
"""


""" Need to then make the functions to generate the equations of motion """

# First: generating the sigma equation of motion 

def scalar_density(baryon):
    """ returns scalar density n_s for a given baryon
        note: we modify the argument of the natural log as to be positive definite to avoid complex numbers 
        this is a symbolic expression (sympy)
    """ 
    coeff_1 = (2*baryon.spin + 1)/(2*Pi**2)
    coeff_2 = baryon.sym_mass_eff
    term_2 = baryon.sym_ef*baryon.sym_kf 
    term_2_2 = sym.sqrt(((baryon.sym_ef + baryon.sym_kf)/baryon.sym_mass_eff)**2)
    term_3 = sym.log(term_2_2)
    
    return coeff_1*coeff_2*(term_2 - baryon.sym_mass_eff**2*term_3)/2


def sigma_eom_init(baryon_list):
    """ returns symbolic sigma equation of motion (sympy) """ 
    
    term_1 = sigma.sym_mass**2*sigma.sym_field 
    term_2 = sigma.sym_b * Neutron.sym_mass * Neutron.sym_g_sigma**3 * sigma.sym_field**2 
    term_3 = sigma.sym_c * Neutron.sym_g_sigma**4 * sigma.sym_field**3
    
    tot = 0
    
    for baryon in baryon_list:
        tot += baryon.sym_g_sigma * scalar_density(baryon)

    return term_1 + term_2 + term_3 - tot

def sigma_eom(baryon_list):
    """ takes symbolic sigma eom expression, substutite in for the symbolic 
        effective mass and effective energy to get a function in terms of 
        fermi momenta and sigma field 
    """ 
    result = sigma_eom_init(baryon_list)

    for baryon in baryon_list:
        result = result.subs(baryon.sym_ef, sym.sqrt(baryon.sym_kf**2 + baryon.sym_mass_eff**2))
        result = result.subs(baryon.sym_mass_eff, baryon.sym_mass - baryon.sym_g_sigma * sigma.sym_field)
    
    return result 

# omega meson EOM

def omega_eom(baryon_list):
    # returns expression for omega equation of motion in terms of baryon fermi momentum
    # note this expression is equal to zero 
    result = 0 
    for baryon in baryon_list:
        result += baryon.sym_g_omega * baryon.sym_kf**3 
    return omega.sym_mass**2 * omega.sym_field * (3*Pi**2) - result 


# rho meson EOM

def rho_eom(baryon_list):
    """ generate symbolic rho equation of motion """
    result = 0 
    for baryon in baryon_list:
        result += baryon.sym_g_rho * baryon.sym_kf**3 * baryon.isospin
    return rho.sym_mass**2 * rho.sym_field * (3*Pi**2) - result 



# phi meson EOM 

def phi_eom(baryon_list):
    # returns eom for phi meson in terms of baryon momenta rather than
    # number density

    result = 0
    for baryon in baryon_list:
        result += baryon.sym_g_phi * baryon.sym_kf**3 
    return phi.sym_mass**2 * (3*Pi**2) * phi.sym_field - result 



# beta equilibrium 

#def baryon_chem_pot(baryon):
    # returns baryon chemical potential in terms of expanded effective mass, 
    # ie, in terms of m - gsigma*sigma 
#    expr = sym.sqrt(baryon.sym_kf**2 + baryon.sym_mass_eff**2) + baryon.sym_g_omega*omega.sym_field\
#            + baryon.sym_g_rho*rho.sym_field*baryon.isospin + baryon.sym_g_phi*phi.sym_field 
#    return expr.subs(baryon.sym_mass_eff, baryon.sym_mass - baryon.sym_g_sigma * sigma.sym_field)

def baryon_chem_pot(baryon, meson_list):
    expr = sym.sqrt(baryon.sym_kf**2 + baryon.sym_mass_eff**2)
    for meson in meson_list:
        if (meson == omega):
            expr += baryon.sym_g_omega*omega.sym_field 
        elif (meson == rho):
            expr += baryon.sym_g_rho * rho.sym_field * baryon.isospin
        elif (meson == phi):
            expr += baryon.sym_g_phi * phi.sym_field 
    
    return expr.subs(baryon.sym_mass_eff, baryon.sym_mass - baryon.sym_g_sigma * sigma.sym_field) 


def neutron_chem_pot_num(fermi, sigma_field = 0.0, omega_field = 0.0, rho_field = 0.0, phi_field = 0.0):
    # returns numerical chem pot for neutron
    return np.sqrt(fermi**2 + (Neutron.num_mass - Neutron.num_g_sigma*sigma_field)**2)\
            + Neutron.num_g_omega * omega_field + Neutron.num_g_phi * phi_field\
            + Neutron.num_g_rho * Neutron.isospin * rho_field


def lepton_chem_pot(lepton):
    """
    returns chemical potential in terms of momenta symbol 
    and mass symbol
    """
    return sym.sqrt(lepton.sym_kf**2 + lepton.sym_mass**2)


def electron_chem_pot_num(fermi):
    """ return numerical chemical potential for electron """ 
    return np.sqrt(fermi**2 + electron.num_mass**2)


def beta_equilibrium(baryon_list, meson_list):
    """ generates a list of all beta equilibrium conditions 
        for baryons 
    """
    neutron_chem_pot = baryon_chem_pot(Neutron, meson_list)
    electron_chem_pot = lepton_chem_pot(electron)

    equation_array =  [] 
    for baryon in baryon_list:
        if (baryon != Neutron):
            equation_array.append(baryon_chem_pot(baryon, meson_list) - neutron_chem_pot + baryon.charge*electron_chem_pot)
                
    return equation_array


def beta_equilibrium_lep(lepton_list):
    """ beta equilibrium condition for leptons """

    result = lepton_chem_pot(electron)
    for lepton in lepton_list:
        if (lepton != electron):
            result -= lepton_chem_pot(lepton)
    return result 


def charge_conservation(baryon_list, lepton_list):
    """ gives charge neutrality equation condition on fermi momentum """ 

    particles = baryon_list + lepton_list 
    expression = 0 

    for particle in particles:
        if (particle.charge > 0):
            expression += particle.sym_kf**3
        elif (particle.charge < 0):
            expression -= particle.sym_kf**3
    
    return expression 


# baryon number conservation

def baryon_num_conservation(baryon_list):
    # gives baryon number conservation equation condition on fermi momentum
    # of individual baryons rather than the individual species' number density 
    result = 0 
    for baryon in baryon_list:
        result += baryon.sym_kf**3
    return 3*Pi**2*sym.symbols('n_B')  - result 


""" System of Equations Generator 
        - Takes the above and generates our system of equations 
"""
def sys_eqn_gen(baryon_list, meson_list, lepton_list):
    # function to generate all our equations and to store in an array
    # called sys_eqn_gen 
    meson_eqn = [] 
    for meson in meson_list:
        if (meson.name == 'sigma'):
            meson_eqn.append(sigma_eom)
        elif (meson.name == 'omega'):
            meson_eqn.append(omega_eom) 
        elif (meson.name == 'rho'):
            meson_eqn.append(rho_eom)
        elif (meson.name == 'phi'):
            meson_eqn.append(phi_eom) 
    
    other_func_gen = [beta_equilibrium, charge_conservation, baryon_num_conservation]
    
    if(len(lepton_list) != 1):
        func_gen = meson_eqn + other_func_gen + [beta_equilibrium_lep]
    else:
        func_gen = meson_eqn + other_func_gen

    sys_eqn = []
    
    for function in func_gen:
        if (function == charge_conservation):
            # since charge conservation depends on both baryons and leptons, we need to 
            # pass to it both baryon and lepton lists 
            sys_eqn.append(function(baryon_list, lepton_list))
        elif (function == beta_equilibrium):
            # beta condition function returns an array with (possibly) multiple equations
            # we unload those functions and append to array 
            beta_conditions = function(baryon_list, meson_list)
            for equation in beta_conditions:
                sys_eqn.append(equation)
        elif (function == beta_equilibrium_lep):
            sys_eqn.append(function(lepton_list))
        else:
            sys_eqn.append(function(baryon_list))
            
    return sys_eqn


""" Substitution Function 
    - Up to this point all of our expressions have been symbolic which makes it easy
    - to verify their legitimateness. Now we want to perform numerical calculations
    - so we want to substitute in for the symbolic expressions for the masses and stuff 
"""
#arg_list = [baryon_list, baryon_list, meson_sym_list, meson_num_list, lepton_sym_list, lepton_num_list]

def substitution(equation, nB, baryon_list, meson_list, lepton_list):

    # loops through baryons, mesons, leptons to replace masses and stuff with numeric values

    # baryons 
    for i in range(len(baryon_list)):
        equation = equation.subs([(baryon_list[i].sym_mass, baryon_list[i].num_mass),\
                                 (sigma.sym_b, sigma.num_b), (sigma.sym_c, sigma.num_c),\
                                 (baryon_list[i].sym_g_sigma, baryon_list[i].num_g_sigma),\
                                 (baryon_list[i].sym_g_omega, baryon_list[i].num_g_omega),\
                                 (baryon_list[i].sym_g_rho, baryon_list[i].num_g_rho),\
                                 (baryon_list[i].sym_g_phi, baryon_list[i].num_g_phi)])
    
    # mesons 
    for i in range(len(meson_list)):
        equation = equation.subs(meson_list[i].sym_mass, meson_list[i].num_mass)
    
    # leptons 
    for i in range(len(lepton_list)):
        equation = equation.subs(lepton_list[i].sym_mass, lepton_list[i].num_mass)
    
    # substitute in Pi for actual value of pi, and baryon density
    equation = equation.subs([(Pi, np.pi), (sym.symbols('n_B'), nB)])


    return equation 


def subs(sys_eqn, nB, baryon_list, meson_list, lepton_list):
    # performs substitution on entire set of equations 
    for i in range(len(sys_eqn)):
        sys_eqn[i] = substitution(sys_eqn[i], nB, baryon_list, meson_list, lepton_list)
    return sys_eqn 


""" Fraction finder """

def fraction(fermi, nB):
    # after having solved for fermi momentum, can 
    # get the corresponding particle fraction 
    return fermi**3/3/np.pi**2/nB



""" Would be good to then include things here that do the solving
    for us 
"""

def potential_baryon_gen(baryon_list):
    """ takes a list of baryons and returns a list of baryons
        with the proton and neutron removed and ordered in terms of mass
    """
    pot_list = []
    for baryon in baryon_list:
        if (baryon != Proton and baryon != Neutron):
            pot_list.append(baryon)
    pot_list.sort(key = mass)
    pot_list.append('None')
    return pot_list


def potential_lepton_gen(lepton_list):
    pot_list = [] 
    for lepton in lepton_list:
        if (lepton != electron):
            pot_list.append(lepton)
    pot_list.append('None')
    return pot_list


def neutron_chem_pot_num_test(fermi, meson_list):
    # gets numerical values for neutron chemical potential 
    bare_chem = np.sqrt(fermi**2 + (Neutron.num_mass - Neutron.num_g_sigma*sigma.num_field)**2)
    result = 0
    for meson in meson_list:
        if (meson == rho):
            result += Neutron.isospin*Neutron.num_g_rho*rho.num_field
        elif (meson == omega):
            result += Neutron.num_g_omega * omega.num_field
        elif (meson == phi):
            result += Neutron.num_g_phi * phi.num_field 
    return bare_chem + result 


def electron_chem_pot_num(fermi):
    # return numerical chemical potential for electron
    return np.sqrt(fermi**2 + electron.num_mass**2)


def bare_chemical_potential(baryon, meson_list):
    """finds bare chemical potential for a baryon given a list of mesons
    assumes meson objects are filled with field values
    this code is ugly and is not easily generalizable... """
    bare_chem = 0.0 
    for meson in meson_list:
        if (meson == sigma):
            bare_chem += baryon.num_mass - baryon.num_g_sigma * sigma.num_field 
        elif (meson == omega):
            bare_chem += baryon.num_g_omega * omega.num_field
        elif (meson == rho):
            bare_chem += baryon.num_g_rho * baryon.isospin * rho.num_field 
        elif (meson == phi):
            bare_chem += baryon.num_g_phi * phi.num_field 
    return bare_chem 


def baryon_threshold(baryon, meson_list):
    """ checks to see if combination of neutron and electron chemical potential """ 
    
    neutron_chem_pot = neutron_chem_pot_num_test(Neutron.num_kf, meson_list) 
    electron_chem_pot = electron_chem_pot_num(electron.num_kf)
    
    if (neutron_chem_pot - baryon.charge * electron_chem_pot >= bare_chemical_potential(baryon, meson_list)):
        return True
    else:
        return False 
    

def lepton_threshold(lepton):
    """ 
    checks to see if electron chemical potential is large enough
    to support entrance of other leptons, namely, muon
    """ 
    if (electron.num_kf >= lepton.num_mass):
        return True
    else:
        return False 


def column_name(baryon_list, meson_list, lepton_list):
    """
    generate column names
    this is used to create the Dataframe in which we store our values
    Data frame is well suited to Jupyter visualization which is great! 
    can convert/extract to numpy array too I suppose 
    """

    columns = ['nB/n0']
    
    for meson in meson_list:
        columns.append(meson.name + " " + 'field (MeV)')
    
    columns.append(Neutron.name + " " + 'kF (MeV)')
    columns.append(Proton.name + " " + 'kF (MeV)')
    columns.append(electron.name + " " + 'kF (MeV)')

    for lepton in lepton_list:
        if (lepton != electron):
            columns.append(lepton.name + " " + "kF (MeV)")
    
    for baryon in baryon_list:
        if (baryon != Proton and baryon != Neutron):
            columns.append(baryon.name + " " + "kF (MeV)")
    
    columns.append(Neutron.name + " " + 'frac')
    columns.append(Proton.name + " " + 'frac')
    columns.append(electron.name + " " + 'frac')

    for lepton in lepton_list:
        if (lepton != electron):
            columns.append(lepton.name + " " + "frac")
    
    for baryon in baryon_list:
        if (baryon != Proton and baryon != Neutron):
            columns.append(baryon.name + " " + "frac")
    
    
    return columns


def reserve_baryons(baryon_list):
    """ returns list of baryons that aren't protons or neutrons """
    reserve_list = []
    for baryon in baryon_list:
        if (baryon != Proton and baryon != Neutron):
            reserve_list.append(baryon)
    return reserve_list 


def reserve_leptons(lepton_list):
    """ return list of leptons that aren't electrons """
    reserve_list = []
    for lepton in lepton_list:
        if (lepton != electron):
            reserve_list.append(lepton)
    return reserve_list 


def mass(particle):
    return particle.num_mass


def ind_variable(baryon_lists, lepton_lists, meson_lists):
    """ get the independent variables in our system """
    
    ind_vars = []
    
    for meson in meson_lists:
        ind_vars.append(meson.sym_field)
    
    ind_vars.append(Neutron.sym_kf)
    ind_vars.append(Proton.sym_kf) 
    ind_vars.append(electron.sym_kf)
    
    # this is done to preserve order we have npe first 
    # and then as muons, and hyperons enter, we want them to be 
    # later in the list so our values are consistent when we
    # write them out to an array
    
    for lepton in lepton_lists:
        if (lepton != electron):
            ind_vars.append(lepton.sym_kf)
    
    
    for baryon in baryon_lists:
        if (baryon != Proton and baryon != Neutron):
            ind_vars.append(baryon.sym_kf)
        
    
    return ind_vars 


def frac(fermi, nb):
    """ calculates fraction given fermi momentum and nB """ 
    return fermi**3 /3/np.pi**2/nb


def full_solve(eos, baryon_list, lepton_list, meson_list, npe_guess, csv_name = 'data', solver_method = 'broyden'):
    """ full solver 
        from an eos, baryon list, lepton list, meson list,
        calculate and return a DataFrame of values of meson fields, 
        particle fractions/momenta, etc as a function of baryon density 
    """
    
    # initialize all things
    init(eos, baryon_list, meson_list, lepton_list)
    
    # create nB array 
    nB = np.arange(0.27, 8.0, 0.01) #nB/n0
    nB_mev = nB*eos.n0*hc**3 #nB_mev in mev+3 
    
    # create data array (pre-allocate)
    row_size = len(nB)
    column_size = len(meson_list) + 2*len(baryon_list + lepton_list) + 1 
    data = np.zeros((row_size, column_size), dtype = 'float') 
    data[:,0] = nB 
    
    # create first system: NPE
    current_baryons = [Neutron, Proton]
    current_leptons = [electron] 
    
    # create lists for potential particles 
    potential_baryons = potential_baryon_gen(baryon_list)
    potential_leptons = potential_lepton_gen(lepton_list) 
    

    # independent variables
    ind_vars = ind_variable(current_baryons, current_leptons, meson_list)
    
    # initial guess for NPE matter 
    x_guess = npe_guess

    # check if correct number of arguments 
    if (len(x_guess) != len(ind_vars)):
        print("Number of parameters in initial guess doesn't\
             match the number of independent variables. Check number of mesons.")
        quit() 
    
    # iterate through baryon density nB 
    for i in range(len(nB_mev)): 
        
        # update our system: 
        
        # old baryon update
        # if (potential_baryons[0] != 'None'):
        #     Bool = baryon_threshold(potential_baryons[0], meson_list) 
        #     if (Bool):
        #         current_baryons.append(potential_baryons[0])
        #         potential_baryons.remove(potential_baryons[0])
        #         x_guess = np.append(x_guess, 27.0)

        ## 04/10/2022 - testing new baryon update that considers possibility of other baryons
        ## not sorted by mass appearing which is possible due to charge affecting the threshold equation
        ## had an issue where some hyperons never appeared and thought it was because 
        # if (potential_baryons[0] != 'None'):
        #     bool_list = np.zeros(len(potential_baryons), dtype = 'bool')
        #     for i in range(len(bool_list)-1):
        #         # iterate through potential baryon list and check if we can add new baryons
        #         Bool = baryon_threshold(potential_baryons[i], meson_list) 
        #         if Bool: 
        #             current_baryons.append(potential_baryons[i])
        #             x_guess = np.append(x_guess, 40.0)
        #         bool_list[i] = Bool 
        #     for i in range(len(bool_list)):
        #         if bool_list[i] == True:
        #             potential_baryons.remove(potential_baryons[i])

        ## 04/10/2022 - testing new baryon update version 2
        ## issue might've been iter variable choice 
        if (potential_baryons[0] != 'None'):
            for shikimori in range(len(potential_baryons) - 1):
                Bools = baryon_threshold(potential_baryons[shikimori], meson_list)
                if (Bools):
                    current_baryons.append(potential_baryons[shikimori])
                    potential_baryons.remove(potential_baryons[shikimori])
                    x_guess = np.append(x_guess, 43.0)
                    break 


        
        if (potential_leptons[0] != 'None'):
            Bool = lepton_threshold(potential_leptons[0])
            if (Bool):
                current_leptons.append(potential_leptons[0])
                potential_leptons.remove(potential_leptons[0])
                x_guess = np.append(x_guess, 2.0)
        
        # 
        #init(eos, baryon_list, meson_list, lepton_list)
            
        # generate system of equations 
        sys_eqn = sys_eqn_gen(current_baryons, meson_list, current_leptons)
        subs(sys_eqn, nB_mev[i], current_baryons, meson_list, current_leptons)
        
        # generate independent variables 
        ind_vars = ind_variable(current_baryons, current_leptons, meson_list)
        
        # call solver, broyden returns a vector with solutions to independent variables
        # very importantly, the independent variables are ordered: sigma, omega, rho, phi
        if (solver_method == 'broyden'):
            answer = broyden(sys_eqn, ind_vars, x_guess)
        elif (solver_method == 'newton'):
            answer = Newton(sys_eqn, ind_vars, x_guess)
        
        # append values to data file 
        #for j in range(column_size - len(current_baryons + current_leptons) - 1):
        #    data[i][j+1] = answer[j]
        for j in range(len(ind_vars)):
            data[i][j+1] = answer[j]
        
        # update important values
        pseudo_dict = [ind_vars, answer]
        for k in range(len(ind_vars)):
            if (pseudo_dict[0][k] == sigma.sym_field):
                sigma.num_field = answer[k]
            elif (pseudo_dict[0][k] == omega.sym_field):
                omega.num_field = answer[k]
            elif (pseudo_dict[0][k] == rho.sym_field):
                rho.num_field = answer[k]
            elif (pseudo_dict[0][k] == phi.sym_field):
                phi.num_field = answer[k]
            elif (pseudo_dict[0][k] == Neutron.sym_kf):
                Neutron.num_kf = answer[k]
            elif (pseudo_dict[0][k] == electron.sym_kf):
                electron.num_kf = answer[k]
        
        # update x_guess
        x_guess = answer 
        
    # fill in fractions 
    for rem in range(len(nB)):
        for emilia in range(len(ind_vars) - len(meson_list)):
            data[rem][emilia + len(meson_list + baryon_list + lepton_list) + 1]\
                = frac(data[rem][emilia + len(meson_list) + 1], data[rem][0]*eos.n0*hc**3)
         
    # convert data array to dataframe 
    data_frame = pd.DataFrame(data, columns = column_name(baryon_list, meson_list, lepton_list))
    
    
    # write out dataframe to csv file 
    data_frame.to_csv(csv_name + '.csv', float_format = '{:.8f}'.format)
    
    return data_frame 