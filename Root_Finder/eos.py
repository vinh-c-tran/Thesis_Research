"""
    Header file (?) 
"""

from solver_helper import eos

gm3 = eos(n0 = 0.153,\
        g_sigma_N = 8.784820, g_omega_N = 8.720086, g_rho_N = 8.544795, g_phi_N = 0.0, b = 0.008628, c = -0.002433,\
        g_sigma_H = 5.408849, g_omega_H = 5.813391, g_rho_H = 0.0, g_phi_H = -4.110688,\
        g_sigma_l = 5.408849, g_omega_l = 5.813391, g_rho_l = 0.0, g_phi_l = -4.110688,\
        g_sigma_sig = 5.408849, g_omega_sig = 5.813391, g_rho_sig = 0.0, g_phi_sig= -4.110688,\
        g_sigma_xi = 5.408849, g_omega_xi = 5.813391, g_rho_xi = 0.0, g_phi_xi = -4.110688)


