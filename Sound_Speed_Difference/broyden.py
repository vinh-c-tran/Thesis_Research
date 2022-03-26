""" Newton and Broyden Numerical Methods for systems of non-linear equations"""

import numpy as np 
import sympy as sym
import time 


class solution:
    def __init__(self, solution):
        self.solution = solution
        
        self.counter = 0
        self.error = 1.0E6
        self.run_time = 0.0 


class broyden_solution:
    def __init__(self, solution):
        self.solution = solution
        self.F_x_current = 0.0 
        self.A_current_inv = 0.0 
        
        self.counter = 0
        self.error = 1.0E10
        self.run_time = 0.0 


def F_x_num(sys_eqn, ind_vars, vector):
    num_array = np.copy(sys_eqn)
    num_array_final = np.zeros(len(sys_eqn), dtype = 'float')
    
    for i in range(len(sys_eqn)):
        for j in range(len(ind_vars)):
            num_array[i] = num_array[i].subs(ind_vars[j], vector[j])
        num_array_final[i] = sym.N(num_array[i])
    return num_array_final


def J_x(sys_eqn, ind_vars):
    Jacobian = np.zeros((len(sys_eqn), len(ind_vars)), dtype='object')
    
    for i in range(len(sys_eqn)):
        for j in range(len(ind_vars)):
            Jacobian[i,j] = sym.diff(sys_eqn[i], ind_vars[j])
    
    return Jacobian


def J_x_num(sys_eqn, ind_vars, vector):
    # substitute in numerical values
    
    A = J_x(sys_eqn, ind_vars)
    
    for i in range(len(A[0][:])):
        for j in range(len(A[:][0])):
            for k in range(len(ind_vars)):
                A[i][j] = A[i][j].subs(ind_vars[k], vector[k])
                
    return np.array(A, dtype = 'float')


def err(x_1, x_2):
    diff = x_1 - x_2
    return np.sqrt(np.dot(diff,diff))


def Newton(sys_eqn, ind_vars, initial_guess, accept_error = 1.0E-10):
    
    # timer variables 
    start = time.time()
    end = 0.0 
    run_time = 60.0 # seconds 
    
    # declare solution object 
    answer = solution(initial_guess)
    
    while(answer.error > accept_error):
        Jacobian = J_x_num(sys_eqn, ind_vars, answer.solution)
        Fx_min = -F_x_num(sys_eqn, ind_vars, answer.solution)
        y = np.linalg.solve(Jacobian, Fx_min)
        
        prior_solution = answer.solution
        answer.solution = answer.solution + y
        answer.counter += 1 
        answer.error = err(answer.solution, prior_solution) 
        
        #print(answer.error)
        
        end = time.time()
        answer.run_time = end - start 
        
        if (end - start > run_time):
            print("Not converged")
            break 
            
    return answer.solution


def update(A_old_inv, s_new, y_new):
    # takes A_old_inv and returns A_new_inv
    
    prod = np.matmul(s_new, np.matmul(A_old_inv, y_new))
    
    one = s_new - np.matmul(A_old_inv, y_new)
    two = np.matmul(s_new, A_old_inv)
    three = np.outer(one, two)
    
    A_new_inv = A_old_inv + 1/prod*three
    
    return A_new_inv


def broyden(sys_eqn, ind_vars, initial_guess, accept_error = 1.0E-10):
    # timer variables 
    start = time.time()
    end = 0.0 
    run_time = 60.0 # seconds 
    
    # declare solution object 
    answer = broyden_solution(initial_guess)
    
    
    # Broyden First Steps
    # Need to find the Jacobian at the first step 
    
    answer.F_x_current = F_x_num(sys_eqn, ind_vars, answer.solution)
    answer.A_current_inv = np.linalg.pinv(J_x_num(sys_eqn, ind_vars, answer.solution))

    
    
    # while loop that quits upon convergence or time out 
    while( answer.error > accept_error and end - start < run_time):
        
        # Current Values 
        x_current = answer.solution
        F_x_current = answer.F_x_current
        A_current_inv = answer.A_current_inv 
        
        # Find x_new, F_new 
        x_new = x_current - np.matmul(A_current_inv, F_x_current)
        F_x_new = F_x_num(sys_eqn, ind_vars, x_new)
        
        
        # Find y_new, s_new 
        y_new = F_x_new - F_x_current 
        s_new = x_new - x_current 
        
        
        # Find A_new_inv
        
        A_new_inv = update(A_current_inv, s_new, y_new)
        
        # Store current values 
        answer.solution = x_new 
        answer.F_x_current = F_x_new  
        answer.A_current_inv = A_new_inv
        
        answer.counter += 1
        answer.error = err(answer.solution, x_current)
        #print(answer.error)
        
        end = time.time()
    
    return answer.solution
