# -*- coding: utf-8 -*-
"""
Created on Fri Apr 6 2024

@author: Riccardo
"""

def check_solver(use_fatrop_solver):
    try: # check if fatropy is installed, otherwise use ipopt
        import fatropy
        use_fatrop_solver = True
    except:
        if use_fatrop_solver:
            print("")
            print("Fatrop solver is not installed! Using ipopt instead")
            use_fatrop_solver = False
    return use_fatrop_solver