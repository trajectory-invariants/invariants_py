# -*- coding: utf-8 -*-
"""
Created on Fri Apr 6 2024

@author: Riccardo
"""

def check_solver(fatrop_solver):
    try: # check if fatropy is installed, otherwise use ipopt
        import fatropy
        pass
    except:
        if fatrop_solver:
            print("")
            print("Fatrop solver is not installed! Using ipopt instead")
            fatrop_solver = False
    return fatrop_solver