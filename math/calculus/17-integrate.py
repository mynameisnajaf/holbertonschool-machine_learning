#!/usr/bin/env python3

"""A module that does the trick"""


def poly_integral(poly, C=0):
    """A function that does the trick"""
    if type(poly) is not list:
        return None
    elif not poly:
        return None
    elif len(poly) == 0:
        return None
    elif type(C) is not int:
        return None
    elif poly == [0]:
        return [C]
    else:
        integral_list = []
        integral_list.append(C)
        for i in range(len(poly)):
            integ = poly[i]/(i+1)
            integral_list.append(int(integ) if integ.is_integer() else integ)
        return integral_list
