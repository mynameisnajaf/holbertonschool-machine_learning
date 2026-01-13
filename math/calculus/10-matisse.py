#!/usr/bin/env python3

"""A module that does the trick"""


def poly_derivative(poly):
    """A function that does the trick"""
    if type(poly) is not list:
        return None
    elif not poly:
        return None
    elif len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        new_list = []
        for i in range(1, len(poly)):
            new_list.append(poly[i]*i)
        return new_list
poly = [5, 3, 0, 1]
print(poly_derivative(poly))
