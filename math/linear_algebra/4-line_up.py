#!/usr/bin/env python3

"""A module that does the trick"""


def add_arrays(arr1, arr2):
    """A function that does the trick"""
    if len(arr1) != len(arr2):
        return None
    new_list = []
    for i in range(len(arr1)):
        new_list += [arr1[i]+arr2[i]]
    return new_list
