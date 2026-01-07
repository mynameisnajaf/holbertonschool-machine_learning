#!/usr/bin/env python3
"""A module to create a dataframe from dictionary"""

import pandas as pd


data = {
    "First": [0.0, 0.5, 1, 1.5],
    "Second": ["one", "two", "three", "four"]
}
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
