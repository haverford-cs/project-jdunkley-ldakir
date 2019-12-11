"""
Authors: Jocelyn Dunkley and Lamiaa Dakir
Date:
Description:
"""

import numpy as np
from util import *

def main() :
    #load data

    train_data, test_data = load_data('spambase/spambase.data')
    print("train data", train_data)


if __name__ == "__main__" :
    main()
