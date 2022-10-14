from preprocess import public_data, test_train_split
import numpy as np

dat = public_data([2])


print(test_train_split(dat))

