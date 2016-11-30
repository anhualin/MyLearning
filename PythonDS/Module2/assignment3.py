import pandas as pd
import numpy as np
# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
# .. your code here ..
df = pd.read_csv('C:/Users/alin/Documents/SelfStudy/PythonDS/DAT210x/Module2/Datasets/servo.data', header = None, 
                 names = ['motor', 'screw', 'pgain', 'vgain', 'class'])

# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
# .. your code here ..
df0 = df[df.vgain == 5]
print df0.shape[0]

# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
# .. your code here ..
df1 = df[(df.motor == 'E') & (df.screw == 'E')]
print df1.shape[0]

# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
# .. your code here ..
df2 = df[df.pgain == 4]
df2.describe()
np.mean(df2.vgain)
# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!
df2.dtypes



