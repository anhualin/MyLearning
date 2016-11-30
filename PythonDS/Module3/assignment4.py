import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv('C:/Users/alin/Documents/SelfStudy/PythonDS/DAT210x/Module3/Datasets/wheat.data')



#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
# 
# .. your code here ..
df0 = df.drop(['id', 'area', 'perimeter'], axis = 1)


#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
# .. your code here ..
plt.figure()
parallel_coordinates(df0, 'wheat_type', alpha = 0.4)
plt.show()


plt.figure()
andrews_curves(df0, 'wheat_type')
plt.show()

df1 = df.drop(['id'], axis = 1)
plt.figure()
andrews_curves(df1, 'wheat_type')
plt.show()

