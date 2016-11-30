import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []
#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
folder = 'C:/Users/alin/Documents/SelfStudy/PythonDS/DAT210x/Module4/Datasets/ALOI/32/'

for i in range(0, 360, 5):
    image = folder + '32_r'+str(i) + '.png'
    img = misc.imread(image)
    samples.append(img.flatten())   
color1 = ['b' for i in range(0, 72)]
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
folder = 'C:/Users/alin/Documents/SelfStudy/PythonDS/DAT210x/Module4/Datasets/ALOI/32i/'

for i in range(110, 230, 10):
    image = folder + '32_i'+str(i) + '.png'
    img = misc.imread(image)
    samples.append(img.flatten())   
color2 = ['r' for i in range(110, 230, 10)]

color = color1 + color2
#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 
df = pd.DataFrame.from_records(samples)

#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
from sklearn.manifold import Isomap
imap = Isomap(n_neighbors=6, n_components=3)
imap.fit(df)
T = imap.transform(df)
TD = pd.DataFrame(T)
#


#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
TD.plot.scatter(x=0, y = 1, c = color)



#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('0')
ax.set_ylabel('1')
ax.set_zlabel('2')

ax.scatter(TD[0], TD[1], TD[2], c=color, marker='.')


plt.show()

