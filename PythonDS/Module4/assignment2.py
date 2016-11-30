import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import math

from sklearn import preprocessing


#import assignment2_helper as helper
def scaleFeatures(df):
  # SKLearn has many different methods for doing transforming your
  # features by scaling them (this is a type of pre-processing).
  # RobustScaler, Normalizer, MinMaxScaler, MaxAbsScaler, StandardScaler...
  # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
  #
  # However in order to be effective at PCA, there are a few requirements
  # that must be met, and which will drive the selection of your scaler.
  # PCA required your data is standardized -- in other words it's mean is
  # equal to 0, and it has ~unit variance.
  #
  # SKLearn's regular Normalizer doesn't zero out the mean of your data,
  # it only clamps it, so it's inappropriate to use here (depending on
  # your data). MinMaxScaler and MaxAbsScaler both fail to set a unit
  # variance, so you won't be using them either. RobustScaler can work,
  # again depending on your data (watch for outliers). For these reasons
  # we're going to use the StandardScaler. Get familiar with it by visiting
  # these two websites:
  #
  # http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
  #
  # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
  #


  # ---------
  # Feature scaling is the type of transformation that only changes the
  # scale and not number of features, so we'll use the original dataset
  # column names. However we'll keep in mind that the _units_ have been
  # altered:
  scaled = preprocessing.StandardScaler().fit_transform(df)
  scaled = pd.DataFrame(scaled, columns=df.columns)
  #print "New Variances:\n", scaled.var()
  #print "New Describe:\n", scaled.describe()
  return scaled


def drawVectors(transformed_features, components_, columns, plt, scaled):
  if not scaled:
    return plt.axes() # No cheating ;-)

  num_columns = len(columns)

  # This funtion will project your *original* feature (columns)
  # onto your principal component feature-space, so that you can
  # visualize how "important" each one was in the
  # multi-dimensional scaling
  
  # Scale the principal components by the max value in
  # the transformed set belonging to that component
  xvector = components_[0] * max(transformed_features[:,0])
  yvector = components_[1] * max(transformed_features[:,1])

  ## visualize projections

  # Sort each column by it's length. These are your *original*
  # columns, not the principal components.
  important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
  important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
  print "Features by importance:\n", important_features

  ax = plt.axes()

  for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
    plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

  return ax
    
  
# Look pretty...
matplotlib.style.use('ggplot')


# Do * NOT * alter this line, until instructed!
scaleFeatures = True


# TODO: Load up the dataset and remove any and all
# Rows that have a nan. You should be a pro at this
# by now ;-)
#
# .. your code here ..
df = pd.read_csv('C:/Users/alin/Documents/SelfStudy/PythonDS/DAT210x/Module4/Datasets/kidney_disease.csv')
df = df.dropna(axis = 0)



# Create some color coded labels; the actual label feature
# will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i=='ckd' else 'green' for i in df.classification]


# TODO: Use an indexer to select only the following columns:
#       ['bgr','wc','rc']
#
# .. your code here ..
df = df[['bgr', 'rc', 'wc']]
df.rc = pd.to_numeric(df.rc, errors = 'coerce')
df.wc = pd.to_numeric(df.wc, errors = 'coerce')


# TODO: Print out and check your dataframe's dtypes. You'll probably
# want to call 'exit()' after you print it out so you can stop the
# program's execution.
#
# You can either take a look at the dataset webpage in the attribute info
# section: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
# or you can actually peek through the dataframe by printing a few rows.
# What kind of data type should these three columns be? If Pandas didn't
# properly detect and convert them to that data type for you, then use
# an appropriate command to coerce these features into the right type.
#
# .. your code here ..
df.rc = pd.to_numeric(df.rc, errors = 'coerce')
df.wc = pd.to_numeric(df.wc, errors = 'coerce')
df.dtypes


# TODO: PCA Operates based on variance. The variable with the greatest
# variance will dominate. Go ahead and peek into your data using a
# command that will check the variance of every feature in your dataset.
# Print out the results. Also print out the results of running .describe
# on your dataset.
#
# Hint: If you don't see all three variables: 'bgr','wc' and 'rc', then
# you probably didn't complete the previous step properly.
#
# .. your code here ..
df.var(axis = 0)


# TODO: This method assumes your dataframe is called df. If it isn't,
# make the appropriate changes. Don't alter the code in scaleFeatures()
# just yet though!
#
# .. your code adjustment here ..
if scaleFeatures: df = scaleFeatures(df)



# TODO: Run PCA on your dataset and reduce it to 2 components
# Ensure your PCA instance is saved in a variable called 'pca',
# and that the results of your transformation are saved in 'T'.
#
# .. your code here ..
 from sklearn.decomposition import PCA
 pca = PCA(n_components=2)
 pca.fit(df)
 T = pca.transform(df)

# Plot the transformed data as a scatter plot. Recall that transforming
# the data will result in a NumPy NDArray. You can either use MatPlotLib
# to graph it directly, or you can convert it to DataFrame and have pandas
# do it for you.
#
# Since we've already demonstrated how to plot directly with MatPlotLib in
# Module4/assignment1.py, this time we'll convert to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
ax = drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


