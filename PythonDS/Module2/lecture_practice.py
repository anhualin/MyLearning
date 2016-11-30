import pandas as pd
df = pd.read_csv('Datasets/direct_marketing.csv')
df[0:10]

ordered_satisfaction = ['Very Unhappy', 'Unhappy', 'Neutral', 'Happy', 'Very Happy']
df = pd.DataFrame({'satisfaction':['Mad', 'Happy', 'Unhappy', 'Neutral']})
df.satisfaction = df.satisfaction.astype("category",
  ordered=True,
  categories=ordered_satisfaction
).cat.codes
# TODO: Load up the 'tutorial.csv' dataset
#
# .. your code here ..



# TODO: Print the results of the .describe() method
#
# .. your code here ..



# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
# .. your code here ..

