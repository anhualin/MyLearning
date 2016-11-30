import pandas as pd
import html5lib

# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2',
                              match = 'TEAM', header = 0, skiprows = 1)[0]
df = df.rename(columns = {'G.1': 'PP_G', 'A.1': 'PP_A', 'G.2': 'SH_G', 'A.2': 'SH_A'})

# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
# .. your code here ..


# TODO: Get rid of any row that has at least 4 NANs in it
#
# .. your code here ..
df =df.dropna(axis = 0, thresh = 4)
df = df[df.RK != 'RK']
# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..


# TODO: Get rid of the 'RK' column
#
# .. your code here ..
df = df.drop(labels = ['RK'], axis = 1)
df= df.reset_index(drop = True)
# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..



# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
df.dtypes
for lab in df.columns:
    if lab != 'PLAYER' and lab != 'TEAM':
        df[lab] = pd.to_numeric(df[lab], errors = 'coerce')

# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

