### This is a project to use home game data to estimate how many points an NBA team would have

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import r2_score
import pickle


df= pd.read_csv("C:\\Users\\Abigail Luwawa\\OneDrive\\Desktop\\CODE\\NBA\\game.csv")

### DATA CLEANING ###

# Checking for missing values
print("#no of rows with missing values: " , df.isnull().sum())
df.dropna(inplace=True) #uncomment if necessary and run again to check

# Checking for duplicates
print("dupes: ", df.duplicated().sum())
#df.drop_duplicates(inplace=True) #uncomment if necessary and run again to check

# Checking for outliers
print(df.describe())

#collecting data for HOME games only
homedf = df[["fgm_home","fga_home", "ftm_home", "fta_home" , "oreb_home" ,	"dreb_home", "reb_home" , "ast_home" ,	"stl_home" , "blk_home" , "tov_home" , "pts_home"]].copy()
print(homedf.describe())
# Data abbreivations: fgm (field goals made), fga (field goals attempted), ftm (free throws made), fta (free throws attempted), oreb (offensive rebounds), dreb (defensive rebounds), assists, steals, blocks, and turnovers
print("Row length: " ,len(homedf)) #row count, making sure I haven't deleted loads of my data


### Data visulation ###
##To see what the data looks like and get a feel for it ##
#sns.pairplot(homedf, diag_kind='hist') #droping team names
#plt.show()




### Creating the test and train elements ###
features = ["fgm_home","fga_home", "ftm_home", "fta_home" , "oreb_home" ,	"dreb_home", "reb_home" , "ast_home" ,	"stl_home" , "blk_home" , "tov_home"]
X = df.loc[:, features]
y = df.loc[:, "pts_home"]


### Scaling the data ###
#The data is scaled as there large difference in between the variables. For example, the max and mean assissts made are far greater than the steals. 
# #If this was not scaled varaibles with larger values such as rebounds and field goal attempts would have a greater weight than ther other variables

scale = mms()  #min-max-scaling
Xscale = scale.fit_transform(X)

##Spilting the data between train and test
X_train, X_test, y_train, y_test = tts(Xscale, y, random_state=0, train_size = .70)

nbamodel = lr() #the NBA model :D

#Fitting the data
nbamodel.fit(X_train, y_train)

#creating the y_pred for predictions
y_pred = nbamodel.predict(X_test)

#Determining the intercept and coefficients
print(f"Intercept: {nbamodel.intercept_}")

nbamodelCoefs = nbamodel.coef_
print(f"Variable Coefficients: {nbamodelCoefs}")

# Evaluating the model using R^2
r2 = r2_score(y_test, y_pred)
print("R^2 Score: {:+.2f}".format(r2)) # to 2.c.p with sign

#save using pickle
pickle.dump(nbamodel, open('nbamodel.pkl', 'wb'))
