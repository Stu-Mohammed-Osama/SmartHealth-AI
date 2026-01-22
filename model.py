from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def desmoke(x):
   if x == "yes":
     x = 1
   else:
      x = 0
   return x
thetable = pd.read_csv("insurance.csv")
newsmoke = thetable["smoker"].map(desmoke)
thetable.drop(["region" , "smoker"] , axis=1 , inplace=True)
newtable = pd.concat([thetable , newsmoke] , axis=1)
X = newtable.drop(["charges"] , axis=1)
y = newtable.charges

X_train , X_valid , y_train , y_valid = train_test_split(X , y , train_size=0.7 , test_size=0.3 , random_state=0)

OH_encode = OneHotEncoder(handle_unknown="ignore" , sparse_output=False)

OH_X_train = pd.DataFrame(OH_encode.fit_transform(X_train[["sex"]]))
OH_X_valid = pd.DataFrame(OH_encode.transform(X_valid[["sex"]]))

OH_X_train.index = X_train.index
OH_X_valid.index = X_valid.index

num_X_train = X_train.drop(["sex"] , axis = 1)
num_X_valid = X_valid.drop(["sex"] , axis = 1)

final_X_train = pd.concat([num_X_train , OH_X_train] , axis=1)
final_X_valid = pd.concat([num_X_valid , OH_X_valid] , axis=1)

final_X_train.columns = final_X_train.columns.astype(str)
final_X_valid.columns = final_X_valid.columns.astype(str)

mymodel = RandomForestRegressor(n_estimators= 200 , random_state=42)
X_full = pd.concat([final_X_train , final_X_valid] , axis=0)
y_full = pd.concat([y_train , y_valid] , axis=0)
mymodel.fit(X_full, y_full)

joblib.dump(mymodel, 'insurance_model.pkl')
# Save the encoder (so you can transform new user input)
joblib.dump(OH_encode, 'gender_encoder.pkl')

print("Files saved successfully!")