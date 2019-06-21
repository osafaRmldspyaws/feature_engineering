# importing the required libraries
import featuretools as ft
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

#import the datasets
train = pd.read_csv('Train_UWu5bXk.csv')
test = pd.read_csv('Test_u94Q5KV.csv')

print('The training data has {} columns and {} rows '.format(train.shape[1], train.shape[0]))
print('The testing data has {} columns and {} rows'.format(test.shape[1], test.shape[0]))
print(train.head())
print(test.head())
print(train.columns)

# data preparation before calling the any classifier
# saving identifiers
test_item_identifiers = test['Item_Identifier']
test_outlet_identifier = test['Outlet_Identifier']
sales = train['Item_Outlet_Sales']
train = train.drop('Item_Outlet_Sales',axis=1)

# combine the test and the train set so that we reduce on performing the same steps twice
combi = train.append(test, ignore_index=True)

#check the missing values
print(combi.isnull().sum())

#imputing missing data
combi['Item_Weight'].fillna(combi['Item_Weight'].mean(), inplace=True)
combi['Outlet_Size'].fillna('missing', inplace=True )

print(combi.isna().sum())

# data pre-processing
print(combi['Item_Fat_Content'].value_counts())

# dictionary to replace the fat-content categories
fat_content_dict = {'Low Fat': 0, 'Regular': 1, 'LF': 0, 'Low fat': 0, 'reg': 1}
combi['Item_Fat_Content'] = combi['Item_Fat_Content'].map(fat_content_dict)
print(combi['Item_Fat_Content'].value_counts())

# feature engineering steps using
# create a unique identifier for the data points/rows
combi['id'] = combi['Item_Identifier']+combi['Outlet_Identifier']
combi.drop('Item_Identifier', axis=1, inplace=True)

# creating an entityset
es = ft.EntitySet(id='sales')
# adding a dataframe
es.entity_from_dataframe(entity_id='bigmart',
                         dataframe=combi,
                         index='id')
es.normalize_entity(base_entity_id='bigmart',
                    new_entity_id='outlet',
                    index='Outlet_Identifier')
additional_variables = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']
print(es)

# Deep Feature Synthesis(DFS)
feature_matrix, feature_name = ft.dfs(entityset=es,
                                      target_entity='bigmart',
                                      max_depth=2,
                                      verbose=1)

print(feature_matrix.columns)

feature_matrix = feature_matrix.reindex(index=combi['id'])
feature_matrix = feature_matrix.reset_index()

# build the model


categorical_features = np.where(feature_matrix.dtypes == 'object')[0]

for i in categorical_features:
    feature_matrix.iloc[:,i] = feature_matrix.iloc[:,i].astype('str')

# split the feature n=matrix into train and test data sets
feature_matrix.drop('id', axis=1, inplace=True)
X_train = feature_matrix[:8523]
X_test = feature_matrix[8523:]

# removing unnecessary variable
X_train.drop('Outlet_Identifier', inplace =True, axis =1)
X_test.drop('Outlet_Identifier', axis=1, inplace=True)

# identifying categorical features
categorical_features = np.where(X_train.dtypes == 'object')[0]
print('\n',categorical_features)

#Split the training dataset into train and validation dataset
X_train, X_valid, y_train, y_valid = train_test_split(X_train, sales,
                                                      test_size=0.25, random_state=123)

model_cat = CatBoostRegressor(iterations=100,
                              learning_rate=0.3,
                              depth=6,
                              eval_metric='RMSE',
                              random_seed=7)
# training the model
model_cat.fit(X_train,y_train, cat_features=categorical_features,
              use_best_model= True)
# validation score
print('Validation score : {} '.format(model_cat.score(X_valid, y_valid)))
# testing score






