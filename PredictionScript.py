import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor


def model_prediction(input_data):
    #importing data and dropping correlated columns
    data = pd.read_csv('bike_sharing_modified.csv')
    data.drop(columns=['cnt_check','temp','atemp','dteday','day_of_year'],inplace=True)

    #splitting data into features and target
    x = data.copy().drop(["cnt"], axis=1) # features
    y = data.loc[:, "cnt"] # target

    #creating a new variable for each type of feature
    categorical_features = x.select_dtypes(include=["O"])
    discrete_features = x.select_dtypes(include=["int64"])
    continuous_features = x.select_dtypes(include=["float64"])

    #############################################

    #PIPELINE CREATION 

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectKBest, chi2, f_regression

    cat_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('label', OrdinalEncoder())
        
    ])

    disc_pipe = Pipeline([

        ('imputer',SimpleImputer(strategy='median')),
    ])

    disc_transformer = ColumnTransformer([
        ("cat", cat_pipe, categorical_features.columns ),
        ('disc', disc_pipe, discrete_features.columns ),

    ],
    remainder="drop")

    discrete_selector = Pipeline([
        ('disc_transformer', disc_transformer),
        ('selector',SelectKBest(chi2,k=8))

    ])



    continuous_selector = Pipeline([    
        ('scaler', MinMaxScaler()),
        ('selector',SelectKBest(f_regression,k=8))
    ])

    selector = ColumnTransformer([
        ('discrete_selector', discrete_selector, pd.concat([discrete_features,categorical_features],axis=1).columns),
        ('continuous_selector', continuous_selector, continuous_features.columns),
    ])

    selected_data = pd.DataFrame(selector.fit_transform(x,y), columns=selector.get_feature_names_out())
    print(len(selected_data.columns))
    selected_data.columns = [col.split('__')[2] if (col.split('__')[1] == 'cat') or (col.split('__')[1] == 'disc')  else col.split('__')[1]for col in selected_data.columns]


    final_data = data[selected_data.columns]

    ################################################################################

    # Define the features and the target variable
    X = final_data
    y = data['cnt']

    #define continuous and numerical data

    num_feat = X.select_dtypes(include=['float64','int64'])
    cat_feat = X.select_dtypes(include=['object'])


    # Create pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine both pipeline into a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_feat.columns),
            ('cat', categorical_transformer, cat_feat.columns)])

    X_preprocessed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


    lgbm = LGBMRegressor(learning_rate=0.05, max_depth=-1, n_estimators=200, num_leaves=50)

    lgbm.fit(X_train, y_train)

    prediction = lgbm.predict(input_data)
    

    return prediction[0]







