# required packages
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

import seaborn as sns

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import category_encoders as ce

'''
@param df: the dataset
@param id: the index column
@param variance_threshold 
'''
def clean_dataset(df, csv_name, variance_threshold=0.1):
    # STEP 1: remove features that have significant missing data (threshold is 80% not mmissing)
    percent_notnull = df.notnull().sum() * 100 / len(df)
    percent_notnull_df = pd.DataFrame(
        {'column_name': df.columns, 'percent_notnull': percent_notnull})
    percent_notnull_df = percent_notnull_df[percent_notnull_df['percent_notnull'] >= 80]

    columns = percent_notnull_df.column_name.to_numpy()
    print('========= Finish step 1 ===========')

    # STEP 2: Remove features that have very low variance, user can
    # define variance threshold
    numeric_features = [
        f for f in columns if np.issubdtype(df[f].dtype, np.number)]
    categorical_features = [
        f for f in columns if not np.issubdtype(df[f].dtype, np.number)]

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit_transform(df[numeric_features])

    # reassign numeric_features to features that have variance higher
    # than a certain threshold
    numeric_features = np.array(numeric_features)[
        selector.get_support(indices=True)].tolist()

    # probably there should be a better way to add VarianceThreshold
    # to the pipeline
    df = df[numeric_features + categorical_features]

    print('========= Finish step 2 ===========')

    # STEP 3: Scale numeric columns and apply encoder to categorical
    # column
    columns = numeric_features + categorical_features
    df = df[columns]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', ce.ordinal.OrdinalEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    df = preprocessor.fit_transform(df)

    # sklearn preprocessor remove the column name so we need to add them back in
    df = pd.DataFrame(df, columns=columns)

    #add in empty sale and list date columns
    df['list_date'] = ['null' for i in range(len(df))]
    df['sale_date'] = ['null' for i in range(len(df))]

    df.to_csv(f'{csv_name}.csv')

    print('========= Finish step 3 ===========')
    print(df.info())
    return df
