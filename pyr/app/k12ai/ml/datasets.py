#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file datasets.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-09-11 17:04

import os
import pandas as pd
import numpy as np

__ALL__ = ['ML_load_dataset']


def _get_house_prices_dataset(path):
    train_csv = os.path.join(path, 'train.csv')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f'dataset: {train_csv}')

    train_data_df = pd.read_csv(train_csv)
    train_data_df.drop('Id', axis=1, inplace=True)

    # Missing value
    for col in ("PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond",
            "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual",
            "BsmtFinType2", "BsmtFinType1", "MasVnrType"):
        train_data_df[col].fillna("None", inplace=True)

    for col in ("MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1",
            "GarageArea"):
        train_data_df[col].fillna(0, inplace=True)

    for col in ('MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical', 'KitchenQual'):
        train_data_df[col] = train_data_df[col].fillna(train_data_df[col].value_counts().index[0])

    train_data_df['LotFrontage'] = train_data_df['LotFrontage'].fillna(train_data_df['LotFrontage'].median())
    train_data_df['Functional'] = train_data_df['Functional'].fillna('typical')

    train_data_df = pd.get_dummies(train_data_df)

    X = train_data_df.drop('SalePrice', axis=1)
    y = train_data_df['SalePrice']
    return np.array(X), np.array(y), X.columns, None


def _get_sfcrime_dataset(path):
    train_csv = os.path.join(path, 'train.csv')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f'dataset: {train_csv}')

    train_data_df = pd.read_csv(train_csv, parse_dates=['Dates'])

    # Date
    train_data_df['Year'] = train_data_df['Dates'].dt.year
    train_data_df['Month'] = train_data_df['Dates'].dt.month
    train_data_df['Day'] = train_data_df['Dates'].dt.day
    train_data_df['Hour'] = train_data_df['Dates'].dt.hour
    week_dict = {
        "Monday": 1,
        "Tuesday":2,
        "Wednesday":3,
        "Thursday":4,
        "Friday":5,
        "Saturday":6,
        "Sunday":7
    }
    train_data_df["DayOfWeek"].replace(week_dict, inplace=True)

    # Category
    categories = train_data_df['Category'].unique()
    category_dict = {}
    for i, c in enumerate(categories, 1):
        category_dict[c] = i
    category_dict
    train_data_df["Category"].replace(category_dict, inplace=True)

    # District
    district = train_data_df["PdDistrict"].unique()
    district_dict = {}
    for i, c in enumerate(district, 1):
        district_dict[c] = i
    district_dict
    train_data_df["PdDistrict"].replace(district_dict, inplace=True)

    train_data_df.drop(['Dates', 'Descript', 'Resolution', 'Address'] , axis=1, inplace=True)

    X = train_data_df.drop('Category', axis=1)
    y = train_data_df['Category']
    return np.array(X), np.array(y), X.columns, categories


def _get_titanic_dataset(path):
    train_csv = os.path.join(path, 'train.csv')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f'dataset: {train_csv}')

    train_data_df = pd.read_csv(train_csv)

    # Age
    def _impute_age(cols):
        Age = cols[0]
        Pclass = cols[1]
        if pd.isnull(Age):
            if Pclass == 1:
                return 37
            elif Pclass == 2:
                return 29
            else:
                return 24
        else:
            return Age
    train_data_df['Age'] = train_data_df[['Age','Pclass']].apply(_impute_age, axis=1)

    # Sex
    sex_dict = {
        'male': 0,
        'female': 1,
    }
    train_data_df["Sex"].replace(sex_dict, inplace=True)

    # Embarked
    train_data_df['Embarked'].fillna('S', inplace=True)
    emb_dict = {
        'S': 0,
        'C': 1,
        'Q': 2,
    }
    train_data_df["Embarked"].replace(emb_dict, inplace=True)

    train_data_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True) 

    X = train_data_df.drop('Survived', axis=1)
    y = train_data_df['Survived']
    return np.array(X), np.array(y), X.columns, ['UnSurvived', 'Survived']


DATASET_LOADER = {
    'houseprice': _get_house_prices_dataset,
    'sfcrime': _get_sfcrime_dataset,
    'titanic': _get_titanic_dataset
}


def ML_load_dataset(dname, dataset_root):
    if dname in DATASET_LOADER.keys():
        return DATASET_LOADER[dname](dataset_root)
    raise FileNotFoundError(dname)
