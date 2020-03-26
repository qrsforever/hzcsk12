#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file house-prices.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-26 22:29

import os
import pandas as pd
import numpy as np
from k12ai.data.base import K12DataLoader


class HousePricesDataLoader(K12DataLoader):
    @staticmethod
    def get_dataset(path):
        train_csv = os.path.join(path, 'train.csv')
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f'dataset: {train_csv}')

        train_data_df = pd.read_csv(train_csv)
        train_data_df.drop('Id', axis=1, inplace=True)

        # Missing value
        for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',
                'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure',
                'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrArea', 'Utilities'):
            train_data_df[col] = train_data_df[col].fillna('None')

        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtHalfBath', 'BsmtFullBath',
                      'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF'):
            train_data_df[col] = train_data_df[col].fillna(0)

        for col in ('MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical', 'KitchenQual'):
            train_data_df[col] = train_data_df[col].fillna(train_data_df[col].value_counts().index[0])

        train_data_df['LotFrontage'] = train_data_df['LotFrontage'].fillna(train_data_df['LotFrontage'].median())
        train_data_df['Functional'] = train_data_df['Functional'].fillna('typical')

        # For dummies
        for col in ('MSSubClass', 'OverallCond', 'YearBuilt', 'YrSold', 'MoSold'):
            train_data_df[col] = train_data_df[col].astype(str)
            train_data_df[col] = train_data_df[col].astype(str)
            train_data_df[col] = train_data_df[col].astype(str)
            train_data_df[col] = train_data_df[col].astype(str)
            train_data_df[col] = train_data_df[col].astype(str)

        train_data_df = pd.get_dummies(train_data_df)

        X = train_data_df.drop('SalePrice', axis=1)
        y = train_data_df['SalePrice']
        return np.array(X), np.array(y), X.columns, None
