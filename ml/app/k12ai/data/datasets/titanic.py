#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file titanic.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-25 22:59

import os
import pandas as pd
from k12ai.data.base import K12DataLoader


class TitanicDataLoader(K12DataLoader):
    @staticmethod
    def get_dataset(path):
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
        return train_data_df.drop('Survived', axis=1), train_data_df['Survived']
