#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file sf-crime.py
# @brief
# @author QRS
# @blog qrsforever.github.io
# @version 1.0
# @date 2020-02-24 20:58

import os
import pandas as pd
from k12ml.data.base import K12DataLoader
from sklearn.model_selection import train_test_split


class SFCrimeDataLoader(K12DataLoader):
    @staticmethod
    def get_dataset(path, kwargs):
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
        return train_test_split(
                train_data_df.drop('Category', axis=1),
                train_data_df['Category'], **kwargs)
