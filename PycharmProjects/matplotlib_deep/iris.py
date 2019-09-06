import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def read_dataset():
    return pd.read_csv('iris.csv'), pd.read_csv('iris_metadata.csv')

def merge_dfs(iris, metadata):
    return pd.merge(iris, metadata, left_on='species', right_on='name')

def split_df(df, ratio):
    split_index=round(len(df)*ratio)
    #len으로 길이 (row 갯수)
    return df.iloc[:split_index], df.iloc[split_index:]
    #iloc
def truncate_non_toxics(df):
    print('#####################')
    print(df['toxic'])
    print(df['toxic']>0)
    return df.loc[df['toxic']>0]
#좀 더 알아보기
def main():
    split_ratio = 0.7
    iris, metadata = read_dataset()
    merge_result=merge_dfs(iris,metadata)
    split_train, split_test = split_df(merge_result,split_ratio)
    print(merge_result)
    print(split_train)
    train_iris=truncate_non_toxics(split_train)
    test_iris=truncate_non_toxics(split_test)

    print(train_iris)
    print(test_iris)
    print(train_iris.describe())
    print(test_iris.describe())

main()




