import json
import pandas as pd
import os
import numpy as np


def extract_json(data_dir_path):
    """
    Takes all json files in given directory, concatenates them and returns a pandas DataFrame
    """
    if not os.path.isdir(data_dir_path):
        raise Exception("Directory does not exist.")
    if not len(os.listdir(data_dir_path)) > 0:
        raise Exception("Directory is empty.")

    data = []
    for file in os.listdir(data_dir_path):
        if file.endswith(".json"):
            with open(os.path.join(data_dir_path, file)) as f:
                data.append(pd.DataFrame.from_dict(json.load(f)))

    # impose same naming convention for all files
    for json_df in data:
        json_df.columns = ["Country", "Customer ID", "Invoice", "Price", "Stream ID", "Times Viewed", "Year", "Month", "Day"]

    df = pd.concat(data)
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    return df.sort_values(by="date").reset_index(drop=True)


def create_features(df):
    """
    - Aggregate data from raw dataframe
    - Impute missing values for Price variable
    - Get historical features of price
    - Produce target variable, i.e. sum of revenue in next 30 days
    """
    df = df.groupby(by=["Country", "date"]).agg({"Price": "sum"}).reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # add missing dates by reindexing each time series
    appended_dfs = []
    for country in df["Country"].unique():
        temp_df = df.loc[df["Country"] == country, :]
        dr = pd.date_range(temp_df["date"].min(),
                           temp_df["date"].max())
        temp_df = temp_df.set_index("date")
        s = temp_df["Price"]
        s = s.reindex(dr, fill_value=0)
        temp_df = pd.DataFrame(s)
        temp_df["Country"] = country
        appended_dfs.append(temp_df)
    df = pd.concat(appended_dfs).reset_index()
    df.columns = ["date", "Price", "Country"]

    # create features and target variable
    for country in df["Country"].unique():
        df.loc[df["Country"] == country, "Price_7d"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=7).sum()
        df.loc[df["Country"] == country, "Price_14d"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=14).sum()
        df.loc[df["Country"] == country, "Price_21d"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=21).sum()
        df.loc[df["Country"] == country, "Price_28d"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=28).sum()
        df.loc[df["Country"] == country, "Price_7d_der"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=2).apply(np.diff).rolling(window=7).mean()
        df.loc[df["Country"] == country, "Price_14d_der"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=2).apply(np.diff).rolling(window=14).mean()
        df.loc[df["Country"] == country, "Price_21d_der"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=2).apply(np.diff).rolling(window=21).mean()
        df.loc[df["Country"] == country, "Price_28d_der"] = df.loc[df["Country"] == country, "Price"].shift(1).rolling(
            window=2).apply(np.diff).rolling(window=28).mean()
        df.loc[df["Country"] == country, "target"] = df.loc[df["Country"] == country, "Price"].shift(-30).rolling(30).sum()
    # include day and month to capture seasonality trends
    df["Month"] = df["date"].dt.month
    df["Day"] = df["date"].dt.day

    return df.dropna()


def select_top_countries(df, n=10):
    """
    Only return subset of dataframe with top n countries based on total revenue, i.e. sum of Price
    """
    top_countries = list(df.groupby("Country").agg({"Price": "sum"}).sort_values(by="Price", ascending=False).iloc[:n].index)
    return df.loc[df["Country"].isin(top_countries), :]


def fetch_data(data_dir_path):
    """
    Load data from directory and preprocess it
    """
    print("Extracting data from folder " + data_dir_path)
    data_df = extract_json(data_dir_path)

    print("Preprocessing data")
    return select_top_countries(create_features(data_df))



