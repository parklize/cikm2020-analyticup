import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from django.contrib.admin.utils import flatten


features = [
    "#followers",
    "#friends",
    "#favorites",
    "timeseg",
    "entity_exist",
    "hashtag_exist",
    "mention_exist",
    "url_exist",
    "top10_entities",
    "top10_hashtags",
    "top10_mentions",
    "tcount",
    "weekend",
    "tlen",
    "entity_count",
    "mention_count",
    "hashtag_count",
    "url_count",
    "mcount",
    "hcount",
    "ecount",
    "dcount",
    "mu_friends_max",
    "mu_follower_max",
    "ratio_fav_#followers",
    "ratio_fri_#followers",
    "day_of_week",
    "sentiment_p",
    "sentiment_n",
    "sentiment_ppn"
]


def get_normal_counter(n, logarithm="10"):
    """ Get normal counter x from log(x+1) value """
    if logarithm == "10":
        return np.array([max(0, x) for x in (np.power(10, n).round().astype(int)-1)])
    else:
        return np.array([max(0, x) for x in (np.exp(n).round().astype(int)-1)])


def msle(y, yhat, logarithm="10"):
    """ Calculate MSLE """
    if logarithm == "10":
        return np.mean(np.square(np.log10(y+1)-np.log10(yhat+1)))
    else:
        return np.mean(np.square(np.log1p(y), np.log1p(yhat)))


def cdf(data):
    """ Plot CDF of the data """
    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf,linestyle='--', marker="o", color='b')
    plt.ylim((0,1))
    plt.ylabel("CDF")
    plt.grid(True)

    plt.show()


######################################
# Feature engineering functions
def add_features(X_train, dataset="train"):
    # get time segment 0-23
    X_train["timeseg"] = X_train["timestamp"].str[11:13]
    X_train['timeseg'] = pd.to_numeric(X_train['timeseg'])
    X_train["date"] = X_train["timestamp"].str[-4:] + "-" + X_train["timestamp"].str[4:10]
    X_train["weekend"] = X_train["timestamp"].str[:3].isin(["Sun", "Sat"])
    X_train["weekend"] = X_train["weekend"].astype(int)
    print('added time seg...')

    # exist or not features
    for col in ["entities", "hashtags", "mentions", "urls"]:
        X_train[col] = X_train[col].astype(str)
    X_train["entity_exist"] = X_train["entities"] != "null;"
    X_train["hashtag_exist"] = X_train["hashtags"] != "null;"
    X_train["mention_exist"] = X_train["mentions"] != "null;"
    X_train["url_exist"] = X_train["urls"] != "null;"
    print('added exit or not features...')

    # h/e/m/url count
    X_train["entity_count"] = X_train["entities"].str.split(";").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null"]))
    X_train["hashtag_count"] = X_train["hashtags"].str.split(" ").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null;"]))
    X_train["mention_count"] = X_train["mentions"].str.split(" ").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null;"]))
    X_train["url_count"] = X_train["urls"].str.split(":-: ").apply(
        lambda x: len([y for y in x if len(y) > 0 and y != "null;"]))
    print('added count of h/e/m/url...')

    # approx length of tweets = sum of all h/e/m/url
    X_train["tlen"] = X_train["entity_count"] + X_train["hashtag_count"] + X_train["mention_count"] + X_train[
        "url_count"]

    # top 10 entities (updated to 20)
    X_train["top10_entities"] = [0] * X_train.shape[0]
    with open('./tmp/entity_top10.pickle', 'rb') as handle:
        m_dict = pickle.load(handle)
    for de in X_train["date"].unique():
        X_train.iloc[X_train[X_train["date"] == de].index, list(X_train.columns.values).index("top10_entities")] = \
        X_train[X_train["date"] == de]["entities"].str.split(";").apply(
            lambda x: len(list(set([x_.split(":")[0] for x_ in x]) & set(m_dict[de]))))

    X_train["hashtags"] = X_train["hashtags"].astype(str)
    X_train["mentions"] = X_train["mentions"].astype(str)
    print('added top 20 entities count')

    # top 10 hashtags
    X_train["top10_hashtags"] = [0] * X_train.shape[0]

    with open('./tmp/hashtag_top10.pickle', 'rb') as handle:
        m_dict = pickle.load(handle)
    for de in X_train["date"].unique():
        X_train.iloc[X_train[X_train["date"] == de].index, list(X_train.columns.values).index("top10_hashtags")] = \
        X_train[X_train["date"] == de]["hashtags"].str.split(" ").apply(lambda x: len(list(set(x) & set(m_dict[de]))))
    print('added top 10 hashtags...')

    # top 10 mentions
    X_train["top10_mentions"] = [0] * X_train.shape[0]

    with open('./tmp/mention_top10.pickle', 'rb') as handle:
        m_dict = pickle.load(handle)
    for de in X_train["date"].unique():
        X_train.iloc[X_train[X_train["date"] == de].index, list(X_train.columns.values).index("top10_mentions")] = \
        X_train[X_train["date"] == de]["mentions"].str.split(" ").apply(lambda x: len(list(set(x) & set(m_dict[de]))))
    print('added top 10 mentions...')

    # tweet count of each user
    with open('./tmp/tcount.npy', 'rb') as f:
        tcount = np.load(f)
    with open('./tmp/mcount.npy', 'rb') as f:
        mcount = np.load(f)
    with open('./tmp/hcount.npy', 'rb') as f:
        hcount = np.load(f)
    with open('./tmp/ecount.npy', 'rb') as f:
        ecount = np.load(f)
    with open('./tmp/dcount.npy', 'rb') as f:
        dcount = np.load(f)
    if dataset == "train":
        X_train['tcount'] = tcount[:8151524]
        X_train['mcount'] = mcount[:8151524]
        X_train['hcount'] = hcount[:8151524]
        X_train['ecount'] = ecount[:8151524]
        X_train['dcount'] = dcount[:8151524]
    elif dataset == "val":
        X_train['tcount'] = tcount[8151524:8151524 + 961182]
        X_train['mcount'] = mcount[8151524:8151524 + 961182]
        X_train['hcount'] = hcount[8151524:8151524 + 961182]
        X_train['ecount'] = ecount[8151524:8151524 + 961182]
        X_train['dcount'] = dcount[8151524:8151524 + 961182]
    else:
        X_train['tcount'] = tcount[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['mcount'] = mcount[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['hcount'] = hcount[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['ecount'] = ecount[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['dcount'] = dcount[8151524 + 961182:8151524 + 961182 + 961183]
    #     X_train.groupby('username')['username'].transform('count')
    print('added tcount...')

    X_train["entity_exist"] = X_train["entity_exist"].astype(int)
    X_train["hashtag_exist"] = X_train["hashtag_exist"].astype(int)
    X_train["mention_exist"] = X_train["mention_exist"].astype(int)
    X_train["url_exist"] = X_train["url_exist"].astype(int)
    print("changed exit features to int type...")

    return X_train


def add_mhedcounters(X_train, dataset="train"):
    """ Add mention/hashtag/entity counters acros all datasets """

    with open('./tmp/mcount.npy', 'rb') as f:
        mcount = np.load(f)
    with open('./tmp/hcount.npy', 'rb') as f:
        hcount = np.load(f)
    with open('./tmp/ecount.npy', 'rb') as f:
        ecount = np.load(f)
    with open('./tmp/dcount.npy', 'rb') as f:
        dcount = np.load(f)
    if dataset == "train":
        X_train['mcount'] = mcount[:8151524]
        X_train['hcount'] = hcount[:8151524]
        X_train['ecount'] = ecount[:8151524]
        X_train['dcount'] = dcount[:8151524]
    elif dataset == "val":
        X_train['mcount'] = mcount[8151524:8151524 + 961182]
        X_train['hcount'] = hcount[8151524:8151524 + 961182]
        X_train['ecount'] = ecount[8151524:8151524 + 961182]
        X_train['dcount'] = dcount[8151524:8151524 + 961182]
    else:
        X_train['mcount'] = mcount[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['hcount'] = hcount[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['ecount'] = ecount[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['dcount'] = dcount[8151524 + 961182:8151524 + 961182 + 961183]

    return X_train


def add_mention_user_info_counters(X_train, dataset="train"):
    """ Add mentioned user info counters for all datasets """

    with open('./tmp/mu_follower_max.npy', 'rb') as f:
        mu_follower_max = np.load(f)
    with open('./tmp/mu_friends_max.npy', 'rb') as f:
        mu_friends_max = np.load(f)

    if dataset == "train":
        X_train['mu_friends_max'] = mu_friends_max[:8151524]
        X_train['mu_follower_max'] = mu_follower_max[:8151524]
    elif dataset == "val":
        X_train['mu_friends_max'] = mu_friends_max[8151524:8151524 + 961182]
        X_train['mu_follower_max'] = mu_follower_max[8151524:8151524 + 961182]
    else:
        X_train['mu_friends_max'] = mu_friends_max[8151524 + 961182:8151524 + 961182 + 961183]
        X_train['mu_follower_max'] = mu_follower_max[8151524 + 961182:8151524 + 961182 + 961183]

    return X_train


def add_date_tcount(X_train, dataset="train"):
    """ Add date tcount for all datasets """

    with open('./tmp/date_tcount.npy', 'rb') as f:
        date_tcount = np.load(f)

    if dataset == "train":
        X_train['date_tcount'] = date_tcount[:8151524]
    elif dataset == "val":
        X_train['date_tcount'] = date_tcount[8151524:8151524 + 961182]
    else:
        X_train['date_tcount'] = date_tcount[8151524 + 961182:8151524 + 961182 + 961183]

    return X_train


def add_ratios(X_train):
    """#followers #friends #favorites"""
    X_train["#followers"] = X_train["#followers"].astype(float)
    X_train["ratio_fav_#followers"] = X_train["#favorites"] / (X_train["#followers"] + 1.0)
    X_train["ratio_fri_#followers"] = X_train["#friends"] / (X_train["#followers"] + 1.0)
    return X_train


def add_flag(X_train):
    with open("./tmp/exist_invaltest.npy", "rb") as f:
        flag = np.load(f)
    X_train["exist_invaltest"] = flag
    return X_train


def add_importance(X_train):
    """2019-SEP 30 will be zero then +1 """
    list_of_dates = list(X_train["date"].unique())
    tqdm.pandas()
    X_train["time_importance"] = X_train["date"].progress_apply(lambda x: list_of_dates.index(x)).values
    return X_train


def add_year_month_date(X_train):
    X_train["year"] = X_train["date"].str[:4]
    X_train["day"] = X_train["date"].str[-2:]
    m_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
              "Nov": 11, "Dec": 12}
    tqdm.pandas()
    X_train["month"] = X_train["date"].str[5:-3].progress_apply(lambda x: m_dict[x])
    X_train["year"] = X_train["year"].astype(int)
    X_train["day"] = X_train["day"].astype(int)
    return X_train


def add_kmeans_group(X_train):
    kmeans = pickle.load(open('./model/kmeans.sav', 'rb'))
    groups = kmeans.predict(X_train[["#followers", "#friends"]].values)
    X_train["group"] = groups
    return X_train


def add_day_of_week(X_train):
    m_dict = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    tqdm.pandas()
    X_train["day_of_week"] = X_train["timestamp"].str[:3].progress_apply(lambda x: m_dict[x])
    return X_train


def add_sentiments(X_train):
    # sentiments
    X_train['sentiment_p'], X_train['sentiment_n'] = X_train['sentiment'].str.split(' ', 1).str
    X_train['sentiment_p'] = pd.to_numeric(X_train['sentiment_p']) + 6.0  # to be positive for log scaling
    X_train['sentiment_n'] = pd.to_numeric(X_train['sentiment_n']) + 6.0
    # plus for training main, minus for patching
    X_train['sentiment_ppn'] = X_train['sentiment_p'] + X_train['sentiment_n']
    return X_train


def get_columns():
    columns = pd.read_csv("data/feature.name", sep="\t", header=None).values[0]
    columns = list(columns)
    print(columns)

    return columns


def load_train_data():
    # load features
    columns = get_columns()

    # train: 8151524
    # val: 961182
    # test: 961183

    # load data
    y_train = pd.read_csv("data/train.solution", header=None).T.values[0]

    file_path = "./tmp/X_train_reformated.csv"
    if os.path.exists(file_path):
        print("loading training data from file...")
        X_train = pd.read_csv(file_path, header=0, index_col=0)
    else:
        X_train = pd.read_csv(
            "data/train.data", delimiter="\t",
            header=None, names=columns, nrows=len(y_train),
            lineterminator="\n", engine="c", quoting=3, quotechar=None,
            #     parse_dates=['timestamp'],
            usecols=["tweet_id", "timestamp",
                     "username", "#followers", "#friends", "#favorites", "entities", "mentions", "hashtags", "urls",
                     "sentiment"
                     ])
        # val = pd.read_csv("data/validation.data", sep="\t", header=None, names=columns[1:])
        # test = pd.read_csv("data/test.data", sep="\t", header=None, names=columns[1:])
        X_train = add_features(X_train)
        X_train = add_mhedcounters(X_train)
        X_train = add_mention_user_info_counters(X_train)
        X_train = add_ratios(X_train)

        X_train.to_csv("./tmp/X_train_reformated.csv")

    X_train = add_day_of_week(X_train)
    X_train = add_sentiments(X_train)

    print(X_train.shape, y_train.shape)

    return X_train, y_train


def load_test_data():
    # load features
    columns = get_columns()

    # Test data
    file_path = "./tmp/test_reformated.csv"
    if os.path.exists(file_path):
        print("loading test data from file...")
        val = pd.read_csv(file_path, header=0, index_col=0)
    else:
        val = pd.read_csv("data/test.data", delimiter="\t",
                              header=None, names=columns[1:],
                              lineterminator="\n", engine="c", quoting=3, quotechar=None)
        val = add_features(val, dataset="test")
        val = add_mhedcounters(val, dataset="test")
        val = add_mention_user_info_counters(val, dataset="test")
        val = add_ratios(val)
        val.to_csv("./tmp/test_reformated.csv")

    val = add_day_of_week(val)
    val = add_sentiments(val)

    print(val.shape)

    return val


def get_merged_tcounts():
    # load from csv files already there, to get merged tweets per twitter account
    X_train_tcounts = pd.read_csv("./tmp/X_train_tcounts.csv", header=0, index_col=0)
    val_tcounts = pd.read_csv("./tmp/val_tcounts.csv", header=0, index_col=0)
    test_tcounts = pd.read_csv("./tmp/test_tcounts.csv", header=0, index_col=0)

    print(len(X_train_tcounts), len(val_tcounts), len(test_tcounts))
    merged_tcounts = X_train_tcounts.merge(
        val_tcounts, how="inner", on="username").merge(test_tcounts, how="inner", on="username")

    return merged_tcounts


def get_patching_result_heavyusers():
    """ Patching with Ridge for users have many examples """
    X_train, y_train = load_train_data()
    val = load_test_data()
    merged_tcounts = get_merged_tcounts()

    index_list = list()
    pred_list = list()
    countnonzero_list = list()
    usernames = merged_tcounts[merged_tcounts["tcounts_x"] >= 100]["username"].values

    print(len(usernames))

    param_grid = {'max_depth': [1, 2, 4, 6, 8, 10, 12, 14, 16]}

    for idx, username in enumerate(usernames):
        print(idx, username)

        # train
        user_df = X_train[X_train["username"] == username]
        user_train_simple = user_df[features].values
        y_train_ = y_train[user_df.index]

        countnonzero = sum((user_df["#favorites"].values * y_train_) > 0)

        if countnonzero >= 100:
            y_train_logscale_ = np.log10(y_train_ + 1)

            # linear regression with log#favorites
            lr = Ridge(alpha=5., fit_intercept=False) \
                .fit(np.log(user_df[features].values + 1.), y_train_logscale_)
            lr_user_train_predict = lr.predict(np.log(user_df[features].values + 1))

            print(lr.coef_)

            rf_user_train = y_train_logscale_ - lr_user_train_predict

            search = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=20), param_grid, cv=5,
                                  n_jobs=3)
            search.fit(user_train_simple, rf_user_train)
            regressor = search.best_estimator_
            print(regressor)

            # predict on val
            user_df_val = val[val["username"] == username]

            # linear regression
            lr_user_val_predict = lr.predict(np.log(user_df_val[features].values + 1))

            user_val_simple = user_df_val[features].values
            user_val_predict = regressor.predict(user_val_simple)
            user_val_predict = get_normal_counter(user_val_predict + lr_user_val_predict)
            index_list.append(user_df_val.index)
            pred_list.append(user_val_predict)
            countnonzero_list += [countnonzero] * len(user_val_predict)
        else:
            print("skipped due to sum zero issue of ytrain and favorites")

    index_list_ = flatten([list(x.values) for x in index_list])
    pred_list_ = flatten([list(x) for x in pred_list])
    patching_df = pd.DataFrame({"index": index_list_, "pred": pred_list_, "countnonzero": countnonzero_list})
    patching_df.to_csv("pactching_df_heavyusers100_test_withcounters.csv", index=False)


def get_patching_result():
    """ Patching with a single feature with LinearRegression for users """
    X_train, y_train = load_train_data()
    val = load_test_data()
    merged_tcounts = get_merged_tcounts()

    index_list = list()
    pred_list = list()
    countnonzero_list = list()
    usernames = merged_tcounts[merged_tcounts["tcounts_x"] >= 7]["username"].values
    print(len(usernames))

    param_grid = {'max_depth': [1, 2, 4, 6, 8, 10, 12, 14, 16]}
    for idx, username in enumerate(usernames):
        print(idx, username)

        # train
        user_df = X_train[X_train["username"] == username]
        user_train_simple = user_df[features].values
        y_train_ = y_train[user_df.index]

        countnonzero = sum((user_df["#favorites"].values * y_train_) > 0)

        #     if np.sum(y_train_) != 0 and np.sum(user_df["#favorites"].values) != 0:
        if countnonzero > 6:
            #         y_train_logscale_ = np.log10(y_train_+1)
            y_train_logscale_ = np.log(y_train_ + 1)

            # linear regression with log#favorites
            lr = LinearRegression(fit_intercept=False) \
                .fit(np.log(user_df["#favorites"].values + 1).reshape(-1, 1), y_train_logscale_)
            lr_user_train_predict = lr.predict(np.log(user_df["#favorites"].values + 1).reshape(-1, 1))
            rf_user_train = y_train_logscale_ - lr_user_train_predict

            if os.path.exists("./model/{}.sav".format(username)):
                print("loading existing model for {}".format(username))
                regressor = pickle.load(open("./model/{}.sav".format(username), 'rb'))
            else:
                search = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=20), param_grid, cv=5,
                                      n_jobs=3)
                search.fit(user_train_simple, rf_user_train)
                regressor = search.best_estimator_
                pickle.dump(regressor, open("./model/{}.sav".format(username), 'wb'))

            # predict on val
            user_df_val = val[val["username"] == username]

            # linear regression
            lr_user_val_predict = lr.predict(np.log(user_df_val["#favorites"].values + 1).reshape(-1, 1))

            user_val_simple = user_df_val[features].values
            user_val_predict = regressor.predict(user_val_simple)
            user_val_predict = get_normal_counter(user_val_predict + lr_user_val_predict, logarithm="e")
            #     print(user_df_val.index)
            index_list.append(user_df_val.index)
            pred_list.append(user_val_predict)
            countnonzero_list += [countnonzero] * len(user_val_predict)
        else:
            print("skipped due to sum zero issue of ytrain and favorites")

    index_list_ = flatten([list(x.values) for x in index_list])
    pred_list_ = flatten([list(x) for x in pred_list])
    patching_df = pd.DataFrame({"index": index_list_, "pred": pred_list_, "countnonzero": countnonzero_list})
    patching_df.to_csv("pactching_df_skipcountzerogt6_test_withcounters.csv", index=False)


if __name__ == "__main__":
    load_test_data()
