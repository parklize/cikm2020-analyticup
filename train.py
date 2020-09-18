import pandas as pd
import numpy as np
import time
import pickle
import utils
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from deepctr.models import xDeepFM, DeepFM
from deepctr.feature_column import  SparseFeat, DenseFeat, get_feature_names


def train_lrrf(X_train_lr, y_train):
    """ Train and store LRRF (LR + Random Forest) """
    y_train_logscale = np.log(y_train + 1.)
    reg = LinearRegression(fit_intercept=False).fit(np.log(X_train_lr.values+1), y_train_logscale)
    filename = './model/lr.sav'
    pickle.dump(reg, open(filename, 'wb'))

    # predict
    lr_y_train_predict = reg.predict(np.log(X_train_lr.values + 1))
    lr_yhat = utils.get_normal_counter(lr_y_train_predict, logarithm="e")
    print(mean_squared_log_error(lr_yhat, y_train))

    # for training residual RF
    rf_y_train = y_train_logscale - lr_y_train_predict

    reg = RandomForestRegressor(max_depth=20,
                                n_estimators=500,
                                random_state=7,
                                n_jobs=3,
                                verbose=5)
    start_time = time.time()
    reg.fit(X_train[features].values, rf_y_train, )
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    # save randomforest regressor
    filename = './model/randomforest_regressor_500e_lrallfeatures_rs7.sav'
    pickle.dump(reg, open(filename, 'wb'))


def train_nnrf(X_train_lr, y_train):
    """ Train and store NNRF (Neural Networks - MLP + Random Forest) """
    regr = MLPRegressor(random_state=7,
                        hidden_layer_sizes=(64, 32, 16, 8, 8),
                        batch_size=1024,
                        learning_rate_init=.01,
                        early_stopping=False,
                        verbose=True,
                        shuffle=True,
                        n_iter_no_change=10)

    y_train_logscale = np.log(y_train + 1.)

    # fit
    start_time = time.time()
    regr.fit(np.log(X_train_lr.values + 1), y_train_logscale)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

    filename = './model/nnnoval_shuffle.sav'
    pickle.dump(regr, open(filename, 'wb'))

    lr_y_train_predict = regr.predict(np.log(X_train_lr.values + 1))
    # for training residual RF
    rf_y_train = y_train_logscale - lr_y_train_predict

    reg = RandomForestRegressor(max_depth=18,
                                n_estimators=500,
                                random_state=77,
                                n_jobs=3,
                                verbose=5)
    start_time = time.time()
    reg.fit(X_train[features].values, rf_y_train, )
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    # save randomforest regressor
    filename = './model/randomforest_regressor_1000e_nnallfeatures_rs7.sav'
    pickle.dump(reg, open(filename, 'wb'))


def train_xdeepfmrf(X_train, y_train):
    """ Train and store FMRF (xDeepFM + Random Forest) """
    X_train[[c for c in features if c not in ["timeseg", "day_of_week"]]] = np.log1p(
        X_train[[c for c in features if c not in ["timeseg", "day_of_week"]]])

    features_ = [f.replace("#", "") for f in features]
    X_train.columns = [f.replace("#", "") for f in X_train.columns]

    sparse_features = ["timeseg", "day_of_week"]
    dense_features = [f for f in features_ if f not in ["timeseg", "day_of_week"]]

    def encoding(data, feat, encoder):
        data[feat] = encoder.fit_transform(data[feat])

    [encoding(X_train, feat, LabelEncoder()) for feat in sparse_features]

    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=X_train[feat].nunique(), embedding_dim=4) \
                              for i, feat in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    # features to be used for dnn part of xdeepfm
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    # features to be used for linear part of xdeepfm
    linear_feature_columns = sparse_feature_columns + dense_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: X_train[name].values for name in feature_names}

    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', seed=28)
    # compiling the model
    model.compile("adam", "mse", metrics=['mse'], )
    # training the model
    y_train_logscale = np.log(y_train + 1.)
    history = model.fit(train_model_input, y_train_logscale, batch_size=256, epochs=200, verbose=2)

    model.save_weights('./model/xDeepFM_w.h5')

    pred = model.predict(train_model_input, batch_size=256)
    rf_y_train = y_train_logscale - pred.reshape(6167810, )

    reg = RandomForestRegressor(max_depth=16,
                                max_features=.5,
                                n_estimators=500,
                                random_state=28,
                                n_jobs=3,
                                verbose=5)
    start_time = time.time()
    reg.fit(X_train[features].values, rf_y_train, )
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    filename = './model/randomforest_regressor_500e_xdeepfm_rs211_28.sav'
    pickle.dump(reg, open(filename, 'wb'))


def train_deepfmrf(X_train, y_train):
    """ Train and store FMRF (DeepFM + Random Forest) """
    X_train[[c for c in features if c not in ["timeseg", "day_of_week"]]] = np.log1p(
        X_train[[c for c in features if c not in ["timeseg", "day_of_week"]]])

    features_ = [f.replace("#", "") for f in features]
    X_train.columns = [f.replace("#", "") for f in X_train.columns]

    sparse_features = ["timeseg", "day_of_week"]
    dense_features = [f for f in features_ if f not in ["timeseg", "day_of_week"]]

    def encoding(data, feat, encoder):
        data[feat] = encoder.fit_transform(data[feat])

    [encoding(X_train, feat, LabelEncoder()) for feat in sparse_features]

    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=X_train[feat].nunique(), embedding_dim=4) \
                              for i, feat in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    # features to be used for dnn part of xdeepfm
    dnn_feature_columns = dense_feature_columns
    # features to be used for linear part of xdeepfm
    linear_feature_columns = dense_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: X_train[name].values for name in feature_names}

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', seed=29)
    # compiling the model
    model.compile("adam", "mse", metrics=['mse'], )
    # training the model
    y_train_logscale = np.log(y_train + 1.)
    history = model.fit(train_model_input, y_train_logscale, batch_size=4096, epochs=150, verbose=2)

    model.save_weights('./model/xDeepFM_w_seed29.h5')

    pred = model.predict(train_model_input, batch_size=4096)
    rf_y_train = y_train_logscale - pred.reshape(6167810, )

    reg = RandomForestRegressor(max_depth=16,
                                max_features=.5,
                                n_estimators=500,
                                random_state=29,
                                n_jobs=3,
                                verbose=5)
    start_time = time.time()
    reg.fit(X_train[features].values, rf_y_train, )
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    filename = './model/randomforest_regressor_500e_xdeepfm_rs211_29.sav'
    pickle.dump(reg, open(filename, 'wb'))


if __name__ == "__main__":

    features = utils.features
    X_train, y_train = utils.load_train_data()

    # filtering
    merged_tcounts = utils.get_merged_tcounts()
    usernames = merged_tcounts[merged_tcounts["tcounts_x"] >= 10]["username"].values
    X_train = X_train[~X_train["username"].isin(usernames)]
    y_train = y_train[X_train.index]
    X_train_lr = pd.get_dummies(X_train[features], columns=["timeseg", "day_of_week"])

    print("training data shape", X_train.shape)

    train_lrrf(X_train_lr, y_train)
    train_nnrf(X_train_lr, y_train)
    train_xdeepfmrf(X_train, y_train)
    train_deepfmrf(X_train, y_train)