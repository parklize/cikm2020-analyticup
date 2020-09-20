import utils
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.python.keras.models import load_model
from deepctr.layers import custom_objects
from deepctr.models import xDeepFM, DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder


def lrrf_predict(val):
    val_lr = pd.get_dummies(val[features], columns=["timeseg", "day_of_week"])

    # model 1 - LRRF
    # load lr
    filename = './model/lr.sav'
    reg = pickle.load(open(filename, 'rb'))
    lr_y_val_predict = reg.predict(np.log(val_lr.values + 1))

    # load rf
    filename = './model/randomforest_regressor_500e_lrallfeatures_rs7.sav'
    reg = pickle.load(open(filename, 'rb'))
    rf_val_predict = reg.predict(val[features].values)

    result_list = list()
    for e in reg.estimators_:
        result_list.append(e.predict(val[features].values))

    result_list = np.array(result_list)
    print(result_list.shape)

    # combine
    y_val_predict = utils.get_normal_counter(lr_y_val_predict + rf_val_predict, logarithm="e")
    np.savetxt("output/model1.predict", y_val_predict.astype(int), fmt='%i')


def nnrf_predict(val):

    val_lr = pd.get_dummies(val[features], columns=["timeseg", "day_of_week"])

    #####################
    # model 2-10
    # load from existing one
    model_dict = {
        './model/nnnoval_shuffle.sav': './model/randomforest_regressor_1000e_nnallfeatures_rs7.sav', # model 2
        './model/nnnoval_shuffle_rs77_128-64-32-16-8-8.sav': './model/randomforest_regressor_500e_nnallfeatures_rs77.sav', # model 3
        './model/nnnoval_shuffle_rs18_128-64-32-16-8.sav': './model/randomforest_regressor_500e_nnallfeatures_rs18.sav', # model 4
        './model/nnnoval_shuffle_rs19_64-16-8.sav': './model/randomforest_regressor_500e_nnallfeatures_rs19.sav', # model 5
        './model/nnnoval_shuffle_rs20_128-64-16-8.sav': './model/randomforest_regressor_500e_nnallfeatures_rs20.sav', # model 6
        './model/nnnoval_shuffle_rs201_128-64-16-8-4.sav': './model/randomforest_regressor_500e_nnallfeatures_rs201.sav', # model 7
        './model/nnnoval_shuffle_rs211_128-64-32-8.sav': './model/randomforest_regressor_500e_nnallfeatures_rs211.sav', # model 8
        './model/nnnoval_shuffle_rs22_128-64-32-8-8.sav': './model/randomforest_regressor_500e_nnallfeatures_rs22.sav',  # model 9
        './model/nnnoval_shuffle_rs27_128-64-32-16-8.sav': './model/randomforest_regressor_500e_nnallfeatures_rs27.sav', # mdoel 10
        './model/nnnoval_shuffle_rs211_128-64-32-32-16.sav': './model/randomforest_regressor_500e_nnallfeatures_rs211_28.sav'
    }

    for idx, regressor_path in enumerate(model_dict.keys()):
        regr = pickle.load(open(regressor_path, 'rb'))
        nn_y_val_predict = regr.predict(np.log(val_lr.values + 1))

        # load rf
        filename = model_dict[regressor_path]
        reg = pickle.load(open(filename, 'rb'))
        rf_val_predict = reg.predict(val[features].values)

        # combine
        y_val_predict = utils.get_normal_counter(nn_y_val_predict + rf_val_predict, logarithm="e")
        # np.savetxt("output/model{}.predict".format(idx+2), y_val_predict.astype(int), fmt='%i')
        np.savetxt("output/model11.predict", y_val_predict.astype(int), fmt='%i')


def fmrf_predict(val):

    val[[c for c in features if c not in ["timeseg", "day_of_week"]]] = \
        np.log1p(val[[c for c in features if c not in ["timeseg", "day_of_week"]]])

    ################# Model 12

    ##########################
    # RF
    # load rf
    filename = './model/randomforest_regressor_500e_xdeepfm_rs211_28.sav'
    reg = pickle.load(open(filename, 'rb'))
    rf_val_predict = reg.predict(val[features].values)

    ###########################
    # DeepFM

    features_ = [f.replace("#", "") for f in features]
    val.columns = [f.replace("#", "") for f in val.columns]

    sparse_features = ["timeseg", "day_of_week"]
    dense_features = [f for f in features_ if f not in ["timeseg", "day_of_week"]]

    def encoding(data, feat, encoder):
        data[feat] = encoder.fit_transform(data[feat])

    [encoding(val, feat, LabelEncoder()) for feat in sparse_features]

    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=val[feat].nunique(), embedding_dim=4) \
                              for i, feat in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    print(len(dense_feature_columns))

    # features to be used for dnn part of xdeepfm
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    # features to be used for linear part of xdeepfm
    linear_feature_columns = sparse_feature_columns + dense_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.load_weights('./model/xDeepFM_w.h5')

    test_model_input = {name: val[name].values for name in feature_names}

    deepfm_pred = model.predict(test_model_input, batch_size=256)
    deepfm_pred = deepfm_pred.reshape(rf_val_predict.shape)
    deepfm_pred_counter = utils.get_normal_counter(deepfm_pred + rf_val_predict, logarithm="e")
    np.savetxt("output/model12.predict", deepfm_pred_counter.astype(int), fmt='%i')

    ########################## Model 13

    ##########################
    # RF
    # load rf
    filename = './model/randomforest_regressor_500e_xdeepfm_rs211_29.sav'
    reg = pickle.load(open(filename, 'rb'))
    rf_val_predict = reg.predict(val[features].values)

    ###########################
    # DeepFM

    features_ = [f.replace("#", "") for f in features]
    val.columns = [f.replace("#", "") for f in val.columns]

    sparse_features = ["timeseg", "day_of_week"]
    dense_features = [f for f in features_ if f not in ["timeseg", "day_of_week"]]

    def encoding(data, feat, encoder):
        data[feat] = encoder.fit_transform(data[feat])

    [encoding(val, feat, LabelEncoder()) for feat in sparse_features]

    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=val[feat].nunique(), embedding_dim=4) \
                              for i, feat in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    print(len(dense_feature_columns))

    # features to be used for dnn part of xdeepfm
    dnn_feature_columns = dense_feature_columns
    # features to be used for linear part of xdeepfm
    linear_feature_columns = dense_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.load_weights('./model/xDeepFM_w_seed29.h5')

    test_model_input = {name: val[name].values for name in feature_names}

    deepfm_pred = model.predict(test_model_input, batch_size=256)
    deepfm_pred = deepfm_pred.reshape(rf_val_predict.shape)
    deepfm_pred_counter = utils.get_normal_counter(deepfm_pred + rf_val_predict, logarithm="e")
    np.savetxt("output/model13.predict", deepfm_pred_counter.astype(int), fmt='%i')


def global_model_predict(val):
    if not os.path.exists('output'):
        os.makedirs('output')
        print('created output folder as it is not existing...')

    if not os.path.exists('output/model1.predict'):
        lrrf_predict(val)
    if not os.path.exists('output/model2.predict'):
        nnrf_predict(val)
    if not os.path.exists('output/model12.predict'):
        fmrf_predict(val)

    # load separate prediction files
    predict = pd.read_csv('output/model1.predict', header=None)
    predict.columns = ["model1"]

    model2_predict = pd.read_csv('output/model2.predict', header=None)
    model2_predict.columns = ["yhat"]
    model3_predict = pd.read_csv('output/model3.predict', header=None)
    model3_predict.columns = ["yhat"]
    model4_predict = pd.read_csv('output/model4.predict', header=None)
    model4_predict.columns = ["yhat"]
    model5_predict = pd.read_csv('output/model5.predict', header=None)
    model5_predict.columns = ["yhat"]
    model6_predict = pd.read_csv('output/model6.predict', header=None)
    model6_predict.columns = ["yhat"]
    model7_predict = pd.read_csv('output/model7.predict', header=None)
    model7_predict.columns = ["yhat"]
    model8_predict = pd.read_csv('output/model8.predict', header=None)
    model8_predict.columns = ["yhat"]
    model9_predict = pd.read_csv('output/model9.predict', header=None)
    model9_predict.columns = ["yhat"]
    model10_predict = pd.read_csv('output/model10.predict', header=None)
    model10_predict.columns = ["yhat"]
    model11_predict = pd.read_csv('output/model11.predict', header=None)
    model11_predict.columns = ["yhat"]
    model12_predict = pd.read_csv('output/model12.predict', header=None)
    model12_predict.columns = ["yhat"]
    model13_predict = pd.read_csv('output/model13.predict', header=None)
    model13_predict.columns = ["yhat"]
    model14_predict = pd.read_csv('output/model14.predict', header=None)
    model14_predict.columns = ["yhat"]

    # arithmetic mean
    predict["model2"] = model2_predict["yhat"]
    predict["model3"] = model3_predict["yhat"]
    predict["model4"] = model4_predict["yhat"]
    predict["model5"] = model5_predict["yhat"]
    predict["model6"] = model6_predict["yhat"]
    predict["model7"] = model7_predict["yhat"]
    predict["model8"] = model8_predict["yhat"] * 2
    predict["model9"] = model9_predict["yhat"]
    predict["model10"] = model10_predict["yhat"]
    predict["model11"] = model11_predict["yhat"]
    predict["model12"] = model12_predict["yhat"]
    predict["model13"] = model13_predict["yhat"] * 2
    predict["model14"] = model14_predict["yhat"]
    predict["yhatavg"] = (
                            predict["model1"] + predict["model2"] \
                            + predict["model3"] + predict["model4"] \
                            + predict["model5"] + predict["model6"] \
                            + predict["model7"] + predict["model8"] \
                            + predict["model9"] + predict["model10"] \
                            + predict["model11"] + predict["model12"] \
                            + predict["model13"] + predict["model14"] \
                        ) / 16.0

    np.savetxt("output/temp.predict", np.round(predict["yhatavg"].values).astype(int), fmt='%i')


def patching_personalized_models():
    ##########################
    # Patching

    # patching with prestored ones
    prediction_file = "output/temp.predict"
    current_predict = pd.read_csv(prediction_file, header=None)
    current_predict.columns = ["yhat"]
    # current_predict

    # patching vals
    if not os.path.exists("tmp/pactching_df_skipcountzerogt6_test_withcounters.csv"):
        utils.get_patching_result()
    patch_predict = pd.read_csv("tmp/pactching_df_skipcountzerogt6_test_withcounters.csv")
    patch_predict = patch_predict[patch_predict["countnonzero"] > 6]
    print(patch_predict.shape)
    # print(current_predict.iloc[patch_predict["index"].values, 0])
    current_predict.iloc[patch_predict["index"].values, 0] = patch_predict["pred"].values
    # print(current_predict.iloc[patch_predict["index"].values, 0])

    np.savetxt("output/test.predict", current_predict["yhat"].values.astype(int), fmt='%i')

    # patching with prestored ones for heavy users
    prediction_file = "output/test.predict"
    current_predict = pd.read_csv(prediction_file, header=None)
    current_predict.columns = ["yhat"]
    # current_predict

    # patching vals
    if not os.path.exists("tmp/pactching_df_heavyusers100_test_withcounters.csv"):
        utils.get_patching_result_heavyusers()
    patch_predict = pd.read_csv("tmp/pactching_df_heavyusers100_test_withcounters.csv")
    patch_predict = patch_predict[patch_predict["countnonzero"] >= 160]
    print(patch_predict.shape)
    current_predict.iloc[patch_predict["index"].values, 0] = patch_predict["pred"].values

    np.savetxt("output/test.predict", current_predict["yhat"].values.astype(int), fmt='%i')


if __name__ == "__main__":

    features = utils.features
    val = utils.load_test_data()

    print("test data shape", val.shape)

    global_model_predict(val)
    patching_personalized_models()