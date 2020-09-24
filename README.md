# Team PH at the COVID-19 Retweet Prediction Challenge at CIKM2020 Analyticup

This repository contains the code and other resources for our proposed solution for the COVID-19 Retweet Prediction Challenge at CIKM2020 Analyticup. The proposed solution ranked 4th on the final testing leaderboard among 20 teams (51 teams in the validation phase). The pre-print of our report is available [here](http://parklize.github.io/publications/CIKM2020_Analuticup.pdf). 

<br/>

## Requirements
- Download relevant data files from [here](https://pan.baidu.com/s/1cE8eapywzoeXPt-W7t-WVA) with password: cgfm, unpack any tar.gz files in the *model* directory
    - *model* directory contains trained models 
    - *tmp* directory contains extracted data for training and prediction
    - *data* directory contains raw data provided by the challenge origranizers
- Check and install relevant packages based on requrirements.txt - `pip install requirements.txt`
- All experiments are run with a commodity
laptop (Intel CoreI5 processor at 2.6 GHz, 8GB of RAM, and with 200GB swap space)

<br/>

## Scripts
- **test.py** loads the test data and run different RERFs to get the prediction results in the *output* directory and ensembles those results. Afterwards, it applied personalized patching to update the final prediction results for users having a sufficient number of tweets in the training set.   
- **train.py** contains code for training different RERFs, which are used in the test.py.
- **utils.py** contains necessary utility functions for train.py and test.py.

<br/>

## Models used for the global ensemble
| No.        | RERF Details  | Weight |
| ------------- |---|:-------------:|
| 1     | LinearRegression(fit_intercept=False)<br>RandomForestRegressor(n_estimators=500, max_depth=20, random_state=7)| 1 |
| 2     | MLPRegressor(batch_size=1024, hidden_layer_sizes=(64,32,16,8,8), random_state=7)<br>RandomForestRegressor(n_estimators=1000, max_depth=18, random_state=7)| 1 |
| 3     | MLPRegressor(batch_size=2048, hidden_layer_sizes=(128,64,32,16,8,8), random_state=77)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=77)| 1 |
| 4     | MLPRegressor(batch_size=2048, hidden_layer_sizes=(128,64,32,16,8), random_state=18)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=18)| 1 |
| 5     | MLPRegressor(batch_size=2048, hidden_layer_sizes=(64,16,8), random_state=19)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=19)| 1 |
| 6     | MLPRegressor(batch_size=2048, hidden_layer_sizes=(128,64,16,8), random_state=20)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=20)| 1 |
| 7     | MLPRegressor(batch_size=4096, hidden_layer_sizes=(128,64,16,8,4), random_state=201)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=201)| 1 |
| 8     | MLPRegressor(batch_size=4096, hidden_layer_sizes=(128,64,32,8), random_state=211)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=211)| 2 |
| 9     | MLPRegressor(batch_size=4096, hidden_layer_sizes=(128,64,32,8,8), random_state=22)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=22)| 1 |
| 10    | MLPRegressor(batch_size=4096, hidden_layer_sizes=(128,64,32,16,8), random_state=27)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=27)| 1 |
| 11    | MLPRegressor(batch_size=4096, hidden_layer_sizes=(128,64,32,32,16), random_state=211)<br>RandomForestRegressor(n_estimators=500, max_depth=18, random_state=28)| 1 |
| 12    | xDeepFM()<br>RandomForestRegressor(n_estimators=500, max_depth=16, random_state=28)| 1 |
| 13    | DeepFM()<br>RandomForestRegressor(n_estimators=500, max_depth=16, random_state=29)| 2 |
| 14    | DeepFM()<br>RandomForestRegressor(n_estimators=500, max_depth=16, random_state=28)| 1 |
<br/>

## Citation
Guangyuan Piao and Weipeng Huang, Regression-enhanced Random Forests with Personalized Patching for COVID-19 Retweet Prediction, *CIKM Analyticup Workshop at CIKM'20*, Galway, Ireland, 2020. \[[PDF](http://parklize.github.io/publications/CIKM2020_Analuticup.pdf)\] [[bibtex]](https://parklize.github.io/bib/CIKM2020_Analyticup.bib)
