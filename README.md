## Team PH at the COVID-19 Retweet Prediction Challenge at CIKM2020 Analyticup

This repository contains the code and other resources for our proposed solution for the COVID-19 Retweet Prediction Challenge at CIKM2020 Analyticup. The proposed solution ranked 4th on the final testing leaderboard among 20 teams (51 teams in the validation phase). The pre-print of our report is available [here](). 

<br/>

### Requirements
- Download relevant data files from [here](https://pan.baidu.com/s/1cE8eapywzoeXPt-W7t-WVA) with password: cgfm, unpack any tar.gz files in the *model* directory
    - *model* directory contains trained models 
    - *tmp* directory contains extracted data for training and prediction
    - *data* directory contains raw data provided by the challenge origranizers
- Check and install relevant packages based on requrirements.txt 
- All experiments are run with a commodity
laptop (Intel CoreI5 processor at 2.6 GHz, 8GB of RAM, and with 200GB swap space)

<br/>

### Scripts
- **test.py** loads the test data and run different RERFs to get the prediction results in the *output* directory and ensembles those results. Afterwards, it applied personalized patching to update the final prediction results for users having a sufficient number of tweets in the training set.   
- **train.py** contains code for training different RERFs, which are used in the test.py.
- **utils.py** contains necessary utility functions for train.py and test.py.

<br/>

### Citation
*Regression-enhanced Random Forests with Personalized Patching for COVID-19 Retweet Prediction*, Guangyuan Piao and Weipeng Huang, CIKM Analyticup Workshop at CIKM'20, Galway, Ireland, 2020. \[[PDF]()\]