## Team PH at the COVID-19 Retweet Prediction Challenge at CIKM2020 Analyticup

This repository contains the code and other resources for our proposed solution for the COVID-19 Retweet Prediction Challenge at CIKM2020 Analyticup. The proposed solution ranked 4th on the final leaderboard among 20 teams.

### Requirements
- Download relevant data files from [here]()
- Check and install relevant packages based on requrirements.txt 

### Scripts
- test.py loads the test data and run different RERFs to get the prediction results in the *output* directory and ensembles those results. Afterwards, it applied personalized patching to update the final prediction results for users having a sufficient number of tweets in the training set.   
- train.py contains code for training different RERFs, which are used in the test.py.
- utils.py contains necessary utility functions for train.py and test.py.

### Model details


### Citation
