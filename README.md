# Technical Analysis Stock Predictor System

Welcome to the repository for my Senior project. Here are the important files to note:

1. **stock_predictor_fin.py**: This is the primary file where the model is trained and a prediction for the one hour change in stock price is made. The ticker symbol can be replaced for any stock that is supported by Yahoo Finance. Due to a save error, the file needs to be updated to scale features post splitting in order to avoid data leakage. The results of each run are appended to results.csv.
2. **run_bitcoin.bat**: This is a batch file that, when executed, will run stock_predictor_fin.py. This is ideal when coupled with a scheduling software like Windows Task Scheduler.
3. **get_real_time_accuracy.py**: This file accepts results.csv and returns the accuracy of the real time predictions. Additional work is needed to only include results that are 1 hour old.
