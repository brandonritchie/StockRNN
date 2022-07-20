#%%
import pandas as pd
import datetime
from datetime import date, timezone

#%%
results = pd.read_csv("results.csv")

stock = "BTC-USD"
date1 = date.today()
days = datetime.timedelta(59)
new_date = date1 - days
date1 = date1 - datetime.timedelta(1)
stock = yf.download(stock, start = new_date, interval = "2m").reset_index()
results['utc_time_at_pred'] = pd.to_datetime(results['utc_time_at_pred'])

merge1 = ((pd.merge_asof(left = results, right = stock, left_on='utc_time_at_pred', right_on = 'Datetime',
                                 tolerance=pd.Timedelta("2m"))[['pred', 'utc_time_at_pred', 'model_r_square', 'model_accuracy', 'Datetime', 'Close']])
                                 .rename(columns = {'pred':'predicted_change', 'utc_time_at_pred':'start_time', 'Datetime':'joined_start_time', 'Close':'beg_price'}))
merge1['start_plus_hour'] =  merge1['start_time'] + pd.to_timedelta(1, unit='h')

# ********* Figure out why na's are returning after join **********
merge2 = pd.merge_asof(left = merge1, right = stock, left_on='start_plus_hour', right_on = 'Datetime',
                                 tolerance=pd.Timedelta("2m")).drop(['High', 'Low', 'Close', 'Adj Close', 'Volume'], axis = 1).rename(columns = {'Datetime':'joined_start_plus', 'Open':'end_price'})
merge2['actual_change'] = merge2['end_price'] - merge2['beg_price']
merge2['correct_pred'] = merge2.apply(lambda dat: (1
                                               if dat['predicted_change']*dat['actual_change'] > 0
                                               else 0),
                                   axis=1)
merge2['difference_pred_act'] = merge2['actual_change'] - merge2['predicted_change']
fin_dat = merge2[['predicted_change', 'actual_change', 'correct_pred', 'difference_pred_act', 'model_r_square', 'model_accuracy']]

def result_data(results, stock):
  # Takes stock data and model predictions and performs a join an hour ahead. Additional work is needed to review NA join and logic for data that is not an hour old.
  results['utc_time_at_pred'] = pd.to_datetime(results['utc_time_at_pred'])
  merge1 = ((pd.merge_asof(left = results, right = stock, left_on='utc_time_at_pred', right_on = 'Datetime',
                                 tolerance=pd.Timedelta("2m"))[['pred', 'utc_time_at_pred', 'model_r_square', 'model_accuracy', 'Datetime', 'Close']])
                                 .rename(columns = {'pred':'predicted_change', 'utc_time_at_pred':'start_time', 'Datetime':'joined_start_time', 'Close':'beg_price'}))
  merge1['start_plus_hour'] =  merge1['start_time'] + pd.to_timedelta(1, unit='h')

  # ********* Figure out why na's are returning after join **********
  merge2 = pd.merge_asof(left = merge1, right = stock, left_on='start_plus_hour', right_on = 'Datetime',
                                  tolerance=pd.Timedelta("2m")).drop(['High', 'Low', 'Close', 'Adj Close', 'Volume'], axis = 1).rename(columns = {'Datetime':'joined_start_plus', 'Open':'end_price'})
  merge2['actual_change'] = merge2['end_price'] - merge2['beg_price']
  merge2['correct_pred'] = merge2.apply(lambda dat: (1
                                                if dat['predicted_change']*dat['actual_change'] > 0
                                                else 0),
                                    axis=1)
  merge2['difference_pred_act'] = merge2['actual_change'] - merge2['predicted_change']
  fin_dat = merge2[['predicted_change', 'actual_change', 'correct_pred', 'difference_pred_act', 'model_r_square', 'model_accuracy']]
  return fin_dat

dat = result_data(results, stock)
acc = len(dat[(dat['correct_pred'] ==1)]) / len(dat)
print(f'Prediction accuracy: {acc}')