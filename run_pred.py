import yfinance as yf
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import argrelextrema
from numpy.polynomial import Polynomial as P
from sklearn.model_selection import train_test_split
import xgboost as xgb
import scipy
#from datetime import datetime, timezone, date
import datetime
from datetime import date, timezone

def get_google_trend(query, start_date, end_date):
    try:
        pytrends = TrendReq(hl = 'en-US', tz = 360)
        pytrends.build_payload(kw_list = [query], timeframe = f'{str(start_date)} {end_date}')
        dat = pytrends.interest_over_time().reset_index()
        dat['year'] = pd.DatetimeIndex(dat['date']).year
        dat['month'] = pd.DatetimeIndex(dat['date']).month
        dat['day'] = pd.DatetimeIndex(dat['date']).day
        return dat
    except:
        return(False)

def main_func():
    stock = "BTC-USD"
    date1 = date.today()
    days = datetime.timedelta(59)
    new_date = date1 - days
    date1 = date1 - datetime.timedelta(1)
    df = yf.download(stock, start = new_date, interval = "2m").reset_index()

    date_inp1 = new_date - datetime.timedelta(1)
    date_inp2 = date1 - datetime.timedelta(1)

    bitcoin = get_google_trend('bitcoin', date_inp1, date_inp2)
    bitcoin_stock = get_google_trend('bitcoin stock', new_date, date1)
    if isinstance(bitcoin, pd.DataFrame) and isinstance(bitcoin_stock, pd.DataFrame):
        bitcoin_dat = bitcoin.query('isPartial == False')
        bitcoin_stock_dat = bitcoin_stock.query('isPartial == False')

    Volume = df.filter(['Volume'])
    obs = Volume.shape[0]
    sc = MinMaxScaler(feature_range=(0,1))
    volume_scaled_1 = sc.fit_transform(Volume)
    X_v = []
    #length_TF_1 = len(testing_length_TF_1)
    for i in range(90, obs):
        if (i + 90) < obs:
            X_v.append(volume_scaled_1[i-90:i, 0])
    # Making change a scale from 0 to 1
    sc = MinMaxScaler(feature_range=(0,1))
    X_v = np.array(X_v)

    close = df.filter(['Close'])
    obs = close.shape[0]
    sc = MinMaxScaler(feature_range=(0,1))
    close_scaled_1 = sc.fit_transform(close)
    date1 = df.filter(['Datetime']).values
    X = []
    date_begin = []
    date_end = []
    y_i = []
    close_length = close.values
    #length_TF_1 = len(testing_length_TF_1)
    for i in range(90, obs):
        if (i + 90) < obs:
            X.append(close_scaled_1[i-90:i, 0])
            date_begin.append(date1[i-90])
            date_end.append(date1[i])
            y_i.append(close_length[i + 30] - close_length[i])
    # Making change a scale from 0 to 1
    sc = MinMaxScaler(feature_range=(0,1))
    y_i = sc.fit_transform(y_i)
    X = np.array(X)

    # Bring it all together into one dataset
    local_maxima_count_list = []
    local_minima_count_list = []

    slope_list = []

    q1_p_list = []
    q2_p_list = []
    q3_p_list = []
    q4_p_list = []

    local_max_q1_list = []
    local_max_q2_list = []
    local_max_q3_list = []
    local_max_q4_list = []

    local_min_q1_list = []
    local_min_q2_list = []
    local_min_q3_list = []
    local_min_q4_list = []

    slope_q1_list = []
    slope_q2_list = []
    slope_q3_list = []
    slope_q4_list = []
    for i in X:
        y = i
        x = list(range(1,91))
        poly = P.fit(x,y, 18)
        fx, fy = poly.linspace(100)
        local_maxima_c = len(argrelextrema(fy, np.greater)[0])
        local_maxima_count_list.append(local_maxima_c)
        local_minima_c = len(argrelextrema(fy, np.less)[0])
        local_minima_count_list.append(local_minima_c)
        slope = (fy[-1]-fy[0])
        slope_list.append(slope)
        quartile_lengths = (max(fy) - min(fy)) / 4
        q1 = (min(fy) + quartile_lengths)
        q2 = q1 + quartile_lengths
        q3 = q2 + quartile_lengths
        # Q1
        q1_p = len(fy[fy < q1]) / len(fy)
        q1_p_list.append(q1_p)
        # Q2
        q2_p =len(list(filter(lambda p: p >= q1 and p < q2, fy))) / len(fy)
        q2_p_list.append(q2_p)
        # Q3
        q3_p =len(list(filter(lambda p: p >= q2 and p < q3, fy)))/ len(fy)
        q3_p_list.append(q3_p)
        # Q4
        q4_p =len(fy[fy >= q3])/ len(fy)
        q4_p_list.append(q4_p)
        q_length = int(len(fy)/4)
        q1_fy = fy[:q_length]
        q2_fy = fy[q_length:(q_length*2)]
        q3_fy = fy[(q_length*2):(q_length*3)]
        q4_fy = fy[(q_length*3):]
        # Local max's
        local_max_q1 = len(argrelextrema(q1_fy, np.greater)[0])
        local_max_q1_list.append(local_max_q1)
        local_max_q2 = len(argrelextrema(q2_fy, np.greater)[0])
        local_max_q2_list.append(local_max_q2)
        local_max_q3 = len(argrelextrema(q3_fy, np.greater)[0])
        local_max_q3_list.append(local_max_q3)
        local_max_q4 = len(argrelextrema(q4_fy, np.greater)[0])
        local_max_q4_list.append(local_max_q4)
        # Local mins
        local_min_q1 = len(argrelextrema(q1_fy, np.less)[0])
        local_min_q1_list.append(local_min_q1)
        local_min_q2 = len(argrelextrema(q2_fy, np.less)[0])
        local_min_q2_list.append(local_min_q2)
        local_min_q3 = len(argrelextrema(q3_fy, np.less)[0])
        local_min_q3_list.append(local_min_q3)
        local_min_q4 = len(argrelextrema(q4_fy, np.less)[0])
        local_min_q4_list.append(local_min_q4)
        # Slopes
        slope_q1 = (q1_fy[-1]-q1_fy[0])
        slope_q1_list.append(slope_q1)
        slope_q2 = (q2_fy[-1]-q2_fy[0])
        slope_q2_list.append(slope_q2)
        slope_q3 = (q3_fy[-1]-q3_fy[0])
        slope_q3_list.append(slope_q3)
        slope_q4 = (q4_fy[-1]-q4_fy[0])
        slope_q4_list.append(slope_q4)

    local_maxima_count_v_list = []
    local_minima_count_v_list = []

    slope_v_list = []

    q1_v_list = []
    q2_v_list = []
    q3_v_list = []
    q4_v_list = []

    local_max_q1_v_list = []
    local_max_q2_v_list = []
    local_max_q3_v_list = []
    local_max_q4_v_list = []

    local_min_q1_v_list = []
    local_min_q2_v_list = []
    local_min_q3_v_list = []
    local_min_q4_v_list = []

    slope_q1_v_list = []
    slope_q2_v_list = []
    slope_q3_v_list = []
    slope_q4_v_list = []
    for i in X_v:
        y = i
        x = list(range(1,91))
        poly = P.fit(x,y, 18)
        fx, fy = poly.linspace(100)
        local_maxima_c = len(argrelextrema(fy, np.greater)[0])
        local_maxima_count_v_list.append(local_maxima_c)
        local_minima_c = len(argrelextrema(fy, np.less)[0])
        local_minima_count_v_list.append(local_minima_c)
        slope = (fy[-1]-fy[0])
        slope_v_list.append(slope)
        quartile_lengths = (max(fy) - min(fy)) / 4
        q1 = (min(fy) + quartile_lengths)
        q2 = q1 + quartile_lengths
        q3 = q2 + quartile_lengths
        # Q1
        q1_p = len(fy[fy < q1]) / len(fy)
        q1_v_list.append(q1_p)
        # Q2
        q2_p =len(list(filter(lambda p: p >= q1 and p < q2, fy))) / len(fy)
        q2_v_list.append(q2_p)
        # Q3
        q3_p =len(list(filter(lambda p: p >= q2 and p < q3, fy)))/ len(fy)
        q3_v_list.append(q3_p)
        # Q4
        q4_p =len(fy[fy >= q3])/ len(fy)
        q4_v_list.append(q4_p)
        q_length = int(len(fy)/4)
        q1_fy = fy[:q_length]
        q2_fy = fy[q_length:(q_length*2)]
        q3_fy = fy[(q_length*2):(q_length*3)]
        q4_fy = fy[(q_length*3):]
        # Local max's
        local_max_q1 = len(argrelextrema(q1_fy, np.greater)[0])
        local_max_q1_v_list.append(local_max_q1)
        local_max_q2 = len(argrelextrema(q2_fy, np.greater)[0])
        local_max_q2_v_list.append(local_max_q2)
        local_max_q3 = len(argrelextrema(q3_fy, np.greater)[0])
        local_max_q3_v_list.append(local_max_q3)
        local_max_q4 = len(argrelextrema(q4_fy, np.greater)[0])
        local_max_q4_v_list.append(local_max_q4)
        # Local mins
        local_min_q1 = len(argrelextrema(q1_fy, np.less)[0])
        local_min_q1_v_list.append(local_min_q1)
        local_min_q2 = len(argrelextrema(q2_fy, np.less)[0])
        local_min_q2_v_list.append(local_min_q2)
        local_min_q3 = len(argrelextrema(q3_fy, np.less)[0])
        local_min_q3_v_list.append(local_min_q3)
        local_min_q4 = len(argrelextrema(q4_fy, np.less)[0])
        local_min_q4_v_list.append(local_min_q4)
        # Slopes
        slope_q1 = (q1_fy[-1]-q1_fy[0])
        slope_q1_v_list.append(slope_q1)
        slope_q2 = (q2_fy[-1]-q2_fy[0])
        slope_q2_v_list.append(slope_q2)
        slope_q3 = (q3_fy[-1]-q3_fy[0])
        slope_q3_v_list.append(slope_q3)
        slope_q4 = (q4_fy[-1]-q4_fy[0])
        slope_q4_v_list.append(slope_q4)

        data = {'pred':y_i.ravel(),
            'start_date':date_begin,
            'end_date':date_end,
            'local_minima_count': local_minima_count_list, 
            'local_maxima_count':local_maxima_count_list, 
            'slope': slope_list, 
            'quartile1_proportion': q1_p_list,
            'quartile2_proportion':q2_p_list,
            'quartile3_proportion':q3_p_list,
            'quartile4_proportion':q4_p_list,
            'q1_local_max':local_max_q1_list,
            'q2_local_max':local_max_q2_list,
            'q3_local_max':local_max_q3_list,
            'q4_local_max':local_max_q4_list,
            'q1_local_min':local_min_q1_list,
            'q2_local_min':local_min_q2_list,
            'q3_local_min':local_min_q3_list,
            'q4_local_min':local_min_q4_list,
            'q1_slope':slope_q1_list,
            'q2_slope':slope_q2_list,
            'q3_slope':slope_q3_list,
            'q4_slope':slope_q4_list,
            'local_minima_count_volume': local_minima_count_v_list, 
            'local_maxima_count_volume':local_maxima_count_v_list, 
            'slope_volume': slope_v_list, 
            'quartile1_proportion_volume': q1_v_list,
            'quartile2_proportion_volume':q2_v_list,
            'quartile3_proportion_volume':q3_v_list,
            'quartile4_proportion_volume':q4_v_list,
            'q1_local_max_volume':local_max_q1_v_list,
            'q2_local_max_volume':local_max_q2_v_list,
            'q3_local_max_volume':local_max_q3_v_list,
            'q4_local_max_volume':local_max_q4_v_list,
            'q1_local_min_volume':local_min_q1_v_list,
            'q2_local_min_volume':local_min_q2_v_list,
            'q3_local_min_volume':local_min_q3_v_list,
            'q4_local_min_volume':local_min_q4_v_list,
            'q1_slope_volume':slope_q1_v_list,
            'q2_slope_volume':slope_q2_v_list,
            'q3_slope_volume':slope_q3_v_list,
            'q4_slope_volume':slope_q4_v_list}
    dat = pd.DataFrame(data)

    df = dat.explode('start_date').explode('end_date')

    df['year'] = pd.DatetimeIndex(df['end_date']).year
    df['month'] = pd.DatetimeIndex(df['end_date']).month
    df['day'] = pd.DatetimeIndex(df['end_date']).day - 3
    if isinstance(bitcoin, pd.DataFrame) and isinstance(bitcoin_stock, pd.DataFrame):
        df_join1 = pd.merge(df, bitcoin_dat, how = 'left', left_on = ['year','month','day'], right_on = ['year','month','day'])
        df_join2 = df_join1.loc[:,~df_join1.columns.isin(['date', 'isPartial', 'start_date','end_date'])]
        df_join3 = pd.merge(df_join2, bitcoin_stock_dat, how = 'left', left_on = ['year','month','day'], right_on = ['year','month','day'])
        dat = (df_join3.loc[:,~df_join3.columns.isin(['date', 'isPartial', 'year', 'month', 'day'])]).dropna(axis = 0)
        dat.columns.values[-1] = 'bitcoin_stock'

    if 'start_date' in set(list(df.columns)):
        df = df.drop(columns = ['start_date','end_date'])

    dat = dat.sample(frac=1)
    train, test = train_test_split(dat, test_size = 0.2)
    X_train = train.drop(columns = ['pred'])
    y_train = pd.DataFrame(train['pred'])
    X_test = test.drop(columns = ['pred'])
    y_test = pd.DataFrame(test['pred'])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:tweedie'}
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 900
    bst = xgb.train(param, dtrain, num_round, evallist)

    results = pd.DataFrame({"pred":sc.inverse_transform(np.array(bst.predict(dtest)).reshape(1, -1))[0], 'actual':sc.inverse_transform(np.array(y_test.pred).reshape(1, -1))[0]})
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(results.pred, results.actual)
    print(r_value)
    for i in range(2):
        stock = "BTC-USD"
        date1 = date.today()
        df = yf.download(stock, start = date1, interval = "2m").reset_index()
        df = df.iloc[-91:-1:]

        local_maxima_count_list = []
        local_minima_count_list = []

        slope_list = []

        q1_p_list = []
        q2_p_list = []
        q3_p_list = []
        q4_p_list = []

        local_max_q1_list = []
        local_max_q2_list = []
        local_max_q3_list = []
        local_max_q4_list = []

        local_min_q1_list = []
        local_min_q2_list = []
        local_min_q3_list = []
        local_min_q4_list = []

        slope_q1_list = []
        slope_q2_list = []
        slope_q3_list = []
        slope_q4_list = []

        y = df.Close
        x = list(range(1,91))
        poly = P.fit(x,y, 18)
        fx, fy = poly.linspace(100)
        local_maxima_c = len(argrelextrema(fy, np.greater)[0])
        local_maxima_count_list.append(local_maxima_c)
        local_minima_c = len(argrelextrema(fy, np.less)[0])
        local_minima_count_list.append(local_minima_c)
        slope = (fy[-1]-fy[0])
        slope_list.append(slope)
        quartile_lengths = (max(fy) - min(fy)) / 4
        q1 = (min(fy) + quartile_lengths)
        q2 = q1 + quartile_lengths
        q3 = q2 + quartile_lengths
        # Q1
        q1_p = len(fy[fy < q1]) / len(fy)
        q1_p_list.append(q1_p)
        # Q2
        q2_p =len(list(filter(lambda p: p >= q1 and p < q2, fy))) / len(fy)
        q2_p_list.append(q2_p)
        # Q3
        q3_p =len(list(filter(lambda p: p >= q2 and p < q3, fy)))/ len(fy)
        q3_p_list.append(q3_p)
        # Q4
        q4_p =len(fy[fy >= q3])/ len(fy)
        q4_p_list.append(q4_p)
        q_length = int(len(fy)/4)
        q1_fy = fy[:q_length]
        q2_fy = fy[q_length:(q_length*2)]
        q3_fy = fy[(q_length*2):(q_length*3)]
        q4_fy = fy[(q_length*3):]
        # Local max's
        local_max_q1 = len(argrelextrema(q1_fy, np.greater)[0])
        local_max_q1_list.append(local_max_q1)
        local_max_q2 = len(argrelextrema(q2_fy, np.greater)[0])
        local_max_q2_list.append(local_max_q2)
        local_max_q3 = len(argrelextrema(q3_fy, np.greater)[0])
        local_max_q3_list.append(local_max_q3)
        local_max_q4 = len(argrelextrema(q4_fy, np.greater)[0])
        local_max_q4_list.append(local_max_q4)
        # Local mins
        local_min_q1 = len(argrelextrema(q1_fy, np.less)[0])
        local_min_q1_list.append(local_min_q1)
        local_min_q2 = len(argrelextrema(q2_fy, np.less)[0])
        local_min_q2_list.append(local_min_q2)
        local_min_q3 = len(argrelextrema(q3_fy, np.less)[0])
        local_min_q3_list.append(local_min_q3)
        local_min_q4 = len(argrelextrema(q4_fy, np.less)[0])
        local_min_q4_list.append(local_min_q4)
        # Slopes
        slope_q1 = (q1_fy[-1]-q1_fy[0])
        slope_q1_list.append(slope_q1)
        slope_q2 = (q2_fy[-1]-q2_fy[0])
        slope_q2_list.append(slope_q2)
        slope_q3 = (q3_fy[-1]-q3_fy[0])
        slope_q3_list.append(slope_q3)
        slope_q4 = (q4_fy[-1]-q4_fy[0])
        slope_q4_list.append(slope_q4)

        local_maxima_count_v_list = []
        local_minima_count_v_list = []

        slope_v_list = []

        q1_v_list = []
        q2_v_list = []
        q3_v_list = []
        q4_v_list = []

        local_max_q1_v_list = []
        local_max_q2_v_list = []
        local_max_q3_v_list = []
        local_max_q4_v_list = []

        local_min_q1_v_list = []
        local_min_q2_v_list = []
        local_min_q3_v_list = []
        local_min_q4_v_list = []

        slope_q1_v_list = []
        slope_q2_v_list = []
        slope_q3_v_list = []
        slope_q4_v_list = []
        y = df.Volume
        x = list(range(1,91))
        poly = P.fit(x,y, 18)
        fx, fy = poly.linspace(100)
        local_maxima_c = len(argrelextrema(fy, np.greater)[0])
        local_maxima_count_v_list.append(local_maxima_c)
        local_minima_c = len(argrelextrema(fy, np.less)[0])
        local_minima_count_v_list.append(local_minima_c)
        slope = (fy[-1]-fy[0])
        slope_v_list.append(slope)
        quartile_lengths = (max(fy) - min(fy)) / 4
        q1 = (min(fy) + quartile_lengths)
        q2 = q1 + quartile_lengths
        q3 = q2 + quartile_lengths
        # Q1
        q1_p = len(fy[fy < q1]) / len(fy)
        q1_v_list.append(q1_p)
        # Q2
        q2_p =len(list(filter(lambda p: p >= q1 and p < q2, fy))) / len(fy)
        q2_v_list.append(q2_p)
        # Q3
        q3_p =len(list(filter(lambda p: p >= q2 and p < q3, fy)))/ len(fy)
        q3_v_list.append(q3_p)
        # Q4
        q4_p =len(fy[fy >= q3])/ len(fy)
        q4_v_list.append(q4_p)
        q_length = int(len(fy)/4)
        q1_fy = fy[:q_length]
        q2_fy = fy[q_length:(q_length*2)]
        q3_fy = fy[(q_length*2):(q_length*3)]
        q4_fy = fy[(q_length*3):]
        # Local max's
        local_max_q1 = len(argrelextrema(q1_fy, np.greater)[0])
        local_max_q1_v_list.append(local_max_q1)
        local_max_q2 = len(argrelextrema(q2_fy, np.greater)[0])
        local_max_q2_v_list.append(local_max_q2)
        local_max_q3 = len(argrelextrema(q3_fy, np.greater)[0])
        local_max_q3_v_list.append(local_max_q3)
        local_max_q4 = len(argrelextrema(q4_fy, np.greater)[0])
        local_max_q4_v_list.append(local_max_q4)
        # Local mins
        local_min_q1 = len(argrelextrema(q1_fy, np.less)[0])
        local_min_q1_v_list.append(local_min_q1)
        local_min_q2 = len(argrelextrema(q2_fy, np.less)[0])
        local_min_q2_v_list.append(local_min_q2)
        local_min_q3 = len(argrelextrema(q3_fy, np.less)[0])
        local_min_q3_v_list.append(local_min_q3)
        local_min_q4 = len(argrelextrema(q4_fy, np.less)[0])
        local_min_q4_v_list.append(local_min_q4)
        # Slopes
        slope_q1 = (q1_fy[-1]-q1_fy[0])
        slope_q1_v_list.append(slope_q1)
        slope_q2 = (q2_fy[-1]-q2_fy[0])
        slope_q2_v_list.append(slope_q2)
        slope_q3 = (q3_fy[-1]-q3_fy[0])
        slope_q3_v_list.append(slope_q3)
        slope_q4 = (q4_fy[-1]-q4_fy[0])
        slope_q4_v_list.append(slope_q4)

        if isinstance(bitcoin, pd.DataFrame) and isinstance(bitcoin_stock, pd.DataFrame):
            bitcoin_l = [bitcoin_dat.query(f'month == {date.today().month} & year == {date.today().year} & day == {date.today().day - 3}').iloc[0,1]]
            bitcoin_stock_l = [bitcoin_stock_dat.query(f'month == {date.today().month} & year == {date.today().year} & day == {date.today().day - 3}').iloc[0,1]]
            data = {'local_minima_count': local_minima_count_list, 
                'local_maxima_count':local_maxima_count_list, 
                'slope': slope_list, 
                'quartile1_proportion': q1_p_list,
                'quartile2_proportion':q2_p_list,
                'quartile3_proportion':q3_p_list,
                'quartile4_proportion':q4_p_list,
                'q1_local_max':local_max_q1_list,
                'q2_local_max':local_max_q2_list,
                'q3_local_max':local_max_q3_list,
                'q4_local_max':local_max_q4_list,
                'q1_local_min':local_min_q1_list,
                'q2_local_min':local_min_q2_list,
                'q3_local_min':local_min_q3_list,
                'q4_local_min':local_min_q4_list,
                'q1_slope':slope_q1_list,
                'q2_slope':slope_q2_list,
                'q3_slope':slope_q3_list,
                'q4_slope':slope_q4_list,
                'local_minima_count_volume': local_minima_count_v_list, 
                'local_maxima_count_volume':local_maxima_count_v_list, 
                'slope_volume': slope_v_list, 
                'quartile1_proportion_volume': q1_v_list,
                'quartile2_proportion_volume':q2_v_list,
                'quartile3_proportion_volume':q3_v_list,
                'quartile4_proportion_volume':q4_v_list,
                'q1_local_max_volume':local_max_q1_v_list,
                'q2_local_max_volume':local_max_q2_v_list,
                'q3_local_max_volume':local_max_q3_v_list,
                'q4_local_max_volume':local_max_q4_v_list,
                'q1_local_min_volume':local_min_q1_v_list,
                'q2_local_min_volume':local_min_q2_v_list,
                'q3_local_min_volume':local_min_q3_v_list,
                'q4_local_min_volume':local_min_q4_v_list,
                'q1_slope_volume':slope_q1_v_list,
                'q2_slope_volume':slope_q2_v_list,
                'q3_slope_volume':slope_q3_v_list,
                'q4_slope_volume':slope_q4_v_list}

        if isinstance(bitcoin, pd.DataFrame) and isinstance(bitcoin_stock, pd.DataFrame):
            data['bitcoin'] = bitcoin_l
            data['bitcoin_stock'] = bitcoin_stock_l
        
        dat = pd.DataFrame(data)

        # Prediction over the next hour
        prediction = sc.inverse_transform(np.array(bst.predict(xgb.DMatrix(dat))).reshape(1, -1))[0]
    now1 = datetime.datetime.now(timezone.utc)

    dat = pd.DataFrame({'pred':[prediction[0]], 'utc_time_at_pred': [now1]})
    return(dat)

if __name__ == "__main__":
    for i in range(3):
        dat = main_func()
        print(dat)
        dat.to_csv('C:\\Users\\britchie\\OneDrive - AgReserves, Inc\\Documents\\stock\\results.csv', mode='a', header=False)