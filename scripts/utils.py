import numpy as np
import pandas as pd
import datetime

#function to generate whether a particular home is listed and sold within a certain time period
def gen_y(t_disc, data, t0=None):
    ''' 
    t_disc: datetime.timedelta(days = XX)
    t0: datetime.datetime(YYYY,MM,DD)
    '''

    if t0 is not None:
        listed = np.array(((data['list_date'] >= t0) & (data['list_date'] < t0 + t_disc)) | ((data['list_date'] < t0) & (data['sale_date'] >= t0)), dtype=np.int8)
        sale = np.array((data['sale_date'] >= t0) & (data['sale_date'] < t0 + t_disc), dtype = np.int8)
        return np.vstack((listed, sale)).T


def gen_x(diff, data, t0):
    start = t0
    end = t0 + diff
    return data[(data.sale_date >= start) & (data.list_date < end)]


def gen_dataset(df, t0, diff):
    t0 = pd.to_datetime(t0, format='%Y-%m-%d')
    diff = datetime.timedelta(days = diff)
    
    tmp = gen_x(diff, df, t0)
    
    tp = {str(k): list(v) for k, v in tmp.groupby(tmp.dtypes, axis=1)}
    
    X = pd.concat([tmp[tp['float64']], tmp[tp['bool']], tmp[tp['int64']]], axis=1)
    y = gen_y(diff, tmp, t0)
    
    return {
        'X': X,
        'y': y[:,1].reshape(-1,1),
        'float': tp['float64'],
        'bool': tp['bool'],
        'int': tp['int64']
    }