import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import pickle
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score


def plot_speed(data,time=1):

    d = data[data.time == time]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(y=d['y_speed'], name='Estimated Current Speed')
    )
    fig.add_trace(
        go.Scatter(y=d['speed'], name='Actual Current Speed')
    )
    fig.update_layout(
        title='Current Speed - Estimated vs Actual',
        xaxis_title='index',
        yaxis_title='m/s'
    )
    fig.update_layout(width=800, height=800)
    return fig

def metrics_speed(data,time=1):
    d = data[data.time == time]
    mae = mean_absolute_error(d['y_speed'], d['speed'])
    mse = sqrt(mean_squared_error(d['y_speed'], d['speed']))
    r2 = r2_score(d['y_speed'], d['speed'], multioutput='variance_weighted')

    return r2,mse,mae

def cb_speed(x,y,data):
    with open('model_pickle_speed', 'rb') as f:
        model_speed = pickle.load(f)

    y_model = model_speed.predict(x)
    Results = y.copy()
    Results["Estimated_Speed"] = y_model
    D = pd.merge(data,Results,left_index=True,right_index=True)
    D["inv_es"] = np.expm1(D.Estimated_Speed)
    return D



