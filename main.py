import streamlit as st
from functions import *
import pandas as pd
import datetime as dt
import plotly.express as px

st.title('Current Prediction Dashboard')
st.subheader('CatBoost Model')
@st.cache
def get_data():
    fp = 'h2_final.csv'
    df = pd.read_csv(fp)
    return df

data = get_data()

# Sidebar
df = st.sidebar.selectbox('Select a dataset',['Dataset 2020', 'Dataset 2020-2021'])
if df == 'Dataset 2020':
    date = st.sidebar.date_input('Select a date',value=dt.date(2020,10,20),
                                 min_value = dt.date(2020,10,20),max_value = dt.date(2020,10,21))
else:
    date = st.sidebar.date_input('Select a date', value=dt.date(2021, 10, 21),
                                 min_value=dt.date(2021, 10, 21), max_value=dt.date(2021, 10, 21))

time = st.sidebar.selectbox('Select the number of hours to predict',data['time'].unique())


X_test = data[['lat', 'lon', 'day_of_year']]
y_test = data[['y_speed', 'y_direction']]


col1, col2, col3 = st.columns(3)
col1.metric("R^2", round(metrics_speed(data,time)[0],3))
col2.metric("MSE", round(metrics_speed(data,time)[1],3))
col3.metric("MAE", round(metrics_speed(data,time)[2],3))


st.plotly_chart(plot_speed(data,time))
############# Direction


fig_geo = px.scatter_geo(data,lat='lat',lon='lon', hover_name='speed', color = 'speed', size_max=10)
fig_geo.update_layout(title='Survey Area - Current Speed')
fig_geo.update_layout(width=800, height=800)

st.plotly_chart(fig_geo)


def plot_model(data):


    fig = px.scatter_3d(data,
                    x='lon',
                    y='lat',
                    z='speed',
                    color='direction',)


    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(marker_size=3)
    fig.update_layout(width=800, height=800)
    return fig


st.plotly_chart(plot_model(data))









