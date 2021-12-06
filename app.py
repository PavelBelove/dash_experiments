# https://www.youtube.com/watch?v=0mfIK8zxUds&list=PLh3I780jNsiS3xlk-eLU2dpW3U-wCq4LW

import dash
# import dash_core_components as dcc
from dash import dcc
# import dash_html_components as html
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import pandas_datareader.data as web
import datetime

# start = datetime.datetime(2021, 9, 1)
# end = datetime.datetime(2021, 12, 1)

# df = web.DataReader(['AMZN', 'GOOGL', 'FB', 'PFE', 'BNTX', 'MARN'],
#                     'stooq', start=start)

# df = df.stack().reset_index()
# print(df.tail())
# df.to_csv("stooq.csv", index=False)

df = pd.read_csv('stooq.csv')

df = df.rename({'level_0': 'Date'}, axis='columns')

print(df.Date[1])

# https://www.bootstrapcdn.com/bootswatch/ -- Bootstrap темы
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
            meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
            )

# Layout section: Bootstrap 
# (https://hackerthemes.com/bootstrap-cheatsheet/) -- Bootstrap компоненты (шпаргалка)
# ************************************************************************

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Br(),
            html.H1('Панель рыночных инструментов', 
            className='text-center text-white-50 mb-4',)
            # width=12)
        ])]),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='symbol',
                multi=False,
                value='AMZN',
                options=[
                    {'label': x, 'value': x} for x in sorted(df['Symbols'].unique())
                ]),
            html.Br(),
            dcc.Graph(id='line-fig', figure={})

        ], width={'size':5, 'offset':0, 'order':1}),

        dbc.Col([
            dcc.Dropdown(
                id='symbol_multi',
                multi=True,
                value=['AMZN', 'FB'],
                options=[
                    {'label': x, 'value': x} for x in sorted(df['Symbols'].unique())
                ]),
            html.Br(),
            dcc.Graph(id='line-fig-multi', figure={})

        ], width={'size':5, 'order':2})],
    # no_gutters=False,
    justify='center' # выравнивание по горизонтали: start, center, end, between, around
    ),
    dbc.Row([
        html.P(""),
        dbc.Col([
            html.H3(
                'Выберите котировки',
                className='text-center mb-4'
            ),
            dcc.Checklist(
                id='check-symbols',
                value=[x for x in sorted(df['Symbols'].unique())],
                options=[{'label': x, 'value': x} for x in sorted(df['Symbols'].unique())],
                labelClassName="pl-3",            
            ),
            dcc.Graph(id='symbols-hist', figure={})
        ], width={'size':5}),
        
        
        dbc.Col([
            html.H3(
                'Выберите котировки',
                className='text-center mb-4'
            ),
            dcc.Checklist(
                id='check-symbols-1',
                value=['FB', 'PFE'],
                options=[{'label': x, 'value': x} for x in sorted(df['Symbols'].unique())],
                labelClassName="pl-3",            
            ),
            dcc.Graph(id='symbols-hist-1', figure={},
            # dcc.Slider(id='date_slider', value=sorted(df['Date'].unique())))
        )], width={'size':5}),
    ],
    justify='center'),

], fluid=True)

# Callback
#****************************************************************************************
# Линейный график одинарный

@app.callback(
    Output('line-fig', 'figure'),
    Input('symbol', 'value')
)
def update_graph(stock_slctd):
    dff = df[df['Symbols']==stock_slctd]
    figln = px.line(dff, x='Date', y='High')
    return figln

# Линейный график множественный выбор

@app.callback(
    Output('line-fig-multi', 'figure'),
    Input('symbol_multi', 'value')
)
def update_multi_graph(stock_slctd):
    dff =df[df['Symbols'].isin(stock_slctd)]
    figlns = px.line(dff, x='Date', y='High', color='Symbols')
    return figlns

# Гистограмма чекбоксы

@app.callback(
    Output('symbols-hist', 'figure'),
    Input('check-symbols', 'value')
)
def update_graph(stock_slctd):
    dff = df[df['Symbols'].isin(stock_slctd)]
    dff = dff[dff['Date']=='2021-11-03']
    fighist = px.histogram(dff, x='Symbols', y='Close')
    return fighist

# def update_hist(checked_box):
#     print(checked_box)
#     dff = df[df['Symbols'].isin(checked_box)],
#     dff = dff[dff['Date'] == df.Date[1]],
#     fig_hist = px.histogram(dff, x='Symbols', y='High')
#     return fig_hist


if __name__ == '__main__':
    app.run_server(debug=True)