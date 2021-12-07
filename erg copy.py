import dash
# import dash_core_components as dcc
from dash import dcc
# import dash_html_components as html
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import math

# Постоянные
PLAYERS = 10 #
GAMES = 20 #

SEED_MONEY = 100

WIN_RATE = 1.5
LOSS_RATE = 0.6

# colors
plot_bgcolor = '#52575c'
paper_bgcolor = '#3a3f44' #'#52575c'
font_color = '#aaa'

d = []
for i in range(PLAYERS):
    x = SEED_MONEY
    for j in range(GAMES):
        d.append({"players": i, "steps": j, "money": x})

        if np.random.randint(2):
            x *= WIN_RATE
        else:
            x *= LOSS_RATE
df = pd.DataFrame(d)

# https://www.bootstrapcdn.com/bootswatch/ -- Bootstrap темы
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
            meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
            )

fig = px.line(df, x='steps', y='money', color='players')

# Layout section: Bootstrap 
# (https://hackerthemes.com/bootstrap-cheatsheet/) -- Bootstrap компоненты (шпаргалка)
# ************************************************************************

app.layout = dbc.Container([
    dbc.Row([
        html.H1('Эргодичность', className='text-center mb-4'),
        html.Details([html.P('span')],
            id='details',
            title='title',
            className='text-center'
            ),
        dbc.Col([
            html.P('Количество игроков'),
            dcc.Slider(
                id='players',
                min=10,
                max=100,
                marks={i: str(i) for i in range(10, 101, 10)},
                value=50,
            ),
            html.P('Количество раундов игры'),
            dcc.Slider(
                id='games',
                min=10,
                max=100,
                marks={i: str(i) for i in range(10, 101, 10)},
                value=50,
            ),
            html.P('Ставка % от депозита'),
            dcc.Slider(
                id='rate',
                min=0,
                max=100,
                marks={i: str(i) for i in range(0, 101, 10)},
                value=100,
            ),
            html.P('Показывать графики десяти'),
            dcc.RadioItems(
                id='select_graph',
            options=[
                {'label': '  Первых', 'value': 'first'},
                {'label': '  Везунчиков', 'value': 'lucky'},
                {'label': '  Лузеров', 'value': 'loser'},
                {'label': '  Случайных', 'value': 'rand'}
            ],
            value='first',
            style={'margin': 10, 'flex': 1},
            ),
            

        ], width={'size':5}
        ),

        dbc.Col([

            dcc.Graph(id='line-money', figure={}, config={'displaylogo': False})
        ], width={'size':5}
        ),
    ], justify='center'),
    html.Br(),
    dbc.Row([                
        dbc.Col([
            dcc.Graph(id='mean-money', figure={}, config={'displaylogo': False})
        ], width={'size':5} 
        ),                
        dbc.Col([
            dcc.Graph(id='hist-money', figure={}, config={'displaylogo': False})
        ], width={'size':5}
        ),
    ], justify='center'),

    dcc.Store(id='data')
], fluid=True)

# Callback
#****************************************************************************************

# Линейный график игроки

@app.callback(
    # Output('line-money', 'figure'),
    # Output('mean-money', 'figure'),
    Output('data', 'data'),
    Input('players', 'value'),
    Input('games', 'value'), 
    Input('rate', 'value')
)
def update_data(players, games, rate):
    d = []
    rate = rate / 100
    arifm_coef = 1 + (WIN_RATE + LOSS_RATE - 2) * rate / 2
    geom_coef = math.sqrt((1 + (WIN_RATE-1) * rate) * (1 + (LOSS_RATE-1) * rate))
    print(rate, arifm_coef, geom_coef)
    for i in range(players):
        x = SEED_MONEY
        for j in range(games):
            d.append({"players": i, "steps": j, "money": x})

            if np.random.randint(2):
                x = x * (1-rate) + x * rate * WIN_RATE
            else:
                x = x * (1-rate) + x * rate * LOSS_RATE
    df = pd.DataFrame(d)
    # print(df)    
    data_json = df.to_json(date_format='iso', orient='split')
    return data_json

# players & radio
#******************************************************************************************

@app.callback(
    Output('line-money', 'figure'),
    Input('data', 'data'),
    Input('select_graph', 'value')
)
def select_graph(data, radio):
    df = pd.read_json(data, orient='split')
    n_graph = 10
    if radio == 'first':
        dff = df[df['players'] < n_graph]
    elif radio == 'lucky':
        filter_players = df[df['steps'] == max(df['steps'])].sort_values('money', ascending=[False])[:n_graph]['players']
        dff = df[df['players'].isin(filter_players)] 
    elif radio == 'loser':
        filter_players = df[df['steps'] == max(df['steps'])].sort_values('money')[:n_graph]['players']
        dff = df[df['players'].isin(filter_players)]
    else:
        dff = df[df['players'].isin(np.random.choice(df['players'].unique(), n_graph, replace=False))] 
    

    # print(dff)
    fig = px.line(dff, x='steps', y='money', color='players')
    fig.update_layout(plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor, 
    modebar_remove=["autoscale", "zoomin", "zoomout"],
    font = dict (color = font_color), legend = dict (font = dict (color = font_color) 
                ))
    return fig

# Mean money in game
#*****************************************************************************************

@app.callback(
    Output('mean-money', 'figure'),
    Input('data', 'data'),
    Input('players', 'value'),
    Input('rate', 'value')
)
def update_graph(data, players, rate):
    df = pd.read_json(data, orient='split')

    rate = rate / 100
    arifm_coef = 1 + (WIN_RATE + LOSS_RATE - 2) * rate / 2
    geom_coef = math.sqrt((1 + (WIN_RATE-1) * rate) * (1 + (LOSS_RATE-1) * rate))

    gdf = df.groupby('steps').sum('money')/players
    gdf.reset_index(level=0, inplace=True)
    gdf['ideal'] = SEED_MONEY * (arifm_coef ** gdf['steps'])
    gdf['expected in time'] = SEED_MONEY * (geom_coef ** gdf['steps'])

    # print(gdf)
    mean_fig = go.Figure()
    mean_fig.add_trace(go.Scatter(x=gdf['steps'], y=gdf['money'],
                    mode='lines',
                    name='mean money'))
    mean_fig.update_traces(fill='tozeroy',line={'color':'blue'})
    mean_fig.add_trace(go.Scatter(x=gdf['steps'], y=gdf['ideal'],
                    mode='lines',
                    name='expected in the ensemble'))
    mean_fig.add_trace(go.Scatter(x=gdf['steps'], y=gdf['expected in time'],
                    mode='lines',
                    name='expected in time'))
    mean_fig.update_layout(plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor,
                            font = dict (color = font_color),
                            modebar_remove=["autoscale", "zoomin", "zoomout"],
                            legend = dict ( 
                                    yanchor = "top", 
                                    y = 0.95, 
                                    xanchor = "left", 
                                    x = 0.05,
                                    font = dict (color = font_color), 
))
    
    return mean_fig

@app.callback(
    Output('hist-money', 'figure'),
    # Input('data', 'data'),
    Input('rate', 'value')
)
def select_graph(rate):
    players = 10000
    games = 6
    d = []
    rate = rate / 100
    arifm_coef = 1 + (WIN_RATE + LOSS_RATE - 2) * rate / 2
    geom_coef = math.sqrt((1 + (WIN_RATE-1) * rate) * (1 + (LOSS_RATE-1) * rate))
    print(rate, arifm_coef, geom_coef)
    for i in range(players):
        x = SEED_MONEY
        for j in range(games):
            if np.random.randint(2):
                x = x * (1-rate) + x * rate * WIN_RATE
            else:
                x = x * (1-rate) + x * rate * LOSS_RATE
        d.append({"players": i, "money": x})
    df = pd.DataFrame(d)
    df = df.round({'money':2})
    gdf = df.groupby('money').count()
    gdf.reset_index(level=0, inplace=True)
    mean_money = df['money'].sum() /df['players'].count() 
    print(mean_money)
    fig = px.line(gdf, x='money', y='players', line_shape='spline')
    fig.add_shape( # Вертикаль средние деньги
    type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
    x0=mean_money, x1=mean_money, xref="x", y0=0, y1=gdf['players'].max(), yref="y")
    fig.add_annotation( # add a text callout with arrow
    text="Average payoff ", textangle=-90, x=mean_money, y=(gdf['players'].max() / 2))
    fig.update_layout(plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor,
                        modebar_remove=["autoscale", "zoomin", "zoomout"],
                        font = dict (color = font_color))
    fig.update_traces(fill='tozeroy',line={'color':'blue'})
    return fig
    # Добавить разделение по медиане по цветам



if __name__ == '__main__':
    app.run_server(debug=True)