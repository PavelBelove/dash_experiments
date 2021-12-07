import dash
# import dash_core_components as dcc
from dash import dcc, html, State
# import dash_html_components as html
# from dash import html
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
PLAYERS = 1 #
GAMES = 20 #

SEED_MONEY = 100

WIN_RATE = 1.5
LOSS_RATE = 0.6

# colors
plot_bgcolor = '#52575c'
paper_bgcolor = '#3a3f44' #'#52575c'
font_color = '#aaa'

text_markdown = "\t\n\n"
with open('erg_info.md') as file:
    for a in file.read():
        if "\n" in a:
            text_markdown += "\n \t"
        else:
            text_markdown += a

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
        dbc.Col([
             dbc.Button(
                    "Чо?",
                    id="collapse-button",
                    className="mb-3 mt-3",
                    color="primary",
                    n_clicks=0,
                )
        ], width={'size':1}), 
        dbc.Col([
            html.H1('Эргодичность', className='text-center mb-4 mt-3'),
        ], width={'size':6}), 
        dbc.Col([], width={'size':1})
    ], justify='center'),
    dbc.Row([   
        dbc.Col([
            html.Div([
                dbc.Collapse(                    
                    dbc.Card([
                        dcc.Markdown(text_markdown),
                        html.Details([
                            html.Div([
                                dcc.Markdown('### Bla-Bla-Bla &copy Greta')
                            ])            
                        ],
            id='details',
            title='Описание симуляции',
            # className='text-center'
            ),
                        ]
                    , className='p-4'),
                    id="collapse",
                    is_open=False,
                ),
            ]
        ),
        ], width={'size':8}),
    ], justify='center'),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.P('Количество игроков'),
            dcc.Slider(
                id='players',
                min=10,
                max=1000,
                marks={10: '10', **{i: str(i) for i in range(100, 1001, 100)}},
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
                {'label': '  Случайных', 'value': 'rand'},
                {'label': '  Все графики', 'value': 'all'}
            ],
            value='first',
            style={},
            labelStyle={'display': 'inline-block', 'margin': '1rem'},
            ),
            html.P('Пресеты:'),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='preset',
                        options=[
                            {'label': 'Успешный успех', 'value': 'success'},
                            {'label': 'Рынок порешает', 'value': 'market'},
                            {'label': 'Коммунизм', 'value': 'communism'},
                            {'label': '(не)Оправданный риск', 'value': 'risk'},
                            {'label': 'Взлеты и падения', 'value': 'story'},
                            {'label': 'Процветание', 'value': '25'},
                        ]),
                ]),
                dbc.Col([
                    # dbc.Button(
                    # "Перезапуск симуляции",
                    #     id="new",
                    #     className="",
                    #     color="primary",
                    #     n_clicks=0,
                    # ),
                ]),
            ])

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

# Пресеты и перезапуск

@app.callback(
    Output('players', 'value'),
    Output('games', 'value'),
    Output('rate', 'value'),
    Output('select_graph', 'value'),
    Input('preset', 'value'),
    # Input('new', 'n_clicks'),     
)
def set_preset(preset):
    players, games, rate, graph = 100, 50, 100, 'rand'

    if preset == 'success':
        players, games, rate, graph = 1000, 10, 100, 'lucky'
    elif preset == 'market':
        players, games, rate, graph = 200, 50, 100, 'rand'
    elif preset == 'communism':
        players, games, rate, graph = 200, 50, 5, 'rand'
    elif preset == 'risk':
        players, games, rate, graph = 10, 100, 100, 'first'
    elif preset == '25':
        players, games, rate, graph = 1000, 100, 25, 'lucky'
    elif preset == 'story':
        players, games, rate, graph = 1000, 100, 100, 'all'

    return players, games, rate, graph

# @app.callback(
#     Output('players', 'value'),
#     # Output('games', 'value'),
#     # Output('rate', 'value'),
#     # Output('select_graph', 'value'),
#     Input('preset', 'value'),
#     Input('players', 'value'),
#     # Input('new', 'n_clicks'),     
# )
# def reset(n, players):
#     return players

# Генерируем данные

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
    for i in range(players):
        x = SEED_MONEY
        for j in range(games):
            d.append({"players": i, "steps": j, "money": x})

            if np.random.randint(2):
                x = x * (1-rate) + x * rate * WIN_RATE
            else:
                x = x * (1-rate) + x * rate * LOSS_RATE
    df = pd.DataFrame(d)
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
    elif radio == 'rand':        
        dff = df[df['players'].isin(np.random.choice(df['players'].unique(), n_graph, replace=False))]
    else:
        dff = df

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


# capital allocation
#******************************************************************

@app.callback(
    Output('hist-money', 'figure'),
    Input('data', 'data'),
    Input('rate', 'value')
)
def select_graph(data, rate):

    rate = rate / 100


    df = pd.read_json(data, orient='split')
    df = df[df['steps'] == 9]
    # df = df[df['steps'] == df['steps'].max()]

    # # генерирование выборки бОльшего объема
    # players = 10000
    # games = 6
    # d = []
    # arifm_coef = 1 + (WIN_RATE + LOSS_RATE - 2) * rate / 2
    # geom_coef = math.sqrt((1 + (WIN_RATE-1) * rate) * (1 + (LOSS_RATE-1) * rate))
    # for i in range(players):
    #     x = SEED_MONEY
    #     for j in range(games):
    #         if np.random.randint(2):
    #             x = x * (1-rate) + x * rate * WIN_RATE
    #         else:
    #             x = x * (1-rate) + x * rate * LOSS_RATE
    #     d.append({"players": i, "money": x})
    # df = pd.DataFrame(d)


    df = df.round({'money':2})
    gdf = df.groupby('money').count()
    gdf.reset_index(level=0, inplace=True)
    mean_money = df['money'].sum() /df['players'].count() 

    fig = px.line(gdf, x='money', y='players', line_shape='spline')

    fig.add_shape( # Вертикаль средние деньги
    type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot",
    x0=mean_money, x1=mean_money, xref="x", y0=0, y1=gdf['players'].max(), yref="y")
    fig.add_annotation( 
    text="Average payoff ", textangle=-90, x=mean_money, y=(gdf['players'].max() / 2))

    fig.add_shape( # Вертикаль стартовый капитал
    type="line", line_color="yellow", line_width=3, opacity=1, line_dash="dash",
    x0=SEED_MONEY, x1=SEED_MONEY, xref="x", y0=0, y1=gdf['players'].max(), yref="y")
    fig.add_annotation( 
    text="Starting value", textangle=-90, x=SEED_MONEY, y=(gdf['players'].max() / 2))
    fig.update_layout(plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor,
                        modebar_remove=["autoscale", "zoomin", "zoomout"],
                        font = dict (color = font_color))
    fig.update_traces(fill='tozeroy',line={'color':'blue'})
    return fig

# collapse
# *********************************************************************

@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open




if __name__ == '__main__':
    app.run_server(debug=False)