import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# import dash_core_components as dcc
# import dash_html_components as html

import numpy as np

from jmspack.NLTSA import (distribution_uniformity, 
                           fluctuation_intensity, 
                           complexity_resonance, 
                           cumulative_complexity_peaks)
from jmspack.utils import apply_scaling
from utils import request_mobility_data_url, summary_window_FUN
from sklearn import decomposition


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/darkly/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


df = pd.read_csv(request_mobility_data_url())

prep_df = (df.drop(['geo_type',
 'alternative_name',
 'sub-region'], axis=1)
           .replace(np.nan, " ")
 .set_index(["country", "region", "transportation_type"])
 .T
 )

prep_df.columns = ['-'.join(col).strip() for col in prep_df.columns.tolist()]


country_list = prep_df.columns.tolist()
region_list = df["region"].unique().tolist()
analysis_list = ["raw", 
                 "minmax scaled", 
                 "fluctuation intensity", 
                 "distribution uniformity",
                 "complexity resonance",
                 "cumulative complexity peaks"]
window_size_list = [{"label": "week", "value": 7},
                    {"label": "fortnight", "value": 14},
                    {"label": "month", "value": 28}
                    ]

app.layout = html.Div([
    html.H1(id='H1', children='Mobility Numbers', 
            style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
    html.Div([dcc.Dropdown(
        id='region_choice',
        options=[{'label': i.title().replace("_", " "), 'value': i} for i in region_list],
        value="Portugal"
    )], style={'width': '20%', 'display': 'inline-block', "color": "#222", "padding": "5px"}),
    dcc.Graph(id='multi_line_plot'),
    html.P(id='P_note', children='''INFO: to hide lines click the marker in the legend''',
                style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 10}),
    html.Div([dcc.Dropdown(
        id='window_choice',
        options=window_size_list,
        value=7,
    )], style={'width': '20%', 'display': 'inline-block', "color": "#222", "padding": "5px"}),  
    dcc.Graph(id='multi_line_NLTSA_plot'),
    html.Div([
        html.H5(id='H5', children='''Due to the memory quota restriction imposed by Heroku, countries with more than 10 
                                            rows will not run when requesting any of the following methods:''',
                style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 10}),
        html.H5(id='H5_2', children=f"{analysis_list[2:]}",
                style={'textAlign': 'center', 'marginTop': 10, 'marginBottom': 40})
        ]),
    html.Div([dcc.Dropdown(
            id='analysis_choice',
            options=[{'label': i.title().replace("_", " "), 'value': i} for i in analysis_list],
            value="raw"
        )], style={'width': '20%', 'display': 'inline-block', "color": "#222", "padding":"5px"}),
    # html.Div(id='display-value')
    dcc.Graph(id="heatmap_plot", style={"height": "800px"}),
    # html.Div([dcc.Dropdown(
    #         id='country_choice',
    #         options=[{'label': i.title().replace("_", " "), 'value': i} for i in country_list],
    #         value="-Portugal-driving"
    #     )], style={'width': '20%', 'display': 'inline-block', "color": "#222", "padding":"5px"}),
    # dcc.Graph(id='line_plot'),
    html.Div(html.A(children="Created by James Twose",
                    href="https://services.jms.rocks",
                    style={'color': "#743de0"}),
                    style = {'textAlign': 'center',
                             'color': "#743de0",
                             'marginTop': 40,
                             'marginBottom': 40})
]
)


@app.callback(Output(component_id='line_plot', component_property='figure'),
              [Input(component_id='country_choice', component_property='value'),
            #   Input(component_id='feature_choice', component_property='value')
              ]
              )
def graph_update(country_choice):

    plot_df = prep_df.reset_index().loc[:, ["index", country_choice]]
    fig = px.line(
        data_frame=plot_df,
        x='index',
        y=country_choice,
        markers=True
    )
    fig.update_traces(line_color='#743de0')

    fig.update_layout(title='Mobility == {}'.format(country_choice),
                      xaxis_title='Date',
                      yaxis_title='{}'.format(country_choice),
                      paper_bgcolor='rgb(34, 34, 34)',
                          plot_bgcolor='rgb(34, 34, 34)',
                          template="plotly_dark",
                      )
    return fig


@app.callback(Output(component_id='multi_line_plot', component_property='figure'),
              [Input(component_id='region_choice', component_property='value'),
              ]
              )
def graph_update_multi(region_choice):

    plot_df = (prep_df
                   .filter(regex=region_choice)
                   .reset_index()
                   .melt(id_vars="index")
    )
    fig = px.line(
        data_frame=plot_df,
        x='index',
        y="value",
        color="variable",
        markers=True
    )
    # fig.update_traces(line_color='#743de0')

    fig.update_layout(title='Mobility == {}'.format(region_choice),
                      xaxis_title='Date',
                      yaxis_title='{}'.format(region_choice),
                      paper_bgcolor='rgb(34, 34, 34)',
                          plot_bgcolor='rgb(34, 34, 34)',
                          template="plotly_dark",
                      )
    return fig


@app.callback(Output(component_id='multi_line_NLTSA_plot', component_property='figure'),
              [Input(component_id='region_choice', component_property='value'),
               Input(component_id='window_choice', component_property='value'),
              ]
              )
def graph_update_multi_NLTSA(region_choice, window_choice):

    decomps_list = [decomposition.DictionaryLearning,
                    # decomposition.FactorAnalysis,
                    # decomposition.FastICA,
                    # decomposition.IncrementalPCA,
                    # decomposition.KernelPCA,
                    decomposition.NMF,
                    decomposition.PCA
                    ]

    tmp_df = (prep_df
                   .filter(regex=region_choice)
                   .replace(" ", np.nan)
                   .dropna(thresh=10, axis=1).dropna(axis=0)
                   .pipe(apply_scaling)
                   # .reset_index()
                   # .melt(id_vars="index")
    )
    plot_df=pd.concat([summary_window_FUN(tmp_df.pipe(apply_scaling), window_size=window_choice, user_func=window_function,
                                          kwargs={"random_state": 42}) for window_function in decomps_list],
                      axis=1).reset_index().melt(id_vars="index")
    fig = px.line(
        data_frame=plot_df,
        x='index',
        y="value",
        color="variable",
        markers=False
    )
    # fig.update_traces(line_color='#743de0')

    fig.update_layout(title=f'Mobility == {region_choice}, Window Size == {window_choice}',
                      xaxis_title='Day Number',
                      yaxis_title='Scaled Value',
                      paper_bgcolor='rgb(34, 34, 34)',
                          plot_bgcolor='rgb(34, 34, 34)',
                          template="plotly_dark",
                      )
    return fig


@app.callback(Output(component_id='heatmap_plot', component_property='figure'),
              [Input(component_id='region_choice', component_property='value'),
              Input(component_id='analysis_choice', component_property='value'),
              Input(component_id='window_choice', component_property='value')
              ]
              )
def heatmap_update(region_choice, analysis_choice, window_choice):
    if analysis_choice == "raw":
        plot_df = prep_df.filter(regex=region_choice)
    elif analysis_choice == "fluctuation intensity":
        tmp_df=(prep_df
                .filter(regex=region_choice)
                .replace(" ", np.nan)
                .dropna(thresh=10, axis=1).dropna(axis=0)
                .pipe(apply_scaling))
        plot_df = fluctuation_intensity(df=tmp_df, 
                      win=window_choice, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
    elif analysis_choice == "distribution uniformity":
        tmp_df=(prep_df
                .filter(regex=region_choice)
                .replace(" ", np.nan)
                .dropna(thresh=10, axis=1).dropna(axis=0)
                .pipe(apply_scaling))
        plot_df = distribution_uniformity(df=tmp_df, 
                      win=window_choice, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
    elif analysis_choice == "complexity resonance":
        tmp_df=(prep_df
                .filter(regex=region_choice)
                .replace(" ", np.nan)
                .dropna(thresh=10, axis=1).dropna(axis=0)
                .pipe(apply_scaling))
        fi_df = fluctuation_intensity(df=tmp_df, 
                      win=window_choice, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
        du_df = distribution_uniformity(df=tmp_df, 
                      win=window_choice, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
        plot_df = complexity_resonance(distribution_uniformity_df=du_df,
                                       fluctuation_intensity_df=fi_df)
    elif analysis_choice == "cumulative complexity peaks":
        tmp_df=(prep_df
            .filter(regex=region_choice)
            .replace(" ", np.nan)
            .dropna(thresh=10, axis=1).dropna(axis=0)
            .pipe(apply_scaling))
        fi_df = fluctuation_intensity(df=tmp_df, 
                      win=window_choice, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
        du_df = distribution_uniformity(df=tmp_df, 
                      win=window_choice, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
        cr_df = complexity_resonance(distribution_uniformity_df=du_df,
                                       fluctuation_intensity_df=fi_df)
        plot_df, _ = cumulative_complexity_peaks(df=cr_df)
    else:
        tmp_df=(prep_df
                .filter(regex=region_choice)
                .replace(" ", np.nan)
                .pipe(apply_scaling))
        plot_df=tmp_df
    
    fig = px.imshow(plot_df.T, aspect="equal")

    fig.update_layout(title=f'Mobility == {region_choice}, Analysis == {analysis_choice}, Window Size == {window_choice}',
                      xaxis_title='Day Number',
                      yaxis_title='Country-Region-Transportation_type',
                      paper_bgcolor='rgb(34, 34, 34)',
                      plot_bgcolor='rgb(34, 34, 34)',
                      font=dict(color="#FFFFFF")
                      )
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
