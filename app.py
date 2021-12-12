import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# import dash_core_components as dcc
# import dash_html_components as html

import json
from urllib.request import urlopen
import numpy as np

from jmspack.NLTSA import (distribution_uniformity, 
                           fluctuation_intensity, 
                           complexity_resonance, 
                           cumulative_complexity_peaks)
from jmspack.utils import apply_scaling


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/darkly/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

def request_mobility_data_url():
    url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
    response = urlopen(url)
    data = json.loads(response.read())
    url = ("https://covid19-static.cdn-apple.com" + data['basePath'] + data['regions']['en-us']['csvPath'])
    return url

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

app.layout = html.Div([
    html.H1(id='H1', children='Mobility Numbers', 
            style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
    html.Div([dcc.Dropdown(
            id='country_choice',
            options=[{'label': i.title().replace("_", " "), 'value': i} for i in country_list],
            value="-Netherlands-driving"
        )], style={'width': '20%', 'display': 'inline-block', "color": "#222", "padding":"5px"}),
    dcc.Graph(id='line_plot'),
    html.Div([dcc.Dropdown(
            id='region_choice',
            options=[{'label': i.title().replace("_", " "), 'value': i} for i in region_list],
            value="Netherlands"
        )], style={'width': '20%', 'display': 'inline-block', "color": "#222", "padding":"5px"}),
    html.Div([dcc.Dropdown(
            id='analysis_choice',
            options=[{'label': i.title().replace("_", " "), 'value': i} for i in analysis_list],
            value="raw"
        )], style={'width': '20%', 'display': 'inline-block', "color": "#222", "padding":"5px"}),
    # html.Div(id='display-value')
    dcc.Graph(id="heatmap_plot", style={"height": "1000px"}),
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


@app.callback(Output(component_id='heatmap_plot', component_property='figure'),
              [Input(component_id='region_choice', component_property='value'),
              Input(component_id='analysis_choice', component_property='value')
              ]
              )
def heatmap_update(region_choice, analysis_choice):
    if analysis_choice == "raw":
        plot_df = prep_df.filter(regex=region_choice)
    elif analysis_choice == "fluctuation intensity":
        tmp_df=(prep_df
                .filter(regex=region_choice)
                .replace(" ", np.nan)
                .dropna(thresh=10, axis=1).dropna(axis=0)
                .pipe(apply_scaling))
        plot_df = fluctuation_intensity(df=tmp_df, 
                      win=7, 
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
                      win=7, 
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
                      win=7, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
        du_df = distribution_uniformity(df=tmp_df, 
                      win=7, 
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
                      win=7, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=tmp_df.shape[1])
        du_df = distribution_uniformity(df=tmp_df, 
                      win=7, 
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

    fig.update_layout(title=f'Mobility == {region_choice}, Analysis == {analysis_choice}',
                      xaxis_title='Day Number',
                      yaxis_title='Country-Region-Transportation_type',
                      paper_bgcolor='rgb(34, 34, 34)',
                      plot_bgcolor='rgb(34, 34, 34)',
                      font=dict(color="#FFFFFF")
                      )
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
