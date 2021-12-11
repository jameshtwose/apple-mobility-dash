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
            #   Input(component_id='feature_choice', component_property='value')
              ]
              )
def heatmap_update(region_choice):
    fig = px.imshow(prep_df.filter(regex=region_choice).T, aspect="equal")

    fig.update_layout(title='Mobility == {}'.format(region_choice),
                      xaxis_title='Day Number',
                      yaxis_title='Country-Region-Transportation_type',
                      paper_bgcolor='rgb(34, 34, 34)',
                      plot_bgcolor='rgb(34, 34, 34)',
                      font=dict(color="#FFFFFF")
                      )
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
