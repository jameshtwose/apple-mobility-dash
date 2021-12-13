import json
from urllib.request import urlopen

import pandas as pd
import numpy as np
from sklearn import decomposition


def request_mobility_data_url():
    url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
    response = urlopen(url)
    data = json.loads(response.read())
    url = ("https://covid19-static.cdn-apple.com" + data['basePath'] + data['regions']['en-us']['csvPath'])
    return url


def summary_window_FUN(x: pd.DataFrame, window_size: int = 7, user_func=decomposition.PCA, kwargs: dict = {}):
    window_range = np.arange(0, len(x)-window_size, window_size)

    cp_df = pd.DataFrame()
    for window_begin in window_range:
        current_cp_df = pd.DataFrame(user_func(n_components=None, **kwargs)
                                     .fit_transform(x.iloc[window_begin: window_begin + window_size])).iloc[:, 0]
        cp_df = pd.concat([cp_df, current_cp_df])

    return cp_df.rename(columns={0: f"windowed_{user_func.__name__}"}).reset_index(drop=True)