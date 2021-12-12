#%%
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
from jmspack.NLTSA import (distribution_uniformity, 
                           fluctuation_intensity, 
                           complexity_resonance, 
                           cumulative_complexity_peaks)
from jmspack.utils import apply_scaling
import seaborn as sns

#%%
def request_mobility_data_url():
    url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
    response = urlopen(url)
    data = json.loads(response.read())
    url = ("https://covid19-static.cdn-apple.com" + data['basePath'] + data['regions']['en-us']['csvPath'])
    return url

df = pd.read_csv(request_mobility_data_url())

#%%
prep_df = (df.drop(['geo_type',
 'alternative_name',
 'sub-region'], axis=1)
           .replace(np.nan, " ")
 .set_index(["country", "region", "transportation_type"])
 .T
 )

prep_df.columns = ['-'.join(col).strip() for col in prep_df.columns.tolist()]
country_list = prep_df.columns.tolist()
# [{'label': i.title().replace("_", " "), 'value': i} for i in country_list]

#%%
plot_df=(prep_df
                              .filter(regex="Finland")      
                              .replace(" ", np.nan)
                              .dropna(thresh=10, axis=1).dropna(axis=0)
                              .pipe(apply_scaling))
fi_df = fluctuation_intensity(df=plot_df, 
                      win=7, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=plot_df.shape[1])
# %%
du_df = distribution_uniformity(df=plot_df, 
                      win=7, 
                      xmin=0, 
                      xmax=1, 
                      col_first=1, 
                      col_last=plot_df.shape[1])
# %%
cr_df = complexity_resonance(distribution_uniformity_df=du_df, 
                             fluctuation_intensity_df=fi_df)
# %%
cumulative_complexity_peaks_df, significant_peaks_df = cumulative_complexity_peaks(df=cr_df)
# %%
