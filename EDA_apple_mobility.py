#%%
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
from jmspack.NLTSA import fluctuation_intensity
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

#%%
# fluctuation_intensity(df=prep_df, win=7)
prep_df.filter(regex="Netherlands").astype(float).pipe(apply_scaling)
# %%
