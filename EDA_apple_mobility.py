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
from utils import request_mobility_data_url, summary_window_FUN
from sklearn import decomposition

#%%
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
decomps_list = [
        # decomposition.DictionaryLearning,
                    decomposition.FactorAnalysis,
                    decomposition.FastICA,
                    # decomposition.IncrementalPCA,
                    decomposition.KernelPCA,
                    decomposition.NMF,
                    decomposition.PCA
                    ]

tmp_df = (prep_df
                .filter(regex="Finland")
                .replace(" ", np.nan)
                .dropna(thresh=10, axis=1).dropna(axis=0)
                .pipe(apply_scaling)
                # .reset_index()
                # .melt(id_vars="index")
)
plot_df=pd.concat([summary_window_FUN(tmp_df.pipe(apply_scaling), window_size=5, user_func=window_function,

                                kwargs={"random_state": 42}) for window_function in decomps_list],
            axis=1).reset_index().melt(id_vars="index")
# %%
