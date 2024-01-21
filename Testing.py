#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:54:27 2024

@author: matteo
"""

import pandas as pd
import numpy as np
import streamlit as st
import math

# raw = pd.read_excel("Time Series.xlsx")
# raw.Date = pd.to_datetime(raw.Date)
# raw.set_index("Date",drop=True,inplace=True)

raw = pd.read_pickle("Time Series.pickle")

assets = pd.read_pickle("./Assets/Assets.pickle")
risk_profiles = pd.read_pickle("./Assets/RiskProfile.pickle")

# data_df = pd.DataFrame(
#     {
#         "assets": assets,
#         "weights" : [0] * assets.shape[0]
#     }
# )


st.set_page_config(layout="wide")

st.header("EFG Asset Management - My Focus", divider=True)
st.sidebar.header("Portfolio")

risk_profile = st.sidebar.selectbox(
    'Risk Profile',
    risk_profiles.Strategy
)


currency = st.sidebar.selectbox(
    'Currency',
    ('USD',)
)

structure, composition, metrics = st.tabs(['Structure','Composition','Metrics'])


with structure:
    with st.container():
        cash = st.slider("Cash",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Cash)
        bond = st.slider("Bond",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Bond)
        equity = st.slider("Equity",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Equity)
        alternative = st.slider("Alternative",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Alternative)
            
    equity_selection = st.multiselect(
        'Pick your equity components',
        assets[(assets.Currency == currency) & (assets.Type=='Equity')])
    bond_selection = st.multiselect(
        'Pick your bond components',
        assets[(assets.Currency == currency) & (assets.Type=='Bond')])
    alternative_selection = st.multiselect(
        'Pick your alternatives',
        assets[(assets.Currency == currency) & (assets.Type=='Alternative')])

    if len(equity_selection) > 0:
        equity_df = pd.DataFrame(equity_selection)
        equity_df['Weight'] = [100/len(equity_selection)] * len(equity_selection)
        
        equity_df = st.data_editor(
            equity_df,
            column_config={
                "equity": st.column_config.SelectboxColumn(
                    "Equity",
                    help="The equity constituents you want in your portfolio",
                    width="medium",
                    options=equity_selection,
                    required=True,
                )
            },
            hide_index=True,
        )

    if len(bond_selection) > 0:
        bond_df = pd.DataFrame(bond_selection)
        bond_df['Weight'] = [100/len(bond_selection)] * len(bond_selection)
    
        bond_df = st.data_editor(
            bond_df,
            column_config={
                "bond": st.column_config.SelectboxColumn(
                    "Bond",
                    help="The bond constituents you want in your portfolio",
                    width="medium",
                    options=bond_selection,
                    required=True,
                )
            },
            hide_index=True,
        )

    if len(alternative_selection) > 0:
        alternative_df = pd.DataFrame(alternative_selection)
        alternative_df['Weight'] = [100/len(alternative_selection)] * len(alternative_selection)
        
        alternative_df = st.data_editor(
            alternative_df,
            column_config={
                "alternative": st.column_config.SelectboxColumn(
                    "Alternative",
                    help="The alternative constituents you want in your portfolio",
                    width="medium",
                    options=alternative_selection,
                    required=True,
                )
            },
            hide_index=True,
        )

   
    if st.button('Calc'):
        weights = data_df.weights /100    
        
        assets = data_df.assets
        weights.index = assets
        # st.write(weights)
        
        # calculate the time series
        ts = raw[assets].copy()
        ret = pd.Series(np.dot(ts,weights),index=ts.index,name='Pfolio')
        pfolio = ret.rolling(window=ret.shape[0],min_periods=1).sum() + 1

        with metrics:        
            # plot the total return
            st.line_chart(data=pfolio)
            
            # get some risk metrics
            metrics = pd.Series(dtype=float,name='Metrics')
            metrics['Annual Return'] = ret.mean()*260
            metrics['Risk'] = ret.std()*math.sqrt(260)
            metrics = metrics * 100
            metrics = metrics.round(decimals=2)
            
            st.write(metrics)
        
