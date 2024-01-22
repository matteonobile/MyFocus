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

# raw = pd.read_pickle("Time Series.pickle")

assets = pd.read_pickle("./Assets/Assets.pickle")
risk_profiles = pd.read_pickle("./Assets/RiskProfile.pickle")


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
        equity_df.columns = ['Asset']
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
        bond_df.columns = ['Asset']
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
        alternative_df.columns = ['Asset']
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
        portfolio_weights = pd.Series([cash,bond,equity,alternative],
                            index=['cash','bond','equity','alternative'])
        
        # normalize portfolio weights
        portfolio_weights = portfolio_weights / portfolio_weights.sum()
        
        # pick the correct benchmark
        
        selected_strategy = risk_profiles.set_index('Strategy').loc[risk_profile,:]
        benchmark_weights = pd.Series([selected_strategy.Cash,
                                       selected_strategy.Bond,
                                       selected_strategy.Equity,
                                       selected_strategy.Alternative],
                                      index = portfolio_weights.index)
        benchmark_weights = benchmark_weights / benchmark_weights.sum()
        
        active_weights = portfolio_weights - benchmark_weights

        with composition:
            st.write("Portfolio Strategic Allocation vs Risk Profile")
            st.bar_chart(data = active_weights)

            full_equity = pd.DataFrame()
            equity_df.Weight = equity_df.Weight / equity_df.Weight.sum()
            for i,e in equity_df.iterrows():
                e_data = pd.read_pickle("./Assets/"+e.Asset+".pickle")
                e_data.Weight = e_data.Weight * e.Weight
                full_equity = pd.concat([full_equity,e_data])
            # st.write(full_equity)
            full_equity = full_equity.groupby(['ISIN','Descr','Currency','Country','Sector']).sum()
            full_equity = full_equity / full_equity.sum()
            full_equity = full_equity.sort_values(by='Weight',ascending=False)
            full_equity.reset_index(inplace=True)
            
            largest_equity = full_equity.iloc[:10,:]
            largest_equity = largest_equity[['Descr','Weight']]
            st.write(largest_equity)
            # st.write(full_equity)
            
            bmk_equity = pd.read_pickle("./Assets/ACWI USD.pickle")
            bmk_equity.Weight = bmk_equity.Weight / bmk_equity.Weight.sum()
            
            # Sectors
            equity_sectors = full_equity[['Sector','Weight']].groupby("Sector").sum()
            bmk_sectors = bmk_equity[['Sector','Weight']].groupby("Sector").sum()
            
            active_sectors = equity_sectors - bmk_sectors
            st.bar_chart(data=active_sectors)
            
            # Countries
            equity_country = full_equity[['Country','Weight']].groupby("Country").sum()
            bmk_country = bmk_equity[['Country','Weight']].groupby("Country").sum()

            active_country = equity_country - bmk_country
            st.bar_chart(data=active_country)
            
            
        # with metrics:        
        #     # plot the total return
        #     st.line_chart(data=pfolio)
            
        #     # get some risk metrics
        #     metrics = pd.Series(dtype=float,name='Metrics')
        #     metrics['Annual Return'] = ret.mean()*260
        #     metrics['Risk'] = ret.std()*math.sqrt(260)
        #     metrics = metrics * 100
        #     metrics = metrics.round(decimals=2)
            
        #     st.write(metrics)
        
