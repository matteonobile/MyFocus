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
            st.write("Largest Equity Positions")
            st.write(largest_equity)
            # st.write(full_equity)
            
            bmk_equity = pd.read_pickle("./Assets/ACWI USD.pickle")
            bmk_equity.Weight = bmk_equity.Weight / bmk_equity.Weight.sum()
            
            # Sectors
            equity_sectors = full_equity[['Sector','Weight']].groupby("Sector").sum()
            equity_sectors.columns = ['Port']
            bmk_sectors = bmk_equity[['Sector','Weight']].groupby("Sector").sum()
            bmk_sectors.columns = ['Bmk']
            
            all_sectors = equity_sectors.join(bmk_sectors,how="outer")
            all_sectors.fillna(0,inplace=True)
            active_sectors = all_sectors['Port'] - all_sectors['Bmk']
            active_sectors.sort_values(ascending=False)
            st.write("Sector Active Allocation")
            st.bar_chart(data=active_sectors)
            
            # Countries
            equity_country = full_equity[['Country','Weight']].groupby("Country").sum()
            equity_country.columns = ['Port']
            bmk_country = bmk_equity[['Country','Weight']].groupby("Country").sum()
            bmk_country.columns = ['Bmk']
            all_country = equity_country.join(bmk_country,how='outer')
            all_country.fillna(0,inplace=True)
            active_country = all_country['Port'] - all_country['Bmk']
            active_country = active_country.astype(float)
            active_country = active_country.sort_values(ascending=False)
            active_country = pd.concat([active_country.iloc[:5],active_country.iloc[-5:]])
            st.write("Country Active Allocation (10 largest deviations")
            st.bar_chart(data=active_country,x=None)

#####################################################################
            # let's go for bonds now            
            full_bond = pd.DataFrame()
            bond_df.Weight = bond_df.Weight / bond_df.Weight.sum()
            for i,b in bond_df.iterrows():
                b_data = pd.read_pickle("./Assets/"+b.Asset+".pickle")
                b_data.Weight = b_data.Weight * e.Weight
                full_bond = pd.concat([full_bond,b_data])
            # st.write(full_equity)
            full_bond = full_bond.groupby(['ISIN','Descr','Currency','Country','Sector','Rating','Duration','Yield']).sum()
            full_bond = full_bond / full_bond.sum()
            full_bond = full_bond.sort_values(by='Weight',ascending=False)
            full_bond.reset_index(inplace=True)
            
            largest_bond = full_bond.iloc[:10,:].copy()
            largest_bond = largest_bond[['Descr','Rating','Duration','Yield','Weight']]
            st.write("Largest Bond Positions")
            st.write(largest_bond)
            # st.write(full_equity)
            
            bmk_bond = pd.read_pickle("./Assets/EuroDollars USD.pickle")
            bmk_bond.Weight = bmk_bond.Weight / bmk_bond.Weight.sum()
            
            
            # duration bucket allocation
            
            def duration_bucket(df):
                df['Bucket'] = '1+'
                df.loc[(df.Duration>=0) & (df.Duration<3),'Bucket'] = '0-3'
                df.loc[(df.Duration>=3) & (df.Duration<5),'Bucket'] = '3-5'
                df.loc[(df.Duration>=5) & (df.Duration<7),'Bucket'] = '5-7'
                df.loc[(df.Duration>=7) & (df.Duration<10),'Bucket'] = '7-10'
                df.loc[(df.Duration>=10),'Bucket'] = '10+'
                return df
            
            full_bond = duration_bucket(full_bond)
            
            bmk_bond = duration_bucket(bmk_bond)
            # st.write(bmk_bond)
            # Duration Bucket
            bond_duration = full_bond[['Bucket','Weight']].groupby("Bucket").sum()
            bond_duration.columns = ['Port']
            bmk_duration = bmk_bond[['Bucket','Weight']].groupby("Bucket").sum()
            bmk_duration.columns = ['Bmk']
            all_duration = bond_duration.join(bmk_duration,how='outer')
            all_duration.fillna(0,inplace=True)
            
            active_duration = all_duration['Port'] - all_duration['Bmk']
            active_duration = active_duration.astype(float)
            active_duration = active_duration.sort_values(ascending=False)
            # active_duration = pd.concat([active_duration.iloc[:5],active_duration.iloc[-5:]])
            st.write("Active Duration Exposure")
            st.bar_chart(data=active_duration,x=None)

            # rating allocation
            bond_rating = full_bond[['Rating','Weight']].groupby("Rating").sum()
            bond_rating.columns = ['Port']
            bmk_rating = bmk_bond[['Rating','Weight']].groupby("Rating").sum()
            bmk_rating.columns = ['Bmk']
            all_rating = bond_rating.join(bmk_rating,how='outer')
            all_rating.fillna(0,inplace=True)
            
            active_rating = all_rating['Port'] - all_rating['Bmk']
            active_rating = active_rating.astype(float)
            active_rating = active_rating.sort_values(ascending=False)
            # active_duration = pd.concat([active_duration.iloc[:5],active_duration.iloc[-5:]])
            st.write("Active Rating Exposure")
            st.bar_chart(data=active_rating,x=None)

            # country allocation
            bond_country = full_bond[['Country','Weight']].groupby("Country").sum()
            bond_country.columns = ['Port']
            bmk_country = bmk_bond[['Country','Weight']].groupby("Country").sum()
            bmk_country.columns = ['Bmk']
            all_country = bond_country.join(bmk_country,how='outer')
            all_country.fillna(0,inplace=True)
            
            active_country = all_country['Port'] - all_country['Bmk']
            active_country = active_country.astype(float)
            active_country = active_country.sort_values(ascending=False)
            active_country = pd.concat([active_country.iloc[:5],active_country.iloc[-5:]])
            st.write("Active Country Exposure")
            st.bar_chart(data=active_country,x=None)

            # sector allocation
            bond_sector = full_bond[['Sector','Weight']].groupby("Sector").sum()
            bond_sector.columns = ['Port']
            bmk_sector = bmk_bond[['Sector','Weight']].groupby("Sector").sum()
            bmk_sector.columns = ['Bmk']
            all_sector = bond_sector.join(bmk_sector,how='outer')
            all_sector.fillna(0,inplace=True)
            
            active_sector = all_sector['Port'] - all_sector['Bmk']
            active_sector = active_sector.astype(float)
            active_sector = active_sector.sort_values(ascending=False)
            # active_sector = pd.concat([active_sector.iloc[:5],active_sector.iloc[-5:]])
            st.write("Active Sector Exposure")
            st.bar_chart(data=active_sector,x=None)
            
            
            
            
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
        
