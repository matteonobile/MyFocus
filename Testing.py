#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:54:27 2024

@author: matteo
"""

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px
import math

# raw = pd.read_excel("Time Series.xlsx")
# raw.Date = pd.to_datetime(raw.Date)
# raw.set_index("Date",drop=True,inplace=True)

# raw = pd.read_pickle("Time Series.pickle")

assets = pd.read_pickle("./Assets/Assets.pickle")
risk_profiles = pd.read_pickle("./Assets/RiskProfile.pickle")


st.set_page_config(layout="wide")

st.header("EFG Asset Management - Portfolio Construction Tool - Alpha - 20240201 10:43", divider=True)
st.sidebar.header("Portfolio")

risk_profile = st.sidebar.selectbox(
    'Risk Profile',
    risk_profiles.Strategy
)


currency = st.sidebar.selectbox(
    'Currency',
    ('USD',)
)

allocation_strategic = st.sidebar.selectbox(
    'Allocation Strategy',
    ['Back to Benchmark','Bank View','Risk Limit'])

relative = st.sidebar.toggle("Relative",value='on')


structure, composition, metrics = st.tabs(['Structure','Composition','Metrics'])


with structure:
    with st.container():
        cash = st.slider("Cash",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Cash)
        bond = st.slider("Bond",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Bond)
        equity = st.slider("Equity",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Equity)
        alternative = st.slider("Alternative",0,100,risk_profiles[risk_profiles.Strategy==risk_profile].iloc[0,:].Alternative)
            
    equity_selection = st.multiselect(
        'Pick your equity components',
        assets[(assets.Currency == currency) & (assets.Type=='Equity')],default = 'ACWI USD')
    bond_selection = st.multiselect(
        'Pick your bond components',
        assets[(assets.Currency == currency) & (assets.Type=='Bond')],default='EuroDollars USD')
    alternative_selection = st.multiselect(
        'Pick your alternatives',
        assets[(assets.Currency == currency) & (assets.Type=='Alternative')], default = 'Multi Hedge Focus USD')

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

   
    if st.sidebar.button('Calc'):
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
        if relative:
            active_weights = portfolio_weights - benchmark_weights
        else:
            active_weights = portfolio_weights



        with composition:
            
            df = px.data.tips()
            fig = px.sunburst(df, path=['day', 'time', 'sex'], values='total_bill')
            fig.show()            
            # Plot!
            st.plotly_chart(fig, use_container_width=True)
            
            
            if relative:
                st.write("Portfolio Strategic Allocation vs Risk Profile")
            else:
                st.write("Portfolio Strategic Allocation")
            st.bar_chart(data = active_weights)


            col1,col2 = st.columns(2)
            with col1:
                if equity_df.Weight.sum() > 0:
                    full_equity = pd.DataFrame()
                    equity_df.Weight = equity_df.Weight / equity_df.Weight.sum()
                    for i,e in equity_df.iterrows():
                        e_data = pd.read_pickle("./Assets/"+e.Asset+".pickle")
                        e_data.Weight = e_data.Weight * e.Weight
                        if full_equity.shape[0] == 0:
                            full_equity = e_data.copy()
                        else:
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
                    if relative:
                        active_sectors = all_sectors['Port'] - all_sectors['Bmk']
                        st.write("Sector Active Allocation")
                    else:
                        active_sectors = all_sectors['Port']
                        st.write("Sector Allocation")
                    active_sectors.sort_values(ascending=False)
                    
                    st.bar_chart(data=active_sectors)
                    
                    # Countries
                    equity_country = full_equity[['Country','Weight']].groupby("Country").sum()
                    equity_country.columns = ['Port']
                    bmk_country = bmk_equity[['Country','Weight']].groupby("Country").sum()
                    bmk_country.columns = ['Bmk']
                    all_country = equity_country.join(bmk_country,how='outer')
                    all_country.fillna(0,inplace=True)
                    if relative:
                        active_country = all_country['Port'] - all_country['Bmk']
                    else:
                        active_country = all_country['Port']
                    active_country = active_country.astype(float)
                    active_country = active_country.sort_values(ascending=False)
                    if relative:
                        active_country = pd.concat([active_country.iloc[:5],active_country.iloc[-5:]])
                        st.write("Country Active Allocation (10 largest deviations)")
                    else:
                        active_country = active_country[:10]
                        st.write("Largest Country Exposures")
                    st.bar_chart(data=active_country,x=None)

#####################################################################

            with col2:
                # let's go for bonds now            
                if bond_df.Weight.sum() > 0:
                    full_bond = pd.DataFrame()
                    bond_df.Weight = bond_df.Weight / bond_df.Weight.sum()
                    for i,b in bond_df.iterrows():
                        b_data = pd.read_pickle("./Assets/"+b.Asset+".pickle")
                        b_data.Weight = b_data.Weight * e.Weight
                        if full_bond.shape[0] == 0:
                            full_bond = b_data.copy()
                        else:
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
                    
                    if relative:
                        active_duration = all_duration['Port'] - all_duration['Bmk']
                        st.write("Active Duration Exposure")
                    else:
                        active_duration = all_duration['Port']
                        st.write("Duration Exposure")
                    active_duration = active_duration.astype(float)
                    active_duration = active_duration.sort_values(ascending=False)
                    # active_duration = pd.concat([active_duration.iloc[:5],active_duration.iloc[-5:]])
                    
                    st.bar_chart(data=active_duration,x=None)
        
                    # rating allocation
                    bond_rating = full_bond[['Rating','Weight']].groupby("Rating").sum()
                    bond_rating.columns = ['Port']
                    bmk_rating = bmk_bond[['Rating','Weight']].groupby("Rating").sum()
                    bmk_rating.columns = ['Bmk']
                    all_rating = bond_rating.join(bmk_rating,how='outer')
                    all_rating.fillna(0,inplace=True)
                    
                    if relative:
                        active_rating = all_rating['Port'] - all_rating['Bmk']
                        st.write("Active Rating Exposure")
                    else:
                        active_rating = all_rating['Port']
                        st.write("Rating Exposure")
                    active_rating = active_rating.astype(float)
                    active_rating = active_rating.sort_values(ascending=False)
                    # active_duration = pd.concat([active_duration.iloc[:5],active_duration.iloc[-5:]])
                    
                    st.bar_chart(data=active_rating,x=None)
        
                    # country allocation
                    bond_country = full_bond[['Country','Weight']].groupby("Country").sum()
                    bond_country.columns = ['Port']
                    bmk_country = bmk_bond[['Country','Weight']].groupby("Country").sum()
                    bmk_country.columns = ['Bmk']
                    all_country = bond_country.join(bmk_country,how='outer')
                    all_country.fillna(0,inplace=True)
                    
                    if relative:
                        active_country = all_country['Port'] - all_country['Bmk']
                        st.write("Active Country Exposure")
                    else:
                        active_country = all_country['Port'] 
                        st.write("Country Exposure")
                    active_country = active_country.astype(float)
                    active_country = active_country.sort_values(ascending=False)
                    if relative:
                        active_country = pd.concat([active_country.iloc[:5],active_country.iloc[-5:]])
                    else:
                        active_country = active_country[:10]
                    
                    st.bar_chart(data=active_country,x=None)
        
                    # sector allocation
                    bond_sector = full_bond[['Sector','Weight']].groupby("Sector").sum()
                    bond_sector.columns = ['Port']
                    bmk_sector = bmk_bond[['Sector','Weight']].groupby("Sector").sum()
                    bmk_sector.columns = ['Bmk']
                    all_sector = bond_sector.join(bmk_sector,how='outer')
                    all_sector.fillna(0,inplace=True)
                    
                    if relative:
                        active_sector = all_sector['Port'] - all_sector['Bmk']
                        st.write("Active Sector Exposure")
                    else:
                        active_sector = all_sector['Port']
                        st.write("Sector Exposure")
                    active_sector = active_sector.astype(float)
                    active_sector = active_sector.sort_values(ascending=False)
                    # active_sector = pd.concat([active_sector.iloc[:5],active_sector.iloc[-5:]])
                    
                    st.bar_chart(data=active_sector,x=None)
                
            
        with metrics:        
        #     # plot the total return
            # portfolio_weights
            # bond_df
            # equity_df
            # alternative_df
            
            bond_portfolio = bond_df.copy()
            bond_portfolio.Weight = bond_portfolio.Weight * portfolio_weights.bond
            equity_portfolio = equity_df.copy()
            equity_portfolio.Weight = equity_portfolio.Weight * portfolio_weights.equity
            alternative_portfolio = alternative_df.copy()
            alternative_portfolio.Weight = alternative_portfolio.Weight * portfolio_weights.alternative / 100
            cash_portfolio = pd.DataFrame([['Cash USD',0]],columns = ['Asset','Weight'])
            cash_portfolio.Weight = portfolio_weights.cash
            
            detail_portfolio = pd.concat([bond_portfolio, equity_portfolio, alternative_portfolio,cash_portfolio])
            # st.write(detail_portfolio)
            
            benchmark_portfolio = pd.DataFrame([
                ['EuroDollars USD',portfolio_weights.bond],
                ['ACWI USD',portfolio_weights.equity],
                ['Multi Hedge Focus USD',portfolio_weights.alternative],
                ['Cash USD', portfolio_weights.cash]],columns = ['Asset','Weight'])
            
            # st.write(benchmark_portfolio)
            
            # gather time series
            ts = pd.DataFrame()
            for i,l in detail_portfolio.iterrows():
                asset_ts = pd.read_pickle("./Assets/"+l.Asset+" TS.pickle")
                asset_ts.name = l.Asset
                ts = ts.join(pd.DataFrame(asset_ts),how="outer")
            ts_bmk = pd.DataFrame()
            for i,l in benchmark_portfolio.iterrows():
                asset_ts = pd.read_pickle("./Assets/"+l.Asset+" TS.pickle")
                asset_ts.name = l.Asset
                ts_bmk = ts_bmk.join(pd.DataFrame(asset_ts),how="outer")

            pfolio_ret = pd.Series(np.dot(ts,detail_portfolio.Weight),index=ts.index,name = 'Portfolio')
            benchmark_ret = pd.Series(np.dot(ts_bmk,benchmark_portfolio.Weight),index=ts_bmk.index,name = 'Benchmark')
            active_ret = pfolio_ret - benchmark_ret
            active_ret.name = 'Active'
            
            pfolio_ts = pd.DataFrame()
            pfolio_ts = pfolio_ts.join(pfolio_ret.rolling(window=pfolio_ret.shape[0],min_periods=0).sum(),how="outer")
            pfolio_ts = pfolio_ts.join(benchmark_ret.rolling(window=benchmark_ret.shape[0],min_periods=0).sum(),how="outer")
            pfolio_ts = pfolio_ts.join(active_ret.rolling(window=active_ret.shape[0],min_periods=0).sum(),how="outer")
            pfolio_ts.reset_index(inplace=True)
            pfolio_ts.columns = ['Date','Portfolio','Benchmark','Active']
            pfolio_ts.set_index("Date",inplace=True,drop=True)
            # st.write(pfolio_ts)
            st.write("Total Return")
            st.line_chart(pfolio_ts)
            
            pfolio_metrics = pd.DataFrame(columns = ['Portfolio','Benchmark'])
            pfolio_metrics = pd.DataFrame([[pfolio_ret.sum()/5,benchmark_ret.sum()/5]],
                                            columns = ['Portfolio','Benchmark'],index=['Annual Return'])
                                          

            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[pfolio_ret.std()*math.sqrt(52),benchmark_ret.std()*math.sqrt(52)]],
                                            columns = ['Portfolio','Benchmark'],index=['Risk'])
                                            ])

            risk_free = ts_bmk['Cash USD'].sum() / 5

            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[((pfolio_ret.sum() / 5) - risk_free) / (pfolio_ret.std()*math.sqrt(52)),
                                              ((benchmark_ret.sum() / 5) - risk_free) / (benchmark_ret.std()*math.sqrt(52))]],
                                            columns = ['Portfolio','Benchmark'],index=['Sharpe Ratio'])
                                            ])
            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[(pfolio_ret.sum() / 5) - (benchmark_ret.sum() / 5) ]],
                                            columns = ['Portfolio'],index=['Active Return'])
                                            ])

            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[(pfolio_ret - benchmark_ret).std() * math.sqrt(52) ]],
                                            columns = ['Portfolio'],index=['Tracking Error'])
                                            ])
            
            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[((pfolio_ret.sum() / 5) - (benchmark_ret.sum() / 5)) / ((pfolio_ret - benchmark_ret).std() * math.sqrt(52)) ]],
                                            columns = ['Portfolio'],index=['Information Ratio'])
                                            ])

            A = np.vstack([benchmark_ret.values, np.ones(pfolio_ret.shape[0])]).T
            y = pfolio_ret.values
            
            beta , alpha = np.linalg.lstsq(A,y,rcond=None)[0]
            alpha = alpha * 52

            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[beta]],
                                            columns = ['Portfolio'],index=['Beta'])
                                            ])

            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[alpha]],
                                            columns = ['Portfolio'],index=['Alpha'])
                                            ])


            
            pfolio_metrics.loc[pfolio_metrics.index.isin(['Annual Return','Risk','Active Return','Tracking Error','Alpha']),:] = \
                pfolio_metrics.loc[pfolio_metrics.index.isin(['Annual Return','Risk','Active Return','Tracking Error','Alpha']),:] * 100
            

            pfolio_metrics = pfolio_metrics.round(decimals=2)
            
            st.write(pfolio_metrics)

            
            pfolio_max = pfolio_ts.rolling(window=pfolio_ts.shape[0],min_periods=0).max()
            pfolio_dd = (pfolio_ts + 1) / (pfolio_max + 1) - 1
            st.write("Drawdown")
            st.line_chart(pfolio_dd)
            
            pfolio_metrics = pd.DataFrame(columns = ['Portfolio','Benchmark'])
            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[pfolio_dd['Portfolio'].min(),pfolio_dd['Benchmark'].min()]],
                                            columns = ['Portfolio','Benchmark'],index=['Max Drawdown'])
                                            ])
            pfolio_metrics = pd.concat([
                pfolio_metrics,pd.DataFrame([[pfolio_dd['Active'].min()]],
                                            columns = ['Portfolio'],index=['Max Relative Drawdown'])
                                            ])
            
            pfolio_metrics = pfolio_metrics * 100
            pfolio_metrics = pfolio_metrics.round(decimals=2)
            st.write(pfolio_metrics)
            
            ef = pd.read_pickle("./Assets/Eff Frontier.pickle")
            ef['Color'] = 'Efficient Frontier'
            ef['Size'] = 0.5
            
            
            ef = pd.concat([ef,pd.DataFrame([[pfolio_ret.std()*math.sqrt(52),pfolio_ret.sum() / 5,'Portfolio',3]],columns = ['Risk','Return','Color','Size'])])
            ef.Risk = ef.Risk * 100
            ef.Return = ef.Return * 100
            
            c = (
                alt.Chart(ef)
                .mark_circle()
                .encode(alt.X("Risk").scale(zero=False),alt.Y("Return").scale(zero=False),color="Color",size='Size')
                )
            st.write("Historical Efficient Frontier on benchmark indices")
            st.altair_chart(c,use_container_width=True)
            
                                            

            
            