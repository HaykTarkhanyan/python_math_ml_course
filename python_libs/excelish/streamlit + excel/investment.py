import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def calculate_returns(starting_amount, years_to_invest, additional_contributions, annual_rate_of_return, compounding_frequency):
    total_amount = starting_amount
    compound_rate = 1 + annual_rate_of_return / 100
    compounding_periods = {
        "Annually": 1,
        "Half-yearly": 2,
        "Quarterly": 4,
        "Monthly": 12
    }
    periods = int(years_to_invest) * compounding_periods[compounding_frequency]
    returns = []

    for year in range(1, int(years_to_invest) + 1):
        total_amount += additional_contributions
        total_amount *= compound_rate
        returns.append(total_amount)

    return returns

st.title("Investment calculator")
st.header("Welcome!")

st.subheader("Enter details")

frequency = st.selectbox("Choose frequency", ['Monthly', 'Quarterly', 'Half-yearly', 'Annually'])
start_amount = st.number_input("Enter amount", min_value=0, max_value=100_000, value=0, step=10)
years_to_invest = st.number_input("Enter years to invest", min_value=0, max_value=100, value=0, step=1)
interest_rate = st.number_input("Enter interest rate", min_value=0, max_value=100, value=0, step=1)
addition_contributions = st.number_input("Enter addition contributions", min_value=0, max_value=100_000, value=0, step=10)  

if st.button("Calculate"):
    returns = calculate_returns(start_amount, years_to_invest, addition_contributions, interest_rate, frequency)    
    st.subheader("Returns")
    st.write(returns)

    df = pd.DataFrame(index=range(1, years_to_invest + 1), data=returns, columns=["Amount"])
    st.dataframe(df)

    fig = px.line(df, x=df.index, y="Amount")
    st.plotly_chart(fig)

    st.line_chart(returns)