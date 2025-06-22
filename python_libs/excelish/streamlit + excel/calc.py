import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def calculate_returns(starting_amount, years_to_invest, additional_contributions, annual_rate_of_return, compounding_frequency):
    total_amount = starting_amount
    compound_rate = 1 + annual_rate_of_return / 100
    compounding_periods = {
        "Annually": 1,
        "Semi-annually": 2,
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

# Title and Header
st.title("S&P 500 Returns Calculator")
st.header("Calculate Future Value of Investment")

# User Inputs
st.subheader("Enter Investment Details")
starting_amount = st.number_input("Starting Amount", min_value=0.0, step=10.)
years_to_invest = st.number_input("Years to Invest", min_value=0.0, step=1.0)
additional_contributions = st.number_input("Additional Contributions", min_value=0.0, step=1.)
annual_rate_of_return = st.number_input("Hypothetical Annual Rate of Return (%)", min_value=0.0, step=0.1)
compounding_frequency = st.selectbox("Compounding Frequency", ("Annually", "Semi-annually", "Quarterly", "Monthly"))

# Calculate and Display Results
if st.button("Calculate"):
    returns = calculate_returns(starting_amount, years_to_invest, additional_contributions, annual_rate_of_return, compounding_frequency)

    st.subheader("Investment Results")
    df_returns = pd.DataFrame({"Year": range(1, int(years_to_invest) + 1), "Return": returns})
    st.dataframe(df_returns)

    # Plotting Yearly Returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_returns["Year"], y=df_returns["Return"], mode='lines+markers', name='Yearly Returns'))
    fig.update_layout(title="Yearly Returns", xaxis_title="Year", yaxis_title="Value")
    st.plotly_chart(fig)
