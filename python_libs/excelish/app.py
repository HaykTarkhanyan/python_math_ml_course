import pandas as pd
import streamlit as st
import plotly.express as px

st.title('CBA Dashboard')
st.write('Welcome to the CBA Dashboard!')

URL = st.text_input('Enter the URL:', 
                    "https://www.cba.am/stat/stat_data_arm/Loans_by_regions_arm.xlsx")

df_himnakan = pd.read_excel(URL)
df_dynamic = pd.read_excel(URL, sheet_name="Դինամիկ շարք")

st.write(df_himnakan.head())
st.header('Dynamic')
st.write(df_dynamic.head())

df_himnakan.drop(columns=["Unnamed: 0"], inplace=True)

vernagir = df_himnakan.iloc[0, 0]
amsativ = df_himnakan.iloc[3, 0]
nshum = df_himnakan.iloc[-1, 0]

st.markdown(f"*Վերնագիր*: {vernagir}")
st.write(f"Ամսաթիվ: {amsativ}")
st.write(f"Նշում: {nshum}")

df_h = df_himnakan.iloc[4:23 ,:]


df_h["Unnamed: 1"].fillna("Վարկի տեսակ", inplace=True)

df_h.columns = df_h.iloc[0]


df_h.drop(index=4, inplace=True)

df_h.reset_index(drop=True, inplace=True)

df_h.fillna(0, inplace=True)

df_long = pd.melt(df_h, id_vars="Վարկի տեսակ", var_name="region", value_name="amount")
df_long.rename(columns={"Վարկի տեսակ": "loan_type"}, inplace=True)

df_long.loan_type = df_long.loan_type.apply(lambda x: x.strip())
df_long.region = df_long.region.apply(lambda x: x.strip())


df_long.loan_type = df_long.loan_type.apply(lambda x: x[0])


st.write(df_long.head())

loan = st.selectbox("Select loan type", df_long["loan_type"].unique())

df_f = df_long[df_long["loan_type"] == loan]

fig = px.bar(df_f, x="region", y="amount", title=f"{loan} by region")

st.plotly_chart(fig)
