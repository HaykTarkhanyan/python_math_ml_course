# !pip install streamlit
import streamlit as st

# ---------- Displaying Text  ---------- 
# Title and Header
st.title("My Streamlit App")
st.header("Welcome!")

# Text
st.subheader("Text")
st.text("This is a text block.")

# # Markdown
st.subheader("Markdown")
st.markdown("This is a markdown block. You can write *italic*, **bold**, or even use [links](https://www.streamlit.io/).")

# # Latex
st.subheader("Latex")
st.latex(r"Y = \alpha + \beta X_i")

# # ---------- Displaying DataFrames  ---------- 

# # # Data Display
st.subheader("Data Display")
# # Displaying a DataFrame

# st.subheader("DataFrame")
import pandas as pd
data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
st.dataframe(df)

# # Displaying a Table
st.subheader("Table")
st.table(df)

import streamlit as st
import pandas as pd
import plotly.express as px

# # DataFrame
data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
st.dataframe(df)


# # Plotly
st.subheader("Plotly Bar Chart")
fig = px.bar(df, x='Name', y='Age')
st.plotly_chart(fig)


# # # Interactive Widgets
st.subheader("Interactive Widgets")
# Button
if st.button("Click me!"):
    st.write("Button clicked!")

# # # Checkbox
checkbox_value = st.checkbox("Check me!")
if checkbox_value:
    st.write("Checkbox checked!")

# # # Selectbox
option = st.selectbox("Choose an option", ['Option 1', 'Option 2', 'Option 3'])
st.write("You selected:", option)

# # # Slider
slider_value = st.slider("Slide me", 0, 100)
st.write("Slider value:", slider_value)


num_input_value = st.number_input("Starting Amount", min_value=0.0, max_value=100, step=5.)
st.write("Number input value:", num_input_value)


# # # File Uploader
uploaded_file = st.file_uploader("Upload a file", )
if uploaded_file is not None:
    st.write("File uploaded!")

# # Displaying Progress
st.subheader("Progress")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)

# # # Sidebar
st.sidebar.header("Sidebar")
st.sidebar.text("This is a sidebar.")
