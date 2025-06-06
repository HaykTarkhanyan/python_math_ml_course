"""
streamlit run streamlit_app.py

- Gallery (official) - https://streamlit.io/gallery
- Gallery (repo) - https://github.com/jrieke/best-of-streamlit
- Example - https://prettymapp.streamlit.app/?ref=streamlit-io-gallery-favorites
- Cheatsheet - https://luluwu-cheatsheet.streamlit.app/
- Reference - https://docs.streamlit.io/develop/api-reference

pip install streamlit
conda install streamlit
"""
from time import sleep

import pandas as pd
import streamlit as st
import plotly.express as px

# with st.echo():
#     # congrats
#     st.balloons()
#     st.snow()
    


with st.echo():
    # ---------- Displaying Text  ----------
    # Title and Header
    st.title("My Streamlit App")
    st.header("Welcome!")

with st.echo():
    # Text
    st.subheader("Text")
    st.text("This is a text block.")

with st.echo():
    # Markdown
    st.subheader("Markdown")
    st.markdown("This is a markdown block. You can write *italic*, **bold**, or even use [links](https://www.streamlit.io/).")

st.divider()
# Latex
st.subheader("Latex")
with st.echo():
    st.latex(r"Y = \alpha + \beta X_i")

with st.echo():
    st.write("You can also use LaTeX for mathematical expressions like this: $E = mc^2$")


st.divider()
st.subheader("Table")
with st.echo():
    # ---------- Displaying DataFrames  ----------
    # Displaying a Table
    st.dataframe([{ 'Name': "Գինի", 'Age': 3 }, { 'Name': "Պանիր", 'Age': 1 }])

with st.echo():
    # DataFrame
    data = {'Name': ['Մալխաս', 'Բարխուդարում', 'Մատատյան ու Մարոխյան'], 'Age': [3, 4, 1]}
    df = pd.DataFrame(data)
    st.dataframe(df)

with st.echo():
    st.json({
        "Product": "Պանիր",
        "Type": "Չանախ",
    })

# Plotly
st.subheader("Plotly Bar Chart")

with st.echo():
    st.divider()
    fig = px.bar(df, x='Name', y='Age')
    st.plotly_chart(fig)

with st.echo():
    st.line_chart([10, 14, 19, 20, 25, 30, 35, 40, 45, 50, 55, 60])

st.divider()
st.subheader("Interactive Widgets")
with st.echo():
    # Interactive Widgets
    # # # Button
    if st.button("Click me!"):
        st.write("Button clicked!")

with st.echo():
    # Checkbox
    checkbox_value = st.checkbox("Check me!")
    if checkbox_value:
        st.write("Checkbox checked!")

with st.echo():
    # Selectbox
    option = st.selectbox("Choose an option", ['Option 1', 'Option 2', 'Option 3'])
    st.write("You selected:", option)

with st.echo():
    # Slider
    slider_value = st.slider("Slide me", 0, 100, step=5)
    st.write("Slider value:", slider_value)

with st.echo():
    num_input_value = st.number_input("Num input", min_value=0.0, max_value=100., step=5.0) # պետքա float լինի
    st.write("Number input value:", num_input_value)

with st.echo():
    text = st.text_input("please input text")
    st.write("Text input value:", text)

with st.echo():
    # File Uploader
    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        st.write("File uploaded!")

st.divider()
# Displaying Progress
st.subheader("Progress")

# with st.echo():
#     progress_bar = st.progress(0)

#     for i in range(100):
#         sleep(0.1)
#         progress_bar.progress(i + 1)

with st.echo():
    # Sidebar
    st.sidebar.header("Sidebar")
    st.sidebar.text("This is a sidebar.")

with st.echo():
    rating1 = st.feedback("stars")
    rating2 = st.feedback("faces")
    st.write(f"Rating 1: {rating1}")
    st.write(f"Rating 2: {rating2}")

with st.echo():
    color = st.color_picker("Pick a color")
    st.write("Selected color:", color)
    
with st.echo():
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)

    if picture:
        st.image(picture)
        
with st.echo():
    audio_value = st.audio_input("Record a voice message")

    if audio_value:
        st.audio(audio_value)
        
with st.echo():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with col2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")

    with col3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg")