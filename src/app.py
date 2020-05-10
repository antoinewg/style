import streamlit as st


st.title("Style Transfer 🎨")
st.write("Transfer the style of a painting to your image !")

painting = st.sidebar.radio("Original Painting", ("Number of station", "Availability Heatmaps"))

