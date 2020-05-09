import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk

from api.velib import get_station_info, get_station_statuses


st.title("Velo 🚴🏻‍♂️")
st.write("Check the availabity of docked bikes in your city !")
st.markdown("_(if your city is Paris)_")


page = st.sidebar.radio("Graph", ("Number of station", "Availability Heatmaps"))

