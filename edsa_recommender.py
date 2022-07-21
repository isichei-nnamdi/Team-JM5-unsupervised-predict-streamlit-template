"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from ssl import Options
import streamlit as st
from turtle import color, width
import streamlit as st
import streamlit.components.v1 as stc
from streamlit_option_menu import option_menu
import joblib, os
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
import time
from PIL import Image
import pickle as pkle
import os.path

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

import base64

favicon = Image.open('resources/imgs/fav.png')
st.set_page_config(page_title="MovieHub", page_icon=favicon)


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('resources/imgs/background_logo.png')

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# def remote_css(url):
#     st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

# def icon(icon_name):
#     st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("utils/style.css")


# App declaration
def main():
    

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    # page_options = ["Home", "Recommender System","Solution Overview"]

    # Design horizontal bar
    # menu = ["Home", "EDA", "Prediction", "About"]
    page_options = ["Home", "Movies", "EDA", "About"]
    selection = option_menu( menu_title=None,
                            options=page_options,
                            icons=["house", "camera-reels", "graph-up", "file-person"],
                            orientation='horizontal',
                            styles={
                                        "container": {"padding": "0!important", "background-color": "#ED2E38"},
                                        "icon": {"color": "black", "font-size": "25px",  },
                                        "nav-link": {
                                            "font-size": "20px",
                                            "text-align": "center",
                                            "margin": "5px",
                                            "--hover-color": "#eee",
                                            "color": "white"
                                        },
                                        "nav-link-selected": {"background-color": "white", "color": "#ED2E38"},
                                    },
        )    

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    # page_selection = st.sidebar.selectbox("Choose Option", page_options)
    page_selection = selection
    if page_selection == "Home":
        # st.column
        st.markdown("<h2 style='text-align: center; color: white;'>MovieHub, your movie dreams come alive!</h2>", unsafe_allow_html=True)
        picture1, picture2, picture3 = st.columns(3)
        with picture1:
            picture1 = Image.open("resources/imgs/Picture1.png")
            st.image(picture1)

        with picture2:
            picture2 = Image.open("resources/imgs/Picture2.jpg")
            st.image(picture2)
        
        with picture3:
            picture3 = Image.open("resources/imgs/Picture3.jpg")
            st.image(picture3)
        
    # if page_selection == "Recommender System":
    if page_selection == "Movies":
        # Header contents
        # st.write('# Movie Recommender Engine')
        

        # st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection

        st.markdown("<h2 style='text-align: center; color: white;'>What movie are you watching today?</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: white;'>Choose one or more of the options below for the best movie </h4>", unsafe_allow_html=True)

        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    if page_selection == "EDA":
        st.subheader("Exploration Data Analysis")
    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    # if page_selection == "Solution Overview":
    if page_selection == "About":
        with st.sidebar:
            logo = Image.open("resources/imgs/my_logo.png")
            st.image(logo, width =None, use_column_width='False')

            st.markdown(" ")
            st.markdown(" ")

            selected = option_menu("About", ["Recommender System", 'About Team', 'Contact'], 
                    icons=['graph-up-arrow', 'people-fill', 'person-circle'], menu_icon="file-person", default_index=0)
            

        if selected == 'Recommender System':
            st.header("**App Documentation**: Learn How to use the Recommender System")
            # time.sleep(3)
            # st.subheader("Text Classification App") 
            # st.button("Go to next page")

            st.markdown("""<p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'> 
                    This app was primarily created for tweets expressing belief in climate change. There are four pages in the app which includes; `home page`, `predictions`, `Exploratory Data Analysis` and `About`.<br><br>
                    <b style='color: #ED2E38'>Home:</b> The home page is the app's landing page and includes a welcome message and a succinct summary of the app.<br><br>
                    <b style='color: #ED2E38'>EDA:</b> The EDA section, which stands for Explanatory Data Analysis, gives you the chance to explore your data. 
                    Based on the number of hash-tags and mentions in the tweet that have been gathered, it also displays graphs of various groups of 
                    words in the dataset, giving you a better understanding of the data you are working with.<br><br>
                    <b style='color: #ED2E38'>Prediction:</b> This page is where you use the main functionality of the app. It contains two subpages which are: `Single Text Prediction` and `Batch Prediction`<br><br>
                    <b style='color: #ED2E38'>Single Text Prediction:</b> You can predict the sentiment of a single tweet by typing or pasting it on the text prediction 
                    page. Enter any text in the textbox beneath the section, then click "Predict" to make a single tweet prediction.<br><br>
                    <b style='color: #ED2E38'>Batch Prediction:</b> You can make sentiment predictions for batches of tweets using this section. It can process multiple tweets in a batch from a `.csv` 
                    file with at least two columns named `message` and `tweetid` and categorize them into different tweet sentiment groups. To predict by file up, 
                    click on the `browse file` button to upload your file, then click on process to do prediction. A thorough output of the prediction will be provided, 
                    including a summary table and the number of tweets that were categorised under each sentiment class.<br><br>
                    <b style='color: #ED2E38'>About:</b> The About page also has two sub-pages;  `Documentation` and `About Team` page.<br><br>
                    <b style='color: #ED2E38'>Documentation:</b> This is the current page. It includes a detailed explanation of the app as well as usage guidelines on
                    how to use this app with ease.<br><br>
                    <b style='color: #ED2E38'>About Team:</b> This page gives you a brief summary of the experience of the team who built and manages the app.
                    </p>""", unsafe_allow_html=True)
            

        elif selected == "About Team":
            st.header("About Team")
            st.markdown(" ")
            olamide_pic = Image.open("resources/imgs/boarder.png")
            nnamdi_pic = Image.open("resources/imgs/Nnamdi_1.jpg")
            kehinde_pic = Image.open("resources/imgs/kehinde.jpg")
            josh_pics = Image.open("resources/imgs/boarder.png")
            john_pics = Image.open("resources/imgs/boarder.png")
            duncan_pics = Image.open("resources/imgs/boarder.png")


            st.subheader("Olamide - Founder/CEO")
            olamide, text1 = st.columns((1,2))
        
            with olamide:
                st.image(olamide_pic)

            with text1:
                st.write("""
                    <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'> 
                    Olamide is the Founder of Nonnel Data Solution Ltd. she is currently a senior website developer with a background in 
                    soft development, information systems security, digital marketing, and data science.  She has majorly worked in the medical, 
                    educational, government, and hospitality niches with both established and start-up companies.<br><br>
                    She is currently pursuing a master's in business administration. 
                    </p>""", unsafe_allow_html=True)

            st.subheader("Nnamdi - Product Manager")
            nnamdi, text2 = st.columns((1,2))
            
            with nnamdi:
                st.image(nnamdi_pic)

            with text2:
                st.write("""
                    <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                    Nnamdi is a senior product manager with extensive expertise creating high-quality software and a background in user 
                    experience design. He has expertise in creating and scaling high-quality products. He has been able to coordinate 
                    across functional teams, work through models, visualizations, prototypes, and requirements thanks to his attention to detail.<br><br>                
                    He frequently collaborates with data scientists, data engineers, creatives, and other professionals with a focus on business. 
                    He has acquired expertise in engineering, entrepreneurship, conversion optimization, online marketing, and user experience. 
                    He has gained a profound insight of the customer journey and the product lifecycle thanks to that experience.
                </p>""", unsafe_allow_html=True)
            
            st.subheader("Kehinde - Senior Data Analyst")
            kehinde, text3 = st.columns((1, 2))
            
            with kehinde:
                st.image(kehinde_pic)

            with text3:
                st.write("""
                <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                Kehinde is a sales driver, business development and data analyst with keen passion for revenue growth management, 
                Go-To-Market strategy, key account management, alternative channels routing, product-market fixing, business finance 
                and elite team development.<br><br>
                He leverages the power of data for productive and effective solution  cross breeding across business functions for 
                profit scaling in the value chain, without compromising optimal satisfaction along the customer/consumer journey.    
                </p>""", unsafe_allow_html=True)

            st.subheader("Joshua - Lead Strategist")
            josh, text4 = st.columns((1,2))
                    

            with josh:
                st.image(josh_pics)

                with text4:
                    st.write("""
                    <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                    Joshau, a passionate problem solver armed with critical thinking with proficiency in Excel, Powerbi, 
                    SQL and Data science and Machine Learning using Python based technologies. Mid-level Flask Developer, and automation engineer.
                    </p>""", unsafe_allow_html=True)

            st.subheader("John - Data Scientist")
            john, text5 = st.columns((1,2))
            
            
            with john:
                st.image(john_pics)

            with text5:
                st.write("""
                <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                John,  an enthusiastic Data Scientist with great euphoria for Exploratory Data Analysis
                (Power-BI, Tableau, Excel, SQL, Python, R) and Machine Learning Engineering(Supervised and Unsupervised Learning), 
                mid-level proficiency in Front-End Web Development(HTML, CSS, MVC, RAZOR, C#).
                </p>""", unsafe_allow_html=True)

            st.header("Duncan - Customer Success")
            duncan, text6 = st.columns((1,2))
            
            with duncan:
                st.image(duncan_pics)

            with text6:
                st.write("""
                <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                Duncan When it comes to personalizing your online store, nothing is more effective than 
                an About Us page. This is a quick summary of your company's history and purpose, and should provide a clear overview of the 
                company's brand story. A great About Us page can help tell your brand story, establish customer loyalty, and turn your bland 
                ecommerce store into an well-loved brand icon. Most importantly, it will give your customers a reason to shop from your brand.
                </p>""", unsafe_allow_html=True)

        else:
            st.header('Contact')

        
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
