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
import random
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
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
import codecs
from pandas_profiling import ProfileReport 

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles, load_most_recent_movies, load_year_data
from utils.data_loader import load_genre_data, load_director_data, load_merged_data, load_ratings_data
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from utils.movie_details import movie_poster_fetcher, get_movie_info
from utils import contact_form

import base64

favicon = Image.open('resources/imgs/fav.png')
st.set_page_config(page_title="MovieHub", page_icon=favicon)

def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


#------------------------------------------------------------------------------------------------------------
# Adding footer to the StreamLit App
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 50px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height= 0.2, #"auto",
        opacity=1             
    )

    style_hr = styles(
        display="border",
        margin=px(5, 5, "auto", "auto"),
        border_style="none",
        border_width=px(0.5),
        color = 'rgba(0,0,0,.5)'
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        " Powered ❤️ by ",
        link("https://twitter.com/ChristianKlose3", "@DeftAlpGlobal"),
        br(),
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()
#------------------------------------------------------------------------------------------------------------

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
most_recent = load_most_recent_movies('resources/data/most_recent.csv')
sample_recent = most_recent.head(100).sample(3)
year_df = load_year_data('resources/data/merged_data.csv')
genre_df = load_genre_data('resources/data/merged_data.csv')
director_df = load_director_data('resources/data/merged_data.csv')
title_list = load_movie_titles('resources/data/movies.csv')
selected_data = load_merged_data('resources/data/merged_data.csv')
sorted_ratings = load_ratings_data('resources/data/ratings.csv')


def local_button_css(file_name):
                    with open(file_name) as f:
                        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
                
local_button_css("utils/button_style.css")

# App declaration
def main():
    

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    # page_options = ["Home", "Recommender System","Solution Overview"]

    page_options = ["Recommender System", "Movies", "EDA", "About"]
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
    if page_selection == "Recommender System":
        st.markdown("<h2 style='text-align: center; color: white;'>MovieHub, your movie dreams come alive!</h2>", unsafe_allow_html=True)
        #st.markdown("<h3 style='text-align: center; color: #ED2E38;'>Most Recent and Rated Movies!</h3>", 
        #            unsafe_allow_html=True)
        #----------------------------------------------------------------
        sys = st.radio("Select an algorithm",
            ('Content Based Filtering',
                'Collaborative Based Filtering'))

            # selected_title = [title1, title2, title3, title4, title5]
            
            # User-based preferences
        st.write('#### Click the button below for more recommendation')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200]) # random.choice(selected_title)
        movie_2 = st.selectbox('Second Option',title_list[25055:25255]) # random.choice(selected_title)
        movie_3 = st.selectbox('Third Option',title_list[21100:21200]) # random.choice(selected_title)
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
            #-----------------------------------------------------------------------------
        # picture1, picture2, picture3 = st.columns(3)
        # with picture1:
        #     picture1 = movie_poster_fetcher(sample_recent['url'].iloc[0]) 
        #     with st.expander("About Movie"):
        #         desc = get_movie_info(sample_recent['url'].iloc[0])
        #         st.markdown(f"""
        #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #             <b style='color: #ED2E38'>Year: </b>{int(sample_recent['year'].iloc[0])} Movie<br><br>
        #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #             <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #             <a href={sample_recent['url'].iloc[0]}>Click to watch trailer</a>
        #             </p>""", unsafe_allow_html=True)

        # with picture2:
        #     picture2 = movie_poster_fetcher(sample_recent['url'].iloc[1])
        #     with st.expander("About Movie"):
        #         desc = get_movie_info(sample_recent['url'].iloc[1])
        #         st.markdown(f"""
        #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #             <b style='color: #ED2E38'>Year: </b>{int(sample_recent['year'].iloc[1])} Movie<br><br>
        #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #             <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #             <a href={sample_recent['url'].iloc[1]}>Click to watch trailer</a>
        #             </p>""", unsafe_allow_html=True)

        
        # with picture3:
        #     picture3 =  movie_poster_fetcher(sample_recent['url'].iloc[2]) 
        #     with st.expander("About Movie"):
        #         desc = get_movie_info(sample_recent['url'].iloc[2])
        #         st.markdown(f"""
        #             <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #             <b style='color: #ED2E38'>Year: </b>{int(sample_recent['year'].iloc[2])} Movie<br><br>
        #             <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #             <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #             <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #             <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
        #             <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
        #             <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
        #             </p>""", unsafe_allow_html=True)

            
    
    
    # if page_selection == "Recommender System":
    if page_selection == "Movies":
        # Header contents
        # st.write('# Movie Recommender Engine')
        

        # st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        # st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        # with st.sidebar:
        #     logo = Image.open("resources/imgs/my_logo.png")
        #     st.image(logo, width =None, use_column_width='False')

        #     st.markdown(" ")
        #     st.markdown(" ")

        #     select_option = option_menu("Recommenders", ['User preference', 'Model-base'], 
        #             icons=['person-workspace', 'modem'], menu_icon="file-person", default_index=0)

        # if select_option == 'User preference':
        st.markdown("<h2 style='text-align: center; color: white;'>What movie are you watching today?</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: white;'>Choose one or more of the options below for the best movie </h4>", unsafe_allow_html=True)
        
        choice_collection = []

        option1, option2, option3 = st.columns(3)
        with option1:
            movie_year = st.checkbox("Choose Movie Year")
            if movie_year:
                selected_year = st.selectbox(
                    "Select a year", 
                    list(year_df))
                if selected_year:
                    choice_one = choice_collection.append(selected_year)
                else:
                    pass

        with option2:
            genre = st.checkbox("Choose Genre")
            if genre:
                selected_genre = st.selectbox(
                    "Select genre", 
                    list(genre_df))
                if selected_genre:
                    choice_two = choice_collection.append(selected_genre)
                else:
                    pass
                
        with option3:
            director = st.checkbox("Choose a Director")
            if director:
                selected_director = st.selectbox(
                    "Select director", 
                    director_df)
                if selected_director:
                    choice_three = choice_collection.append(selected_director)
                else:
                    pass
        button1, button2, button3 = st.columns(3)

        with button1:
            pass

        with button3:
            pass
        
        with button2:
            button_pressed = button2.button('Search for Movies')

        
        if button_pressed:
            selected_movieid = np.where((selected_data['year'] == choice_collection[0]) | ((selected_data['genre'] == choice_collection[1])))
            sliced_selected_movies = selected_data.iloc[selected_movieid]
        
            suggested_head = sliced_selected_movies.sort_values('year', ascending=False)#.head(50)
            suggested = suggested_head.sample(5)

            
            suggestion1, suggestion2, suggestion3, suggestion4, suggestion5 = st.columns(5)
            
            if "load_state" not in st.session_state:
                st.session_state.load_state = False                 

            with suggestion1:
                suggestion1 = movie_poster_fetcher(suggested['url'].iloc[0])
                if 'url1' not in st.session_state:
                    st.session_state['url1'] = suggested['url'].iloc[0]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[0])
                    title1 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[0])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion2:
                suggestion2 = movie_poster_fetcher(suggested['url'].iloc[1])
                if 'url2' not in st.session_state:
                    st.session_state['url2'] = suggested['url'].iloc[1]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[1])
                    title2 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[1])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion3:
                suggestion3 = movie_poster_fetcher(suggested['url'].iloc[2])
                if 'url3' not in st.session_state:
                    st.session_state['url3'] = suggested['url'].iloc[2]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[2])
                    title3 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[2])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion4:
                suggestion4 = movie_poster_fetcher(suggested['url'].iloc[3])
                if 'url4' not in st.session_state:
                    st.session_state['url4'] = suggested['url'].iloc[3]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[3])
                    title4 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[3])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)

            with suggestion5:
                suggestion5 = movie_poster_fetcher(suggested['url'].iloc[4])
                if 'url5' not in st.session_state:
                    st.session_state['url5'] = suggested['url'].iloc[4]

                with st.expander("About Movie"):
                    desc = get_movie_info(suggested['url'].iloc[4])
                    title5 = desc['Title']
                    st.markdown(f"""
                        <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                        <b style='color: #ED2E38'>Year: </b>{int(suggested['year'].iloc[4])} Movie<br><br>
                        <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
                        <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
                        <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
                        <b style='color: #ED2E38'>Story: </b>{desc['Story']}<br><br>
                        <b style='color: #ED2E38'>Watch Trailer: 👇 </b><br>
                        <a href={sample_recent['url'].iloc[2]}>Click to watch trailer</a>
                        </p>""", unsafe_allow_html=True)
            
            st.write("""
            <h6 style='text-align: center; color: white'>
            Congratulations! Here are some recommendations for you: settle in, unwind, and have fun.</h6>""", unsafe_allow_html=True)
            st.balloons()
                
    
                    
        # else:
        #     st.write("""
        #         <h6 style='text-align: center; color: white'>
        #         Have you seen any of these movie before? Scroll down for further recommendation.</h6>""", unsafe_allow_html=True)
        #     poster1, poster2, poster3, poster4, poster5 = st.columns(5)               

        #     with poster1:
        #         poster1 = movie_poster_fetcher(st.session_state['url1'])

        #         with st.expander("About Movie"):
        #             desc = get_movie_info(st.session_state['url1'])
        #             title1 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>                
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}
        #                 </p>""", unsafe_allow_html=True)

        #     with poster2:
        #         poster2 = movie_poster_fetcher(st.session_state['url2'])
                
        #         with st.expander("About Movie"):
        #             desc = get_movie_info(st.session_state['url2'])
        #             title2 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}
        #                 </p>""", unsafe_allow_html=True)

        #     with poster3:
        #         poster3 = movie_poster_fetcher(st.session_state['url3'])

        #         with st.expander("About Movie"):
        #             desc = get_movie_info(st.session_state['url3'])
        #             title3 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}
        #                 </p>""", unsafe_allow_html=True)

        #     with poster4:
        #         poster4 = movie_poster_fetcher(st.session_state['url4'])

        #         with st.expander("About Movie"):
        #             desc = get_movie_info(st.session_state['url4'])
        #             title4 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}
        #                 </p>""", unsafe_allow_html=True)

        #     with poster5:
        #         poster5 = movie_poster_fetcher(st.session_state['url5'])

        #         with st.expander("About Movie"):
        #             desc = get_movie_info(st.session_state['url5'])
        #             title5 = desc['Title']
        #             st.markdown(f"""
        #                 <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
        #                 <b style='color: #ED2E38'>Director: </b>{desc['Director']}<br><br>
        #                 <b style='color: #ED2E38'>Title: </b>{desc['Title']}<br><br>
        #                 <b style='color: #ED2E38'>Cast: </b>{desc['Cast']}<br><br>
        #                 <b style='color: #ED2E38'>Story: </b>{desc['Story']}
        #                 </p>""", unsafe_allow_html=True)

            # sys = st.radio("Select an algorithm",
            # ('Content Based Filtering',
            #     'Collaborative Based Filtering'))

            # # selected_title = [title1, title2, title3, title4, title5]
            
            # # User-based preferences
            # st.write('#### Click the button below for more recommendation')
            # movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200]) # random.choice(selected_title)
            # movie_2 = st.selectbox('Second Option',title_list[25055:25255]) # random.choice(selected_title)
            # movie_3 = st.selectbox('Third Option',title_list[21100:21200]) # random.choice(selected_title)
            # fav_movies = [movie_1,movie_2,movie_3]

            # # Perform top-10 movie recommendation generation
            # if sys == 'Content Based Filtering':
            #     if st.button("Recommend"):
            #         try:
            #             with st.spinner('Crunching the numbers...'):
            #                 top_recommendations = content_model(movie_list=fav_movies,
            #                                                     top_n=10)
            #             st.title("We think you'll like:")
            #             for i,j in enumerate(top_recommendations):
            #                 st.subheader(str(i+1)+'. '+j)
            #         except:
            #             st.error("Oops! Looks like this algorithm does't work.\
            #                     We'll need to fix it!")


            # if sys == 'Collaborative Based Filtering':
            #     if st.button("Recommend"):
            #         try:
            #             with st.spinner('Crunching the numbers...'):
            #                 top_recommendations = collab_model(movie_list=fav_movies,
            #                                                 top_n=10)
            #             st.title("We think you'll like:")
            #             for i,j in enumerate(top_recommendations):
            #                 st.subheader(str(i+1)+'. '+j)
            #         except:
            #             st.error("Oops! Looks like this algorithm does't work.\
            #                     We'll need to fix it!")
            
            # for key in st.session_state.keys():
            #     del st.session_state[key]
            
    if page_selection == "EDA":
        with st.sidebar:
            logo = Image.open("resources/imgs/my_logo.png")
            st.image(logo, width =None, use_column_width='False')

            st.markdown(" ")
            st.markdown(" ")
            st.write("Select the module to use for the EDA")

            selected = option_menu("Visualizations", ["Pandas Profiling", 'Sweet Visualization'], 
                    icons=['graph-up-arrow', 'list-task'], menu_icon="cast", default_index=0)
            
            

        if selected == 'Pandas Profiling':
            st.markdown("This provides an indepth view of the data.",unsafe_allow_html=True)
            ds = st.radio("choose the data source", ("Movies", "Ratings"))
            if ds == "Movies":
                data_file = 'resources/data/movies.csv'
            else:
                data_file = 'resources/data/ratings.csv'
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.dataframe(df.head())
                profile = ProfileReport(df)
                st_profile_report(profile)
            pass
         
        if selected == 'Sweet Visualization':
            st.markdown("Visualize the data using the SweetViz module",unsafe_allow_html=True)
            ds = st.radio("choose the data sorce", ("movies data", "ratings data"))
            if ds == "movies data":
                data_file = 'resources/data/movies.csv'
            else:
                data_file = 'resources/data/ratings.csv'
            if data_file is not None:
                df1 = pd.read_csv(data_file)
                st.dataframe(df1.head())
                if st.button("Generate Sweetviz Report"):
                    report = sv.analyze(df1)
                    report.show_html()
                    st_display_sweetviz("SWEETVIZ_REPORT.html")
        pass
    
 
    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    # if page_selection == "Solution Overview":
    if page_selection == "About":
        with st.sidebar:
            logo = Image.open("resources/imgs/my_logo.png")
            st.image(logo, width =None, use_column_width='False')

            st.markdown(" ")
            st.markdown(" ")

            selected = option_menu("About", ["Recommender", 'About Team', 'Contact Us'], 
                    icons=['graph-up-arrow', 'people-fill', 'person-circle'], menu_icon="file-person", default_index=0)
            

        if selected == 'Recommender':
            st.header("**App Documentation**: Learn How to use the Recommender System")

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
            olamide_pic = Image.open("resources/imgs/olamide.JPG")
            nnamdi_pic = Image.open("resources/imgs/Nnamdi_1.JPG")
            kehinde_pic = Image.open("resources/imgs/kehinde.JPG")
            josh_pics = Image.open("resources/imgs/josh.JPG")
            john_pics = Image.open("resources/imgs/john.JPG")
            duncan_pics = Image.open("resources/imgs/duncan.JPG")


            st.subheader("Olamide - Founder/CEO")
            olamide, text1 = st.columns((1,2))
        
            with olamide:
                st.image(olamide_pic)

            with text1:
                st.write("""
                    <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'> 
                    Olamide is the Founder of DeftAlp Global. She is results-driven chief executive officer and data scientist with over 
                    6 months of experience leading and increasing growth in small and medium projects through employee engagement. 
                    I am Seeking to lead and grow alongside other data scientists as the following problem solvers.<br><br>
                    At DeftAlp Global, We Seek to raise earnings by 40% through organizational restructuring with the aim of giving our 
                    users an unbeatable experience. 
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

            st.subheader("Joshua - Lead Project Manager")
            josh, text4 = st.columns((1,2))
                    

            with josh:
                st.image(josh_pics)

                with text4:
                    st.write("""
                    <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                    Josh is an enthusiastic and prolific young man with a special interest in technology. He is well skilled in project 
                    management and has seen the incubation and execution of several projects within the allocated time frame.<br><br>
                    He is presently into Data Science and desires to leverage data to execute some contemporary projects.
                    </p>""", unsafe_allow_html=True)

            st.subheader("John - Robotic Engineer")
            john, text5 = st.columns((1,2))
            
            
            with john:
                st.image(john_pics)

            with text5:
                st.write("""
                <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                John is a seasoned mathematician whose hands-on expertise has greatly impacted organizations and investments within the 
                retail and the Oil and Gas sector of the economy. <br><br>
                He is keenly interested in the  application of Data Science and Computational Mathematics in Robotics, Artificial Intelligence, 
                and Machine Learning to drive growth, efficiency and cost effectiveness in the energy, health, and innovative technology space.
                </p>""", unsafe_allow_html=True)

            st.header("Duncan - Statistical Analyst")
            duncan, text6 = st.columns((1,2))
            
            with duncan:
                st.image(duncan_pics)

            with text6:
                st.write("""
                <p style='text-align: left; color: white; background-color: rgba(0,0,0,.5)'>
                Duncan is a dedicated, self-driven, forward thinking and result oriented statistician specialized in designing, implementing 
                survey methodologies questionnaires, and sampling frames. Duncan is skilled in Data Analytics, Statistical Modelling, Development 
                and Maintenance of Databases, Market Research, and Data Collection. Strong research and professional  experience with Bachelor in 
                Economics and Statistics.<br><br>
                Duncan is currently in Data Science to sharpen skills in math, statistics, programming to organize large 
                data in uncovering solutions hidden in data to take business challenges and goals.
                </p>""", unsafe_allow_html=True)

        else:
            st.header('Contact Us')
            st.write('Kindly fill the form below 👇 and we will get back to you ASAP:')
            # cst.header(":mailbox: Get In Touch With Me!")


            contact_form = """
            <form action="https://formsubmit.co/isichei.nnamdi@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your name" required>
                <input type="email" name="email" placeholder="Your email" required>
                <textarea name="message" placeholder="Your message here"></textarea>
                <button type="submit">Send</button>
            </form>
            """

            st.markdown(contact_form, unsafe_allow_html=True)

            # Use Local CSS File
            def local_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


            local_css("utils/style.css")
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
