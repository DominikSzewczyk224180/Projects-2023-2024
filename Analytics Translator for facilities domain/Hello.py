import streamlit as st
from PIL import Image
from functools import partial
import webbrowser

st.set_page_config(
    page_title = 'Home',
    layout = 'wide',
    initial_sidebar_state = 'expanded',
    page_icon = '👋'
)


st.markdown(
    """
    <style>
   
    img {
        border-radius: 10px;
    }
    
    .css-198228p.e1tzin5v0 {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html = True
)

st.sidebar.header('Page Content')

st.sidebar.markdown(
    '''
    - About Us

    - About Our Project
    '''
)



st.title('Meet Our Team 👨‍💻👩‍💻')

if 'members' not in st.session_state:
    
    st.session_state.members = ['Thomas','Romina','Dominik','Amyr','Fedya']


about_us_dict = {
    'Fedya': {
        'image': 'Breda_Municipality_safty/finalApp/data/images/Fedya_cropped.png',
        'short-ov':'''
            - Data Scientisct

            - AI Engineer

            - Streamlit Developer
            ''',
        'full-ov':'''
            - First-year student in the Data Science and Artificial Intelligence program at the Breda University of Applied Science.
            - Played multiple roles in the project: Data Scientist, Data Engineer, AI Specialist, and Streamlit developer.
            - I was responsible for performing an EDA, cleaning and merging the data, creating a model and working on the streamlit app. Enjoys hobbies such as reading, gym workouts, and mobile development.
        '''
    },
    'Thomas': {
        'image': 'Breda_Municipality_safty/finalApp/data/images/thomas_cropped.png',
        'short-ov':'''
            - Data Scientist   

            - Data Analyst 

            - Ethics Analyst
            ''',
        'full-ov':'''
            - First-year student in the Data Science and Artificial Intelligence program at the Breda University of Applied Science.
            - Responsibilities in the assignment included performing an EDA and analyzing the Police dataset for types of crimes in regions of Breda, cleaning the data, and visualizing the findings.
            - Also involved in working on parts of the machine-learning model, creating a data quality report, and visualizing the data on Streamlit. Additionally, a member of a student esports team and enjoys traveling.
        '''
    },
    'Romina': {
        'image': 'Breda_Municipality_safty/finalApp/data/images/romina_cropped.png',
        'short-ov':'''
            - Data Analyst

            - Data Scientist

            - Ethics Analyst
            ''',
        'full-ov':'''
            - First-year Data Science and AI student at BUAS (Breda University of Applied Sciences)
            - Contributions in the project included legal and ethical aspects, machine learning model development, and data analysis
            - Emphasized data quality to ensure accuracy. Enjoys hobbies such as reading, painting, gym workouts, and creating electronic components with Arduino and Raspberry Pi.
        '''
    },
    'Dominik': {
        'image': 'Breda_Municipality_safty/finalApp/data/images/dominik_cropped.png',
        'short-ov':'''
            - Data Scientisct

            - Ai Specialist

            - Streamlit Developer
            ''',
        'full-ov':'''
            - First-year Data Science and AI student passionate about programming and AI
            - Played multiple roles in the project: Data Analyst, Data Scientist, Data Engineer, AI Specialist, and Streamlit developer
            - Developed a user-friendly Streamlit app and collaborated on creating ML models. Enjoys sports, especially gym workouts and football.
        '''
    },
    'Amyr': {
        'image': 'Breda_Municipality_safty/finalApp/data/images/amyr_cropped.png',
        'short-ov':'''
            - Data Scientisct

            - Data Analyst

            - Streamlit Developer
            ''',
        'full-ov':'''
            - I had the task of cleaning and plotting the dataset of Police Performance, Crimes per district and the livability index
            - I had the task of analyzing the data I plotted and created conclusions 
            - I made interactable graphs in the streamlit page and made functions
        '''
    },


}

def change_members_array(name):

    members = st.session_state.members

    this_member_index = members.index(name)

    members[this_member_index] = members[-1]
    members[-1] = name

    return members

def change_member(name):
        
        st.session_state.members = change_members_array(name)




names_pics_col_1, overview, names_pics_col_2 = st.columns([1.5,2,1.5])

with names_pics_col_1:

    picture, text = st.columns([0.5,0.75])

    with picture:

        picture.subheader('  ')

        picture.image(Image.open(about_us_dict[st.session_state.members[0]]['image']))

        button_callback = partial(change_member,st.session_state.members[0])

        picture.button('Read More!', on_click=button_callback, key = st.session_state.members[0])

    with text:

        text.caption(" ")

        text.subheader(st.session_state.members[0])

        text.markdown(
            about_us_dict[st.session_state.members[0]]['short-ov']
        )

        
    

with names_pics_col_1:
    

    picture, text = st.columns([0.5,0.75])

    with picture:

        picture.subheader('  ')

        picture.image(Image.open(about_us_dict[st.session_state.members[1]]['image']))

        button_callback = partial(change_member,st.session_state.members[1])

        picture.button('Read More!', on_click=button_callback, key = st.session_state.members[1])

    with text:

        text.caption(" ")

        text.subheader(st.session_state.members[1])

        text.markdown(
            about_us_dict[st.session_state.members[1]]['short-ov']
        )




with names_pics_col_2:
    

    picture, text = st.columns([0.5,0.75])

    with picture:

        picture.subheader('  ')

        picture.image(Image.open(about_us_dict[st.session_state.members[2]]['image']))

        button_callback = partial(change_member,st.session_state.members[2])

        picture.button('Read More!', on_click=button_callback, key = st.session_state.members[2])

    with text:

        text.caption(" ")

        text.subheader(st.session_state.members[2])

        text.markdown(
           about_us_dict[st.session_state.members[2]]['short-ov']
        )




with names_pics_col_2:
    

    picture, text = st.columns([0.5,0.75])

    with picture:

        picture.subheader('  ')

        picture.image(Image.open(about_us_dict[st.session_state.members[3]]['image']))

        button_callback = partial(change_member,st.session_state.members[3])

        picture.button('Read More!', on_click=button_callback, key = st.session_state.members[3])

    with text:

        text.caption(" ")

        text.subheader(st.session_state.members[3])

        text.markdown(
            about_us_dict[st.session_state.members[3]]['short-ov']
        )




with overview:

    picture, text = st.columns([0.75,1.5])

    with picture:

        picture.divider()

        picture.image(Image.open(about_us_dict[st.session_state.members[-1]]['image']))
        
        picture.text(" ")


    with text:

        text.divider()

        text.markdown(
           about_us_dict[st.session_state.members[-1]]['full-ov']
        )

with overview:
    

    picture, b1,b2,b3 = st.columns([0.75,0.5,0.6,0.5])

    with picture:

        picture.caption(' ')

        picture.subheader(st.session_state.members[-1])


    overview.divider()




st.divider()
st.title('Our Project 📑')
    
safety, combined, ai = st.columns(3)

with safety:

    safety.divider()

    # safety.title(" ")

    c1, c2, c3 = st.columns(3)

    c2.subheader("‎ ‎ ‎ Safety")

    safety.markdown(
        '''
            Safety is one of the critical aspects of a high-quality life. 
            The municipality faces different challenges in identifying the key factors that impact safety and allocating resources to provide safety to all Breda citizens. 
            We decided to focus on physical safety and work with public crimes.
        '''
    )


with combined:

    combined.divider()

    c1, c2, c3 = st.columns(3)

    c2.subheader("‎ ‎ ‎Project")

    combined.markdown(
        '''
        To utilise safety with the help of DS and AI we will:
        - conduct research and perform an analysis to define whether predefined extraneous factors affect safety in different neighbourhoods of Breda
        - based on the research we will create an ML model that forecasts the number of crimes in different neighbourhoods of Breda monthly 
        '''
    )




with ai:

    ai.divider()

    # ai.title(" ")

    c1, c2, c3 = st.columns(3)

    c2.subheader("‎ ‎ DS & AI")

    ai.markdown(
        '''
            Artificial Intelligence and Data Science are getting more popular every day and are utilised in almost every aspect of our lives.
              Because of that, we believe that a data-driven approach will help us find insights into our business case and also build a valuable product that will be used by the municipality of Breda
        '''
    )
