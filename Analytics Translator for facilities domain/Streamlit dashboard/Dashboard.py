import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
import altair as alt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PIL import Image



st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #4287f5, #00a99d);
        color: white;
    }
    .stTitle {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .stText {
        font-size: 18px;
        text-align: left;
    }
    .stHeader {
        font-size: 24px;
        font-weight: bold;
        text-align: left;
        background-color: #00a99d;
        color: white;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)   






# Title and comment
st.title("Perspectives and Insights on AI Incorporation within Facility Management")
st.write("This research explores how students and teaching staff perceive AI integration in Facility Management, focusing on its impact in educational settings and the potential to enhance security, privacy, and space optimization.")
st.write("Authors: Dominik Szewczyk, Imani Senior, Martin Vladimirov, Matey Nedyalkov, and Simona Dimitrova")

# Introduction
st.header("Introduction")
st.write("The problem is the absence of AI in facility management education, leading to a skills gap. Breda University of Applied Sciences is studying how students and staff view AI to shape future policies. AI improves facility management by boosting efficiency, cutting costs, and enhancing safety through predictive maintenance and space optimization.")

# Objective
st.header("Objective")
st.write("The objective of the project is to try to integrate artificial Intelligence in facility management program at Breda University of Applied Sciences.")

# Methodology
st.header("Methodology")
st.write("The research employed a mixed-method approach, utilizing both qualitative and quantitative methods. Interviews were conducted and analyzed thematically, while survey data was processed using the R programming language, including hypothesis testing.")

# Results
st.header("Results")
st.write("This study delves into AI's impact on BUas facility management. It finds broad support for AI's potential benefits but also significant reservations, especially among educators. Attitudes toward AI are influenced by various factors, including experience and positive interactions. Feedback on BUas facilities is generally positive, with varying views from students and educators.")

# Analysis
st.header("Analysis")
st.write("Data suggests AI's promise in improving maintenance skills through practical education, but the statistical methodologies used in the study didn't yield concrete evidence, pointing to the need for further research or refined models. Furthermore, students generally hold positive views of AI, while educators exhibit slightly more varied opinions, though the sentiment gap between these two groups remains relatively minor, providing a comprehensive overview of AI attitudes.")

# Recommendation
st.header("Recommendation")
st.write("The recommendations include updating the curriculum with AI content for students, enhancing campus navigation through AI systems, investing in AI research and development, implementing feedback mechanisms for continuous improvement, and addressing research limitations by broadening the scope and conducting periodic follow-up studies.")

# Conclusion
st.header("Conclusion")
st.write("This study explores AI's role in academic facility management at BUas, with a focus on student and staff opinions. It highlights optimism about AI's potential but also concerns around ethics and over-dependence on technology. The key is a balanced approach that values human judgment, provides necessary skills, and maintains open dialogue. Educational strategies should adapt to enhance the academic journey as technology evolves.")


participants_distribution = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/participants.png")
demo_INT_INCR = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/acc_3.png")
demo_USED_AI = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/used_ai.png")
demo_KNOWLEDGE = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/used_ai.png")
demo_EXPERIENCE = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/Distribution of Experience Levels.png")


show_images = st.button("Show Distribution Images")

if st.button("Clear Distribution Images"):
    show_images = False

if show_images:
    # Display the images when the button is clicked
    st.title("Distribution Images Page")
    
    st.header("Participants Distribution")
    st.image(participants_distribution, use_column_width=True, caption="Participants Distribution")
    
    st.header("Demo INT INCR")
    st.image(demo_INT_INCR, use_column_width=True, caption="Demo INT INCR")
    
    st.header("Demo USED AI")
    st.image(demo_USED_AI, use_column_width=True, caption="Demo USED AI")
    
    st.header("Demo KNOWLEDGE")
    st.image(demo_KNOWLEDGE, use_column_width=True, caption="Demo KNOWLEDGE")

    st.header("Demo EXPERIENCE")
    st.image(demo_EXPERIENCE, use_column_width=True, caption="Demo EXPERIENCE")


# education = st.sidebar.selectbox('AI experience', ['Results', 'Discussion', '---'], index=2)
# space = st.sidebar.selectbox('Space Optimization', ['Results', 'Discussion', '---'], index=2)
# maintenance = st.sidebar.selectbox('Predictive Maintenance', ['Results', 'Discussion', '---'], index=2)
# security = st.sidebar.selectbox('Security and Privacy', ['Results', 'Discussion', '---'], index=2)
# impact = st.sidebar.selectbox('AI Impact', ['Results', 'Discussion', '---'], index=2)
# Display the selected options

#IMAGES

# Simona
participants_distribution = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/participants.png")
demo_INT_INCR = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/acc_3.png")
demo_USED_AI = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/used_ai.png")
demo_KNOWLEDGE = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/used_ai.png")
demo_EXPERIENCE = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/Distribution of Experience Levels.png")
mlr_domains = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/mlr_int_famil_expr_all_domain.png")
mlr_facility = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/mlr_facility.png")
lr_facility = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Simona/lr_facility_know_intend.png")



# Martin

Ethical_considerations_chitest = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Martin/Distribution of responses for ethical implications.png")
AI_future_use = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Martin/Distribution of responses for adoption of AI in future uses.png")
AI_difficulty_implementation = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Martin/Distribution of responses for difficulty of implementation.png")
AI_industry_standard = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Martin/Distribution of responses for belief in AI becoming future industry standard.png")
box_plot = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Martin/Linear Regression Boxplot.png")

# Imani

# Dominik

AI_Anxiety = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/AI Anxiety.png")
AI_vs_H_1 = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/AI vs Humans Assist in Creative Tasks.png")
AI_vs_H_2 = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/AI vs Humans Automating Repetitive Tasks.png")
AI_Advantages = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/AI's Advantages.png")
AI_Impact_ee = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/AI's Impact on Facility Energy Efficiency.png")
AI_distribution = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/Distribution of Experience Levels.png")
AI_Key_Roles = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/Facility Management and AI Key Roles and Expectations.png")
AI_Impact = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/Impact of AI on FM.png")
AI_Importance = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/Importance of AI in FM by Experience Level.png")
Linear_regresion = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Dominik/Linear Regresion Importace vs Impact.png")

# Matey

f_predictive_maintenance_answers = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Matey/PM_answers.png")
f_predictive_maintenance_methods = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Matey/pm_text_methods_bar_chart.png")
test_graph = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Matey/One-sample_T-Test.png")
linear_graph = Image.open("Analytics Translator for facilities domain/Streamlit dashboard/Images/Matey/Linear_graph.png")

section = st.sidebar.selectbox('Select Section', ['---', 'Education', 'Space Optimization', 'Predictive Maintenance', 'AI Impact','Security and Privacy'])

if section == "---":
    st.header(" ")

elif section == "Education":
    st.header("Results")
    st.image([mlr_domains,mlr_facility,lr_facility], use_column_width=True)


elif section == "Space Optimization":
    st.header("Results")
    st.image([Ethical_considerations_chitest, AI_future_use, AI_difficulty_implementation,AI_industry_standard, box_plot ], use_column_width=True)

    st.header("Discussion")

    st.subheader("AI as a Future Standard")
    st.markdown("Educators expect AI to be a standard tool, with students less certain but still optimistic.")

    st.subheader("Confidence in AI for Space Optimization")
    st.markdown("Both students and educators trust in AI's ability to improve space optimization.")

    st.subheader("Implementing AI: Difficulty Perception")
    st.markdown("Students are unsure about AI implementation ease, while educators are confident, reflecting experience and optimism.")

    st.subheader("Ethical Views on AI")
    st.markdown("Educators are more concerned about AI's ethical implications than students.")

    st.subheader("Statistical Significance")
    st.markdown("A chi-squared test confirms significant differences in ethical views on AI between lecturers and students.")


elif section == "Predictive Maintenance":
    st.header("Results")
    st.image([f_predictive_maintenance_answers, f_predictive_maintenance_methods, test_graph, linear_graph])

    st.header("Discussion")

    st.markdown("In this section of the report, the discussion of the findings is presented. The descriptive analysis indicates that most participants believe that AI and predictive maintenance will significantly impact their skill development and its real-life application. They also express a preference for learning through real-world case studies and interactive workshops. The t-test and linear regression results suggest that the statistical analysis doesn't provide strong evidence to reject the null hypotheses, indicating that there is a positive impact in implementing predictive analysis in maintenance on participants' skills. However, the statistical models used in the study may not be reliable, as indicated by low F-statistic value. Therefore, the report recommends further research or the exploration of different statistical models to strengthen the findings.")


elif section == "Security and Privacy":
    st.header("Results")

elif section == "AI Impact":
    st.header("Results")

    st.subheader("Are people in Facility Management afraid of artificial intelligence?")
    st.image([AI_Anxiety], use_column_width=True)
    st.markdown("The good news is that most people in Facility Management (FM) aren't worried about AI. (based on the survey) This is a positive sign because it means they are open to using AI in their work. Using AI in FM can make things work better, help manage resources, and improve services. This benefits both the industry and the people involved")
 
    st.subheader("Do Facility Management Members Believe AI Can Help Humans at Work?")
    st.image([AI_vs_H_1, AI_vs_H_2], use_column_width=True)
    st.write("Most participants in the survey share a positive view of AI. They believe that AI can not only replace humans in repetitive tasks but also assist in creative tasks. This is a promising sign, indicating that they recognize AI's potential to enhance work processes.")
    st.image([ AI_Advantages], use_column_width=True)
    st.write("Members of Facility Management are already aware of the numerous beneficial applications of AI, which can greatly assist in real-life situations.")

    st.subheader("Where Do Facility Management Members Believe AI Will Play Key Roles?")
    st.image([AI_Key_Roles,  AI_Impact_ee ], use_column_width=True)
    st.write("Facilities management members anticipate that AI will play a key role in a variety of aspects, including energy efficiency, space optimization, predictive maintenance, and security. Most of them agree that AI's significant impact will be in improving energy efficiency. The second plot highlights agreement regarding AI's potential in enhancing energy efficiency.")

    st.subheader("Do People in Facility Management Believe Importance Equals Positive Impact for AI?")
    st.image([AI_Impact, AI_Importance, Linear_regresion], use_column_width=True)
    st.write("Most people in Facility Management agree that AI is important and will have a positive impact. What's even more interesting is that, as per the linear regression plot, when people consider AI to be more important, they also believe it will have a more positive impact on FM.")






# SIMONA


# if education == 'Results':
#     # Display the images for "Results" section
#     st.image([], use_column_width=True)

# elif education == 'Discussion':
#     # Display the discussion content
#     st.header("Discussion")
# # place for the short summary of Discussion

# elif education == "---":
#     st.header(" ")
# # Dominik

# AI_Anxiety = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\AI Anxiety.png")
# AI_vs_H_1 =  Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\AI vs Humans Assist in Creative Tasks.png")
# AI_vs_H_2 = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\AI vs Humans Automating Repetitive Tasks.png")
# AI_Advantages = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\AI's Advantages.png")
# AI_Impact_ee = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\AI's Impact on Facility Energy Efficiency.png")
# AI_distribution = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\Distribution of Experience Levels.png")
# AI_Key_Roles = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\Facility Management and AI Key Roles and Expectations.png")
# AI_Impact = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\Impact of AI on FM.png")
# AI_Importance = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\Importance of AI in FM by Experience Level.png")
# Linear_regresion = Image.open(r"C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Dominik\Linear Regresion Importace vs Impact.png")

# if impact == 'Results':
#     # Display the images for "Results" section
#     st.image([AI_Anxiety, AI_vs_H_1, AI_vs_H_2, AI_Advantages, AI_Impact_ee, AI_distribution, AI_Key_Roles, AI_Impact, AI_Importance, Linear_regresion], use_column_width=True)

# elif impact == 'Discussion':
#     # Display the discussion content
#     st.header("Discussion")
    
#     st.subheader("7.1 Opportunities for AI in Facility Management")
#     st.markdown("Figures 6 and 7 revealed insights into BUAS students' perceptions of AI in facility management. Notably, 42% of respondents believe AI could significantly impact 'Energy Efficiency,' aligning with the global sustainability focus. This emphasizes the potential for AI to enhance facility management practices, contributing to sustainability and operational efficiency.")
    
#     st.subheader("7.2 Perceptions Among Different Experience Levels")
#     st.markdown("Surprisingly, there was no statistically significant difference in AI perception among respondents with varying levels of experience (p = 0.264). This consistent perception simplifies AI adoption and integration within the field, as there is a shared consensus on its importance.")
    
#     st.subheader("7.3 Perceptions and Impact of AI")
#     st.markdown("The linear regression analysis (Figure 8) revealed a significant positive relationship between students' perceived importance of AI and their belief in AI's potential impact on facility management. Students exhibited a level of ambivalence regarding AI's scariness, with 20% strongly disagreeing and 40% expressing neutrality. Additionally, 50% strongly disagreed with the idea of beneficial AI applications in facility management. However, 84% considered AI's impact 'Somewhat positive.' This reflects a high degree of optimism and readiness for AI in the field, especially in automating repetitive tasks, where opinions vary.")
# elif impact == "---":
#     st.header(" ")


# # Matey

# f_predictive_maintenance_answers = Image.open(r'C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Matey\PM_answers.png')
# f_predictive_maintenance_methods = Image.open(r'C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Matey\pm_text_methods_bar_chart.png')
# test_graph = Image.open(r'C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Matey\One-sample_T-Test.png')
# linear_graph = Image.open(r'C:\Users\domin\Desktop\2023-24a-fai2-adsai-group-team-facility\Digital Presentation\presenatation\Images\Matey\Linear_graph.png')

# if maintenance == "Results":
#     # Display the images for "Results" section
#     st.image([f_predictive_maintenance_answers, f_predictive_maintenance_methods, test_graph, linear_graph])

# elif maintenance == "Discussion":

#     st.header("Discussion")

#     st.markdown("In this section of the report, the discussion of the findings is presented. The descriptive analysis indicates that most participants believe that AI and predictive maintenance will significantly impact their skill development and its real-life application. They also express a preference for learning through real-world case studies and interactive workshops. The t-test and linear regression results suggest that the statistical analysis doesn't provide strong evidence to reject the null hypotheses, indicating that there is a positive impact in implementing predictive analysis in maintenance on participants' skills. However, the statistical models used in the study may not be reliable, as indicated by low F-statistic value. Therefore, the report recommends further research or the exploration of different statistical models to strengthen the findings.")
# elif maintenance == "---":
#     st.header(" ")

# #Martin

# # place for pictures u want to visualise


# if space == 'Results':
#     # Display the images for "Results" section
#     st.image([], use_column_width=True)

# elif space == 'Discussion':
#     # Display the discussion content
#     st.header("Discussion")
#     st.markdown()
# # place for the short summary of Discussion

# elif space == "---":
#     st.header(" ")

# #Imani

# # place for pictures u want to visualise


# if security == 'Results':
#     # Display the images for "Results" section
#     st.image([], use_column_width=True)

# elif security == 'Discussion':
#     # Display the discussion content
#     st.header("Discussion")
#     st.markdown()
# # place for the short summary of Discussion

# elif security == "---":
#     st.header(" ")