import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os


def load_data():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


UFC_fighter_photo_loc = os.getcwd()+"/UFC_Fighters_Photos/UFCFightersPhotos"

image = Image

data = load_data()
weight_classes = data["weight_classes"]
fighter_classes = data["fighter_classes"]
fighter_list = data["fighter_list"]
fighter_stats = data["fighter_stats"]
model = data["model"]
model_acc = data["model_acc"]

def show_page():
    m_acc = model_acc * 100
    st.title("UFC Fight Prediciton Using Machine Learning")
    st.write(os.getcwd())
    st.write("""#### Model Accuracy: {:.2f}%""".format(m_acc))
    st.title("")
    st.write("""### Choose Your Weight Class""")
    st.write("##")
    weight_class = st.selectbox(label = "Weight Classes", options = weight_classes)
    weight_class = weight_class.replace("\'", "")
    weight_class = weight_class.replace(" ", "_")
    weight_class = weight_class.lower()
    
    fighters_ = fighter_classes[weight_class]
    fighters_ = set(fighters_.dropna())
    st.title("")
    st.title("")
    col1, col2, col3 = st.columns(3)
    with col1:
        Red_Fighter = st.selectbox(label = "Select Red Fighter", options = fighters_)
        rFighter = Red_Fighter.replace(" ", "-")
        rFighter_loc = UFC_fighter_photo_loc+'/'+rFighter+".jpg"
        st.write(rFighter_loc)
        try: rImage = Image.open(rFighter_loc)
        except:
            if(weight_class == 'womens_strawweight' or weight_class == 'womens_flyweight' or weight_class == 'womens_bantamweight' or weight_class == 'womens_featherweight'):
                rFighter_loc = UFC_fighter_photo_loc+'/'+"Default-G.jpg"
                rImage = Image.open(rFighter_loc)
            else:
                rFighter_loc = UFC_fighter_photo_loc+'/'+"Default-B.jpg"
                rImage = Image.open(rFighter_loc)

        st.image(rImage)

    with col3:
        Blue_Fighter = st.selectbox(label = "Select Blue Fighter", options = fighters_)
        bFighter = Blue_Fighter.replace(" ", "-")
        bFighter_loc = UFC_fighter_photo_loc+'/'+bFighter+".jpg"
        try:
            bImage = Image.open(bFighter_loc)
        except:
            if(weight_class == 'womens_strawweight' or weight_class == 'womens_flyweight' or weight_class == 'womens_bantamweight' or weight_class == 'womens_featherweight'):
                bFighter_loc = UFC_fighter_photo_loc+'/'+"Default-G.jpg"
                bImage = Image.open(bFighter_loc)
            else:
                bFighter_loc = UFC_fighter_photo_loc+'/'+"Default-B.jpg"
                bImage = Image.open(bFighter_loc)

        st.image(bImage)

    with col2:
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        ok = st.button("Predict")
        if ok:
            if Red_Fighter == Blue_Fighter:
                st.write("Red Fighter and Blue Fighter should not be the same person")
            elif Red_Fighter == ' ':
                st.write("Please choose Red Fighter")
            elif Blue_Fighter == ' ':
                st.write("Please choose Blue Fighter")
            elif Red_Fighter == ' ' and Blue_Fighter == ' ':
                st.write("Please choose Red and Blue Fighter")
            else:
                b_index = fighter_stats.index[fighter_stats['Name'] == Blue_Fighter]
                b_index = b_index[0]

                r_index = fighter_stats.index[fighter_stats['Name'] == Red_Fighter]
                r_index = r_index[0]

                win_streak_diff = fighter_stats['current_win_streak'][b_index] - fighter_stats['current_win_streak'][r_index]
                loss_diff = fighter_stats['losses'][b_index] - fighter_stats['losses'][r_index]
                round_count_diff = fighter_stats['total_rounds_fought'][b_index] - fighter_stats['total_rounds_fought'][r_index]
                reach_diff = fighter_stats['Reach_cms'][b_index] - fighter_stats['Reach_cms'][r_index]
                age_diff = fighter_stats['age'][b_index] - fighter_stats['age'][r_index]
                avg_TD_landed_diff = fighter_stats['avg_TD_landed'][b_index] - fighter_stats['avg_TD_landed'][r_index]
                sig_str_diff = fighter_stats['sig_strikes_landed'][b_index] - fighter_stats['sig_strikes_landed'][r_index]

                X = np.array([[win_streak_diff, loss_diff, round_count_diff, reach_diff, age_diff, avg_TD_landed_diff, sig_str_diff]])

                Winner = model.predict(X)
                proba = model.predict_proba(X)
                if proba[0][0] > proba[0][1]: proba = proba[0][0]*100
                else: proba = proba[0][1]*100
                st.title("")
                st.title("")
                if Winner[0] == 0:
                    st.write("""## Winner:""")
                    st.image(bImage)
                    st.title("")
                    st.write("""### {}""".format(Blue_Fighter))
                    st.write("""#### Prediction Probability {} wins: {:.2f}%""".format(Blue_Fighter, proba))
                else:
                    st.write("""## Winner:""")
                    st.image(rImage)
                    st.title("")
                    st.write("""### {}""".format(Red_Fighter))
                    st.write("""#### Prediction Probability {} wins: {:.2f}%""".format(Red_Fighter, proba))
                