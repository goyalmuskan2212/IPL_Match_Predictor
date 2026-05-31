
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="centered")

@st.cache_data
def load_and_train():
    matches = pd.read_csv('matches.csv')
    matches = matches[matches['winner'].notna()]
    matches = matches[matches['result'] == 'normal']

    win_counts   = matches['winner'].value_counts().to_dict()
    total_counts = pd.concat([matches['team1'], matches['team2']]).value_counts().to_dict()

    matches['team1_winrate']  = matches['team1'].map(lambda t: win_counts.get(t,0)/total_counts.get(t,1))
    matches['team2_winrate']  = matches['team2'].map(lambda t: win_counts.get(t,0)/total_counts.get(t,1))
    matches['winrate_diff']   = matches['team1_winrate'] - matches['team2_winrate']
    matches['team1_won_toss'] = (matches['toss_winner'] == matches['team1']).astype(int)
    matches['team1_won']      = (matches['winner'] == matches['team1']).astype(int)

    le_team  = LabelEncoder()
    le_venue = LabelEncoder()
    all_teams = pd.concat([matches['team1'], matches['team2']]).unique()
    le_team.fit(all_teams)

    matches['team1_enc'] = le_team.transform(matches['team1'])
    matches['team2_enc'] = le_team.transform(matches['team2'])
    matches['venue_enc'] = le_venue.fit_transform(matches['venue'])

    features = ['team1_enc','team2_enc','venue_enc','team1_won_toss','winrate_diff']
    X = matches[features]
    y = matches['team1_won']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    teams  = sorted(le_team.classes_.tolist())
    venues = sorted(le_venue.classes_.tolist())
    return model, le_team, le_venue, teams, venues, win_counts, total_counts, features

model, le_team, le_venue, teams, venues, win_counts, total_counts, features = load_and_train()

st.title("🏏 IPL Match Win Predictor")
st.markdown("Select two teams and a venue to predict the match winner.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("🔵 Team 1", teams, index=teams.index("Mumbai Indians"))
with col2:
    team2 = st.selectbox("🔴 Team 2", [t for t in teams if t != team1], index=0)

venue = st.selectbox("📍 Venue", venues)
st.divider()

if st.button("⚡ Predict Winner", use_container_width=True):
    if team1 == team2:
        st.error("Please select two different teams!")
    else:
        t1_enc  = le_team.transform([team1])[0]
        t2_enc  = le_team.transform([team2])[0]
        v_enc   = le_venue.transform([venue])[0]
        t1_wr   = win_counts.get(team1, 0) / total_counts.get(team1, 1)
        t2_wr   = win_counts.get(team2, 0) / total_counts.get(team2, 1)
        wr_diff = t1_wr - t2_wr

        row   = pd.DataFrame([[t1_enc, t2_enc, v_enc, 0, wr_diff]], columns=features)
        proba = model.predict_proba(row)[0]
        t1_prob = round(proba[1] * 100, 1)
        t2_prob = round(proba[0] * 100, 1)
        winner  = team1 if t1_prob > t2_prob else team2

        st.success(f"🏆 Predicted Winner: **{winner}**")
        st.markdown("### Win Probability")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label=team1, value=f"{t1_prob}%")
            st.progress(int(t1_prob))
        with col_b:
            st.metric(label=team2, value=f"{t2_prob}%")
            st.progress(int(t2_prob))

        st.markdown("### Historical Win Rates")
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric(f"{team1} overall", f"{t1_wr*100:.1f}%")
        with col_d:
            st.metric(f"{team2} overall", f"{t2_wr*100:.1f}%")

        st.caption("Model: Logistic Regression | Dataset: IPL 2008-2019")
