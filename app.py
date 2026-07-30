"""
Bank Term Deposit — Subscription Likelihood Predictor
Streamlit web app for the MLDP project (CAI2C08).

Run locally:   streamlit run app.py
The app loads 'bank_model.pkl', produced by running the Jupyter notebook.
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Term Deposit Predictor",
                   page_icon="🏦", layout="centered")

# The model pipeline expects these columns, in this exact order.
FEATURE_ORDER = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
    'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed', 'was_previously_contacted',
]

JOBS = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown']
MARITAL = ['married', 'single', 'divorced', 'unknown']
EDUCATION = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
             'professional.course', 'university.degree', 'unknown']
YES_NO_UNKNOWN = ['no', 'yes', 'unknown']
CONTACT = ['cellular', 'telephone']
MONTHS = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
DAYS = ['mon', 'tue', 'wed', 'thu', 'fri']
POUTCOME = ['nonexistent', 'failure', 'success']


@st.cache_resource
def load_model():
    return joblib.load("bank_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file 'bank_model.pkl' not found. "
             "Run the Jupyter notebook first to generate it, and keep it in "
             "the same folder as this app.")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"⚠️ Could not load the model: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏦 Term Deposit Subscription Predictor")
st.write(
    "Enter a client's profile and campaign context. The model estimates how "
    "likely the client is to **subscribe to a term deposit**, so the call "
    "centre can prioritise the clients most worth contacting."
)
st.divider()

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("👤 Client profile")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
    job = st.selectbox("Job", JOBS, index=JOBS.index('admin.'))
    marital = st.selectbox("Marital status", MARITAL)
    education = st.selectbox("Education", EDUCATION,
                             index=EDUCATION.index('university.degree'))
with c2:
    default = st.selectbox("Has credit in default?", YES_NO_UNKNOWN)
    housing = st.selectbox("Has housing loan?", YES_NO_UNKNOWN,
                           index=YES_NO_UNKNOWN.index('yes'))
    loan = st.selectbox("Has personal loan?", YES_NO_UNKNOWN)

st.subheader("📞 Campaign contact")
c3, c4 = st.columns(2)
with c3:
    contact = st.selectbox("Contact type", CONTACT)
    month = st.selectbox("Last contact month", MONTHS, index=MONTHS.index('may'))
    day_of_week = st.selectbox("Last contact day", DAYS)
with c4:
    campaign = st.number_input("Contacts during this campaign",
                               min_value=1, max_value=60, value=2, step=1)
    prev_contacted = st.checkbox("Contacted in a previous campaign?", value=False)

# The engineered flag and its related fields depend on the checkbox above.
if prev_contacted:
    was_previously_contacted = 1
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        pdays = st.number_input("Days since last contact", min_value=0,
                                max_value=40, value=6, step=1)
    with cc2:
        previous = st.number_input("Previous contacts", min_value=1,
                                   max_value=10, value=1, step=1)
    with cc3:
        poutcome = st.selectbox("Previous outcome", ['failure', 'success'])
else:
    was_previously_contacted = 0
    pdays = 0            # matches the notebook: the 999 sentinel is reset to 0 in training
    previous = 0
    poutcome = 'nonexistent'

with st.expander("📈 Economic context (advanced — sensible defaults preset)"):
    e1, e2 = st.columns(2)
    with e1:
        emp_var_rate = st.number_input("Employment variation rate",
                                       value=1.1, step=0.1, format="%.1f")
        cons_price_idx = st.number_input("Consumer price index",
                                         value=93.994, step=0.1, format="%.3f")
        cons_conf_idx = st.number_input("Consumer confidence index",
                                        value=-36.4, step=0.1, format="%.1f")
    with e2:
        euribor3m = st.number_input("Euribor 3-month rate", value=4.857,
                                    step=0.1, format="%.3f")
        nr_employed = st.number_input("Number employed (thousands)",
                                      value=5191.0, step=1.0, format="%.1f")

st.divider()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("🔮 Predict subscription likelihood", type="primary",
             use_container_width=True):
    row = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact, 'month': month, 'day_of_week': day_of_week,
        'campaign': campaign, 'pdays': pdays, 'previous': previous,
        'poutcome': poutcome, 'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m, 'nr.employed': nr_employed,
        'was_previously_contacted': was_previously_contacted,
    }
    X_new = pd.DataFrame([row])[FEATURE_ORDER]

    try:
        proba = float(model.predict_proba(X_new)[0, 1])
        pred = int(model.predict(X_new)[0])
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Result")
    m1, m2 = st.columns(2)
    m1.metric("Predicted probability of subscribing", f"{proba*100:.1f}%")
    m2.metric("Decision", "Likely to subscribe ✅" if pred == 1
              else "Unlikely to subscribe ❌")
    st.progress(proba)

    if pred == 1:
        st.success("**Recommend calling this client** — above the model's "
                   "decision threshold for subscribing.")
    else:
        st.warning("**Lower priority for calling** — below the decision "
                   "threshold. Call-centre time is likely better spent elsewhere.")

    st.caption("Model: tuned Random Forest trained on the UCI Bank Marketing "
               "dataset. Probabilities are estimates, not guarantees.")
else:
    st.info("Set the client details above, then click **Predict** to see the "
            "estimated subscription likelihood.")
