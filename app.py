import os 
import json
import joblib 
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title= 'FinTech Churn Intelligence', layout= 'wide')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MODELS_DIR, 'best_churn_pipeline.pkl'))
    with open(os.path.join(MODELS_DIR, 'optimal_threshold.json'), 'r') as f:
        config = json.load(f)
    return model, config['champion_model'], config['threshold']
    
model, model_name, threshold = load_assets()

st.title('FinTech Saving Churn Intelligence Platform')
st.markdown(f"**Model in Deployment: ** '{model_name}' | ** Optimal Threshold: ** '{threshold:.3f}'")

tab1, tab2, tab3 = st.tabs(['Single Customer Scorer', 'Portfolio Risk Analysis', 'Batch CSV Scoring'])

with tab1:
    st.subheader('Simulate Individual Customer Profile')
    with st.form('score_form'):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input('Age', 18, 100, 32)
            region = st.selectbox('Region', ['Metro', 'Tier 1', 'Tier 2', 'Rural'])
            employment = st.selectbox('Employment', ['Salaried', 'Self-Employed', 'Freelancer', 'Student'])
            annual_income = st.number_input('Annual Income (₹)',5000, 10000000, 650000, step= 25000)
            tennure = st.number_input('Tennure (Months)', 1, 120, 24)
            app_plan = st.selectbox('App Plan', ['Basic', 'Standared', 'Premium'])

        with c2:
            risk_seg = st.selectbox('Risk Segment', ['Low', 'Medium', 'High'])
            monthly_trans = st.number_input('Monthly Transaction', 0, 300, 15)
            monthly_spend = st.number_input('Monthly Spend (₹)', 0, 10000000, 18000, step= 1000)
            avg_balance = st.number_input('Average Balance (₹)', 0, 50000000, 45000, step= 5000)
            failed_payments = st.number_input('Failed Payment (6m)', 0, 20, 1)

        with c3:
            support_tickets = st.number_input('Support Tickets (6m)', 0, 20, 2)
            days_since_last_login = st.number_input('Days Since Last Login', 0, 180, 15)
            has_goal = st.selectbox('Has Saving Goal ?', [1, 0], format_func= lambda x: 'Yes' if x == 1 else 'No')
            auto_save = st.selectbox('Auto-Save Enabled ?', [1, 0], format_func= lambda x: 'Yes' if x == 1 else 'No')

        submitted = st.form_submit_button('Predict Churn Risk', use_container_width= True)

    if submitted:
        spend_to_income = (monthly_spend * 12) / (annual_income + 1)
        bal_to_spend = avg_balance / (monthly_spend + 1)
        friction = failed_payments + (support_tickets * 1.5)

        input_df = pd.DataFrame([{
            'age' : age, 
            'region' : region,
            'employment_type' : employment,
            'annual_income_inr' : annual_income,
            'account_tenure_months' : tennure,
            'app_plan' : app_plan,
            'risk_segment' : risk_seg,
            'monthly_transactions' : monthly_trans,
            'monthly_spend_inr' : monthly_spend,
            'average_balance_inr' : avg_balance,
            'failed_payments_6m' : failed_payments,
            'support_tickets_6m' : support_tickets,
            'days_since_last_login' : days_since_last_login,
            'has_savings_goal' : has_goal,
            'auto_save_enabled' : auto_save,
            'spend_to_income_ratio' : spend_to_income,
            'balance_to_spend_ratio' : bal_to_spend,
            'friction_score' : friction
        }])

        prob = model.predict_proba(input_df)[0, 1]

        st.markdown('---')
        m1, m2, m3 = st.columns(3)
        m1.metric('Predicted Churn Probability', f'{prob * 100:.2f}%')

        if prob >= 0.65:
            m2.error('Risk Tier: HIGH RISK')
            m3.warning('Action: Immediate Relationship Manager Outreach & Fee Check')
        elif prob >= 0.35:
            m2.warning('Risk Tier: MEDIUM RISk')
            m3.info('Action: Trigger In-App Savings Incentives & Push Notifications')
        else:
            m2.success('Risk Tier: LOW RISK')
            m3.write('Action: Standared Engagement & Financial Product Cross-sell')

with tab2:
    st.subheader('Customer Portfolio Overview')
    pred_path = os.path.join('data', 'customer_churn_risk_prediction.csv')
    if os.path.exists(pred_path):
        report_df = pd.read_csv(pred_path)
        r1, r2, r3 = st.columns(3)
        r1.metric('Total Customer Evaluated', f'{len(report_df):,}')
        hhigh_risk_n = (report_df['risk_category'] == 'High Risk').sum()
        r2.metric('High-Risk Accounts', f'{high_risk_n:,}', delta= f'{(high_risk_n/len(report_df)) *100:.1f}%')
        r3.metric('Decision Cutoff Threshold', f'{threshold:.3f}')

        st.bar_chart(report_df['risk_category'].value_counts())
        st.dataframe(report_df.head(50), use_container_width= True)
    else:
        st.info('Run the batch scoring cell to see portfolio predictions.')

with tab3:
    st.subheader('Batch File Prediction')
    uploaded_file = st.file_uploader('Upload customer CSV file', type=['csv'])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_df["spend_to_income_ratio"] = (batch_df["monthly_spend_inr"] * 12) / (batch_df["annual_income_inr"] + 1)
        batch_df["balance_to_spend_ratio"] = batch_df["average_balance_inr"] / (batch_df["monthly_spend_inr"] + 1)
        batch_df["friction_score"] = batch_df["failed_payments_6m"] + (batch_df["support_tickets_6m"] * 1.5)
            
        predict = model.predict_proba(batch_df)[:, 1]
        batch_df['churn_probability'] = np.round(predict, 4)
        batch_df['risk_category'] = np.where(predict >= threshold, 'High Risk', np.where(predict >= 0.35, 'Medium Risk', 'Low Risk'))

        st.dataframe(batch_df[['customer_id', 'churn_probability', 'risk_category']].head(25), use_container_width= True)
        csv_bytes = batch_df.to_csv(index= False).encode('utf-8')
        st.download_button('Download Prediction CSV', csv_bytes, 'batch_churn_prediction.csv', 'text/csv')