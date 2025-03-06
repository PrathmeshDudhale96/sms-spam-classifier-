import streamlit as st
import imaplib
import pickle
import email
from email import message_from_bytes
from email.header import decode_header
import numpy as np

# Load trained model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to extract email body
def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                return part.get_payload(decode=True).decode(errors="ignore")
    else:
        return msg.get_payload(decode=True).decode(errors="ignore")
    return None  

# Function to fetch latest emails
def fetch_latest_emails(username, password, imap_server='imap.gmail.com', num_emails=10):
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(username, password)
        mail.select("inbox")

        status, messages = mail.search(None, 'ALL')
        email_ids = messages[0].split()
        latest_email_ids = email_ids[-num_emails:]

        emails = []
        for email_id in latest_email_ids:
            _, msg_data = mail.fetch(email_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else 'utf-8')
                    from_ = msg.get("From")
                    body = get_email_body(msg)
                    if body:
                        emails.append({'subject': subject, 'from': from_, 'body': body})

        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

# Function to classify emails
def classify_emails(emails, model, vectorizer):
    if not emails:
        return []
    email_bodies = [email['body'] for email in emails]
    email_tfidf = vectorizer.transform(email_bodies)
    predictions = model.predict(email_tfidf)
    return predictions

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", page_icon="📩", layout="wide")

# Header with color
st.markdown(
    "<h1 style='text-align: center; color: white; background-color: #6A1B9A; padding: 10px;'>📩 Email Spam Classifier</h1>",
    unsafe_allow_html=True,
)

st.markdown("### 📧 Enter Email Credentials to Check Spam")

# Sidebar with description
st.sidebar.title("🔍 About This App")
st.sidebar.info(
    "This tool fetches your latest emails and classifies whether they are **Spam or Not** "
    "using Machine Learning (TF-IDF & Naive Bayes)."
)

# Input fields
email_id = st.text_input("📬 Enter Email ID", placeholder="your-email@gmail.com")
password = st.text_input("🔒 Enter Password", placeholder="Enter App Password", type="password")

# Button to fetch and classify emails
if st.button("🚀 Check Emails"):
    if email_id and password:
        with st.spinner("Fetching emails... Please wait!"):
            emails = fetch_latest_emails(email_id, password)

        if emails:
            predictions = classify_emails(emails, best_model, vectorizer)
            st.markdown("## 📩 Email Classification Results")

            for email_data, prediction in zip(emails, predictions):
                label = "🛑 **Spam**" if prediction == 1 else "✅ **Not Spam**"
                color = "red" if prediction == 1 else "green"

                st.markdown(
                    f"""
                    <div style='border: 2px solid {color}; padding: 10px; border-radius: 10px;'>
                        <b>📌 From:</b> {email_data['from']}  
                        <b>📢 Subject:</b> {email_data['subject']}  
                        <b>📝 Classification:</b> <span style='color: {color};'>{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("\n---")
        else:
            st.warning("No emails found or incorrect credentials. Try again.")

# Footer
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "🚀 Built with ❤️ by Prathmesh Dudhale</div>",
    unsafe_allow_html=True,
)
