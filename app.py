import streamlit as st
import requests
import time

# ---------- Streamlit Layout ----------
st.title("Banking Support Ticket System")

st.subheader("Submit a Ticket")

# Ticket input form
with st.form(key="ticket_form"):
    channel = st.selectbox("Channel", ["Email", "Chat", "Phone"])
    severity = st.selectbox("Severity", ["High", "Medium", "Low"])
    summary = st.text_area("Summary")
    submit_button = st.form_submit_button(label="Submit Ticket")

# ---------- API Call ----------
API_URL = "http://localhost:8000/ticket"  # replace with your backend API
RESULT_URL = "http://localhost:8000/result"  # optional endpoint for async results

if submit_button:
    ticket_data = {
        "channel": channel,
        "severity": severity,
        "summary": summary
    }

    try:
        response = requests.post(API_URL, json=ticket_data)
        if response.status_code == 200:
            res = response.json()

            # Check if ticket is async
            if res.get("status") == "queued":
                st.info(f"Ticket queued with Job ID: {res['ticket_id']}")
                result_placeholder = st.empty()
                
                # Poll API until result is ready
                while True:
                    result_resp = requests.get(f"{RESULT_URL}/{res['ticket_id']}")
                    if result_resp.status_code == 200:
                        result_data = result_resp.json()
                        if result_data.get("status") == "completed":
                            result_placeholder.success("Result:")
                            result_placeholder.write(f"Decision: {result_data['result']['decision']}")
                            result_placeholder.write(f"Reason: {result_data['result']['reason']}")
                            result_placeholder.write("Next Actions:")
                            for step in result_data['result']['next_actions']:
                                result_placeholder.write(f"- {step}")
                            break
                        else:
                            result_placeholder.info("Processing...")
                    time.sleep(2)

            else:
                # Sync ticket
                st.success("Result:")
                st.write(f"Decision: {res['decision']}")
                st.write(f"Reason: {res['reason']}")
                st.write("Next Actions:")
                for step in res['next_actions']:
                    st.write(f"- {step}")

        else:
            st.error("Error submitting ticket")
    except Exception as e:
        st.error(f"API request failed: {e}")
