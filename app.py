import streamlit as st
import requests
import time

# ---------- Streamlit Layout ----------
st.set_page_config(layout="wide")
st.title("Banking Support Ticket System")

st.subheader("Submit a Ticket")

# Ticket input form
with st.form(key="ticket_form"):
    channel = st.selectbox("Channel", ["Email", "Chat", "Phone"])
    severity = st.selectbox("Severity", ["High", "Medium", "Low"])
    summary = st.text_area("Summary", "Example: My credit card payment is not going through.")
    submit_button = st.form_submit_button(label="Submit Ticket")

# Define API URLs
API_URL = "http://localhost:8000/ticket"
RESULT_URL = "http://localhost:8000/result" 

if submit_button:
    if not summary.strip():
        st.error("Please provide a summary for the ticket.")
    else:
        ticket_data = {
            "channel": channel,
            "severity": severity,
            "summary": summary
        }

        try:
            # Main API call to the orchestrator
            response = requests.post(API_URL, json=ticket_data)
            
            if response.status_code == 200:
                res = response.json()

                # --- Handle ASYNC (Medium/Low Severity) ---
                if res.get("status") == "queued":
                    st.info(f"Ticket submitted (Async). Job ID: {res['ticket_id']}")
                    st.write("Polling for results...")
                    result_placeholder = st.empty()
                    
                    # Poll API until result is ready
                    while True:
                        try:
                            result_resp = requests.get(f"{RESULT_URL}/{res['ticket_id']}")
                            
                            if result_resp.status_code == 200:
                                result_data = result_resp.json()

                                # Check the status from the result store
                                if result_data.get("status") == "completed":
                                    result_placeholder.success("Result:")
                                    result_json = result_data.get("result", {})
                                    st.write(f"**Decision:** {result_json.get('decision')}")
                                    st.write(f"**Reason:** {result_json.get('reason')}")
                                    st.write("**Next Actions:**")
                                    for step in result_json.get('next_actions', []):
                                        st.write(f"- {step}")
                                    break # Exit polling loop
                                
                                elif result_data.get("status") == "error":
                                    result_placeholder.error(f"Async processing failed: {result_data.get('detail')}")
                                    break # Exit polling loop
                                
                                else:
                                    # Status is "pending" or unknown, keep polling
                                    result_placeholder.info("Processing... please wait.")
                                    time.sleep(2)
                            else:
                                result_placeholder.warning(f"Result endpoint error: {result_resp.status_code}")
                                time.sleep(2)

                        except requests.ConnectionError as ce:
                            st.warning(f"Polling failed to connect: {ce}. Retrying...")
                            time.sleep(2)


                # --- Handle SYNC (High Severity) ---
                else:
                    # Check if the sync call itself returned an error
                    if res.get("status") == "error":
                        st.error(f"Sync processing failed: {res.get('detail')}")
                    else:
                        st.success("Result (Sync):")
                        st.write(f"**Decision:** {res.get('decision')}")
                        st.write(f"**Reason:** {res.get('reason')}")
                        st.write("**Next Actions:**")
                        for step in res.get('next_actions', []):
                            st.write(f"- {step}")
            
            else:
                # Handle non-200 from the main orchestrator
                st.error(f"Error submitting ticket to {API_URL}. Status: {response.status_code}")
                st.json(response.text)

        except requests.ConnectionError as e:
            st.error(f"API request failed: Could not connect to the backend at {API_URL}.")
            st.error("Please ensure the backend services (orchestrator, sync, async) are running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
