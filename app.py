import streamlit as st
import requests
import time
import json

# --- THIS IS THE NEW LINE ---
st.set_page_config(layout="wide")
# ----------------------------

# ---------- Streamlit Layout ----------
st.title("Banking Support Ticket System")
st.subheader("Submit a Ticket")

# --- NEW: Create two columns ---
col1, col2 = st.columns(2)

# --- Column 1: The Form ---
with col1:
    with st.form(key="ticket_form"):
        channel = st.selectbox("Channel", ["Email", "Chat", "Phone"])
        
        # Updated selectbox for clarity
        severity_option = st.selectbox(
            "Severity (Determines Processing Path)", 
            [
                "High (Sync - ~5 sec)", 
                "Medium (Async - ~30 sec)", 
                "Low (Async - ~30 sec)"
            ]
        )
        
        summary = st.text_area("Summary", "Example: My credit card payment is not going through.")
        submit_button = st.form_submit_button(label="Submit Ticket")

# Map user-friendly option to the API value
severity_mapping = {
    "High (Sync - ~5 sec)": "High",
    "Medium (Async - ~30 sec)": "Medium",
    "Low (Async - ~30 sec)": "Low"
}
severity = severity_mapping[severity_option]

# ---------- API Call ----------
API_URL = "http://localhost:8000/ticket"  # Orchestrator
RESULT_URL = "http://localhost:8000/result" # Orchestrator's result endpoint

# --- Column 2: The Results ---
with col2:
    if submit_button:
        ticket_data = {
            "channel": channel,
            "severity": severity,
            "summary": summary
        }
        
        start_time = time.time() # Start round-trip timer

        # --- NEW: Define the API call logic as a reusable function ---
        def make_api_call():
            try:
                response = requests.post(API_URL, json=ticket_data)
                
                if response.status_code == 200:
                    try:
                        res = response.json()
                    except json.JSONDecodeError:
                        st.error(f"Error: Could not decode JSON response from orchestrator. Response: {response.text}")
                        res = None

                    if res:
                        # Check if ticket is async
                        if res.get("status") == "queued":
                            # --- UPDATED ASYNC MESSAGE ---
                            st.success(f"Your ticket has been submitted to our support team (Job ID: {res['ticket_id']})")
                            st.info("Our team is reviewing the issue and will get back to you as soon as the problem is traced. The results will appear here automatically.")
                            result_placeholder = st.empty()
                            
                            # Poll API until result is ready
                            while True:
                                result_resp = requests.get(f"{RESULT_URL}/{res['ticket_id']}")
                                
                                if result_resp.status_code == 200:
                                    try:
                                        result_data = result_resp.json()
                                    except json.JSONDecodeError:
                                        result_placeholder.error(f"Error: Could not decode JSON response from result endpoint. Response: {result_resp.text}")
                                        break

                                    if result_data.get("status") == "completed":
                                        result_placeholder.empty() # Clear the "processing" message
                                        st.success("Support Team Response:")
                                        
                                        result = result_data.get('result', {})
                                        st.write(f"**Decision:** {result.get('decision', 'N/A')}")
                                        st.write(f"**Reason:** {result.get('reason', 'N/A')}")
                                        st.write("**Next Actions:**")
                                        for step in result.get('next_actions', []):
                                            st.write(f"- {step}")
                                        
                                        # Display Processing Time
                                        processing_time = result.get("processing_time")
                                        if processing_time:
                                            st.write(f"**AI Processing Time:** {processing_time:.2f} seconds")

                                        retrieved_context = result.get("retrieved_context")
                                        if retrieved_context:
                                            with st.expander("Show RAG Context"):
                                                st.text(retrieved_context)
                                        
                                        break # Exit polling loop
                                    
                                    # --- UPDATED ASYNC POLLING MESSAGES ---
                                    elif result_data.get("status") == "processing":
                                        result_placeholder.info("Our support team is actively reviewing your ticket now...")
                                    
                                    elif result_data.get("status") == "queued":
                                        result_placeholder.info("Your ticket is in the queue. Our team will review it shortly.")
                                        
                                    elif result_data.get("status") == "error":
                                        result_placeholder.error(f"Error processing ticket: {result_data.get('detail')}")
                                        break

                                    else:
                                        result_placeholder.info(f"Waiting for result... (Status: {result_data.get('status')})")
                                
                                elif result_resp.status_code == 404:
                                    st.error("Error: Result endpoint not found (404). Check Orchestrator (`sync_async_routing_API.py`).")
                                    break
                                else:
                                    st.error(f"Error polling for result: {result_resp.status_code} - {result_resp.text}")
                                    break
                                
                                time.sleep(2)

                        # Sync ticket
                        elif res.get("decision"):
                            total_time = time.time() - start_time
                            # --- UPDATED SYNC MESSAGE ---
                            st.success("Received Real-Time Support Response:")
                            
                            st.write(f"**Decision:** {res['decision']}")
                            st.write(f"**Reason:** {res['reason']}")
                            st.write("**Next Actions:**")
                            for step in res['next_actions']:
                                st.write(f"- {step}")
                            
                            # Display Processing Time
                            processing_time = res.get("processing_time")
                            if processing_time:
                                st.write(f"**AI Processing Time:** {processing_time:.2f} seconds")
                            
                            st.write(f"**Total round-trip time:** {total_time:.2f} seconds")

                            retrieved_context = res.get("retrieved_context")
                            if retrieved_context:
                                with st.expander("Show RAG Context"):
                                    st.text(retrieved_context)
                        
                        else:
                            st.error(f"Error: Unknown response from orchestrator: {res}")

                else:
                    st.error(f"Error submitting ticket: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("API connection failed. Is the Orchestrator (port 8000) running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        
        # --- NEW: Conditional spinner logic ---
        if severity == "High":
            # SYNC: Show spinner while waiting
            with st.spinner("Contacting support... Please wait."):
                make_api_call()
        else:
            # ASYNC: No spinner, just make the call.
            # The "queued" status from make_api_call() will appear instantly.
            make_api_call()

