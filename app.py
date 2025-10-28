import streamlit as st
import requests
import time
import json

# --- Page Config ---
st.set_page_config(layout="wide")

# ---------- Streamlit Layout ----------
st.title("Banking Support Ticket System")

# --- Use Columns for Layout ---
col1, col2 = st.columns(2)

# --- Left Column: Form and Clear Button ---
with col1:
    st.subheader("Submit a Ticket")
    # Ticket input form
    with st.form(key="ticket_form"):
        channel = st.selectbox("Channel", ["Email", "Chat", "Phone"])

        # Updated selectbox for clarity
        severity_option = st.selectbox(
            "Severity (Determines Processing Path)",
            [
                "High (Sync - AI response ~5-10s)",
                "Medium (Async - Queued, AI response ~10-15s)",
                "Low (Async - Queued, AI response ~10-15s)"
            ]
        )

        summary = st.text_area("Summary", "Example: My credit card payment is not going through.")
        submit_button = st.form_submit_button(label="Submit Ticket")

    # Map user-friendly option to the API value
    severity_mapping = {
        "High (Sync - AI response ~5-10s)": "High",
        "Medium (Async - Queued, AI response ~10-15s)": "Medium",
        "Low (Async - Queued, AI response ~10-15s)": "Low"
    }
    severity = severity_mapping[severity_option]

    # --- Clear Memory Button (Moved Here) ---
    st.divider() # Add a visual separator above the button
    CLEAR_MEMORY_URL = "http://localhost:8000/clear_memory" # Orchestrator endpoint

    if st.button("Clear RAG Memory"):
        try:
            response = requests.post(CLEAR_MEMORY_URL, timeout=10) # Add timeout
            if response.status_code == 200:
                st.success("Successfully requested memory clear for both services.")
                st.json(response.json()) # Show detailed results
            else:
                st.error(f"Error clearing memory: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to clear memory endpoint: {e}")
    # --- End Clear Memory Button ---


# --- Right Column: API Call and Results Display ---
with col2:
    st.subheader("Results")
    # Use ONE main placeholder in the results column for all dynamic content
    results_placeholder = st.empty()

    # ---------- API Call Logic ----------
    API_URL = "http://localhost:8000/ticket"  # Orchestrator
    RESULT_URL = "http://localhost:8000/result" # Orchestrator's result endpoint

    if submit_button:
        ticket_data = {
            "channel": channel,
            "severity": severity,
            "summary": summary
        }

        # --- Use the placeholder's container context manager ---
        with results_placeholder.container():
            try:
                # Show spinner only for Sync path, inside the container
                if severity == "High":
                    with st.spinner("Processing ticket in real-time..."):
                        start_time = time.time() # Start round-trip timer for sync only
                        response = requests.post(API_URL, json=ticket_data, timeout=60) # Increased timeout
                else:
                    # For Async, just make the request, no spinner needed here
                     response = requests.post(API_URL, json=ticket_data, timeout=15) # Shorter timeout for async submission

                # --- Process Response ---
                if response.status_code == 200:
                    try:
                        res = response.json()
                    except json.JSONDecodeError:
                        st.error(f"Error: Could not decode JSON response. Status: {response.status_code}, Response: {response.text}")
                        res = None

                    if res:
                        # --- Async Path ---
                        if res.get("status") == "queued":
                            st.success(f"Your ticket has been submitted (Job ID: {res['ticket_id']})")
                            polling_placeholder = st.empty() # New placeholder JUST for polling status within this container
                            polling_placeholder.info("Our team is reviewing. Results appear here when ready.")

                            # Poll API until result is ready
                            while True:
                                try:
                                    result_resp = requests.get(f"{RESULT_URL}/{res['ticket_id']}", timeout=10)

                                    if result_resp.status_code == 200:
                                        try:
                                            result_data = result_resp.json()
                                        except json.JSONDecodeError:
                                            polling_placeholder.error(f"Error: Could not decode result JSON. Response: {result_resp.text}")
                                            break # Stop polling on decode error

                                        if result_data.get("status") == "completed":
                                            polling_placeholder.empty() # Clear polling message
                                            st.success("Support Update:") # Display final results

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

                                            # Display RAG Context
                                            retrieved_context = result.get("retrieved_context")
                                            if retrieved_context and retrieved_context != "No relevant past cases found.":
                                                with st.expander("Show RAG Context Used"):
                                                    st.text(retrieved_context)
                                            elif retrieved_context: # Still show if no cases found
                                                 with st.expander("Show RAG Context Used"):
                                                    st.info(retrieved_context)

                                            break # Exit polling loop

                                        elif result_data.get("status") == "processing":
                                            polling_placeholder.info("Status: AI worker processing...") # Update polling status
                                        elif result_data.get("status") == "queued":
                                             polling_placeholder.info("Status: Waiting in queue...") # Update polling status
                                        elif result_data.get("status") == "error":
                                            polling_placeholder.error(f"Error processing ticket: {result_data.get('detail')}")
                                            break
                                        else: # Pending or other unknown status
                                            polling_placeholder.info(f"Status: {result_data.get('status', 'Waiting...')}") # Update polling status

                                    elif result_resp.status_code == 404:
                                        polling_placeholder.error("Error: Result endpoint not found (404).")
                                        break
                                    else:
                                        polling_placeholder.error(f"Error polling: {result_resp.status_code} - {result_resp.text}")
                                        break

                                except requests.exceptions.Timeout:
                                    polling_placeholder.warning("Polling timed out. Still waiting...") # Update polling status
                                except requests.exceptions.ConnectionError:
                                     polling_placeholder.error("Connection error polling.")
                                     break
                                except Exception as poll_e:
                                    polling_placeholder.error(f"Polling error: {poll_e}")
                                    break

                                time.sleep(3) # Wait longer between polls

                        # --- Sync Path ---
                        elif res.get("decision"):
                            # Spinner already cleared automatically when 'with' block exits
                            total_time = time.time() - start_time
                            st.success("Real-time Support Response:") # Display results

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

                            # Display RAG Context
                            retrieved_context = res.get("retrieved_context")
                            if retrieved_context and retrieved_context != "No relevant past cases found.":
                                 with st.expander("Show RAG Context Used"):
                                    st.text(retrieved_context)
                            elif retrieved_context: # Still show if no cases found
                                 with st.expander("Show RAG Context Used"):
                                    st.info(retrieved_context)


                        # --- Error from Orchestrator's initial call ---
                        elif res.get("status") == "error":
                             st.error(f"Error from backend: {res.get('detail')}")
                        else:
                            st.error(f"Error: Unknown response format: {res}")

                # --- HTTP Error from Orchestrator ---
                else:
                     st.error(f"Error submitting ticket: {response.status_code} - {response.text}")

            # --- Connection or other request errors ---
            except requests.exceptions.Timeout:
                 st.error("API request timed out.")
            except requests.exceptions.ConnectionError:
                st.error("API connection failed. Is the backend running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

