import streamlit as st
import pandas as pd
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# ================= HEADER =================
st.markdown("## ⚙️ Predictive Maintenance Platform")
st.info(
    "🚧 **Prediction model not integrated yet**\n\n"
    "This application demonstrates UI structure and CSV-based data ingestion. "
    "Prediction outputs will be enabled once the ML model is connected."
)

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Dashboard", "📥 Data Upload", "📊 Fleet Overview", "📄 Reports"]
)

# ================= DASHBOARD =================
if page == "🏠 Dashboard":
    st.subheader("Factory Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Machines", "—")
    col2.metric("Healthy", "—")
    col3.metric("Warning", "—")
    col4.metric("Critical", "—")

    st.markdown("---")
    st.write("Machine status will appear here once prediction engine is enabled.")

# ================= DATA UPLOAD =================
elif page == "📥 Data Upload":
    st.subheader("Upload Machine Sensor Data (CSV)")

    st.markdown("### Expected CSV Format")
    st.code(
        "machine_id,machine_type,temperature,vibration,pressure,rpm,operating_cycles\n"
        "MTR-01,motor,0.45,0.32,0.58,0.70,120"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully")
        st.subheader("Preview Data")
        st.dataframe(df, use_container_width=True)

        st.warning("Prediction engine not connected yet. Uploaded data is stored for processing.")

    st.button("Send to Prediction Engine", disabled=True)

# ================= FLEET OVERVIEW =================
elif page == "📊 Fleet Overview":
    st.subheader("Fleet Overview")

    st.write("Fleet-level analytics will be shown here after model integration.")

    placeholder_df = pd.DataFrame({
        "Machine ID": ["—"],
        "Risk Level": ["—"],
        "Maintenance Priority": ["—"]
    })

    st.dataframe(placeholder_df, use_container_width=True)

    fig = px.bar(
        x=["Healthy", "Warning", "Critical"],
        y=[0, 0, 0],
        title="Fleet Health Distribution (Placeholder)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= REPORTS =================
elif page == "📄 Reports":
    st.subheader("Maintenance Report Generator")

    st.selectbox("Select Machine", ["—"])
    st.selectbox("Report Format", ["Text", "HTML", "JSON"])

    st.button("Generate Report", disabled=True)

    st.text_area(
        "Report Preview",
        value="Reports will be generated after prediction engine integration.",
        height=250
    )
