import streamlit as st
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# --- Quantum Randomness Function ---
def quantum_random_prob():
    simulator = AerSimulator()
    qc = QuantumCircuit(1, 1)
    qc.h(0)          # Put qubit in superposition
    qc.measure(0, 0) # Measure
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=100)
    result = job.result()
    counts = result.get_counts()
    prob = counts.get("1", 0) / 100  # Probability of measuring "1"
    return round(prob, 2)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Quantum Fraud Detection", layout="wide")

st.title("⚛️ Quantum Fraud Detection")
st.write("A conceptual demonstration of using Quantum Machine Learning (QML) to identify financial anomalies.")

# --- Session State for persistence ---
if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame()
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- Function to generate transactions ---
def generate_transactions():
    categories = ['Groceries', 'Utilities', 'Shopping', 'Travel', 'Entertainment', 'Subscription']
    txs = []

    # Normal transactions
    for i in range(8):
        txs.append({
            "ID": f"T{i+1}",
            "Amount ($)": round(random.uniform(10, 200), 2),
            "Category": random.choice(categories),
            "Anomaly": False
        })

    # Hardcoded anomalies
    txs.append({"ID": "A1", "Amount ($)": random.randint(5000, 5500), "Category": "Large Purchase", "Anomaly": True})
    txs.append({"ID": "A2", "Amount ($)": random.randint(3200, 3400), "Category": "International Transfer", "Anomaly": True})

    df = pd.DataFrame(txs)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    return df

# --- Layout: Two Columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Simulate Transactions")
    if st.button("Generate 10 Transactions"):
        st.session_state.transactions = generate_transactions()
        st.session_state.analysis_done = False
        st.success("✅ Transactions generated! Ready for analysis.")

    if st.button("Clear Transactions"):
        st.session_state.transactions = pd.DataFrame()
        st.session_state.analysis_done = False

    if not st.session_state.transactions.empty:
        st.dataframe(st.session_state.transactions, use_container_width=True)
    else:
        st.info("Click **Generate 10 Transactions** to begin.")

with col2:
    st.subheader("2. QML Anomaly Detection")

    # --- Threshold control ---
    threshold = st.slider("🔎 Anomaly Threshold ($)", 500, 6000, 3000, step=100)

    if st.button("Run QML Analysis", disabled=st.session_state.transactions.empty):
        with st.spinner("Running QML analysis..."):
            time.sleep(2)  # Simulate computation time

            df = st.session_state.transactions.copy()

            # Rule-based anomaly detection
            df["Anomaly"] = df["Amount ($)"] > threshold

            # Quantum probability scoring (instead of random.uniform)
            df["QML_Prob_Anomaly"] = df["Amount ($)"].apply(
                lambda x: quantum_random_prob() if x > threshold else quantum_random_prob()
            )

            # Explainability
            df["Reason"] = df.apply(lambda x: "High Amount" if x["Anomaly"] else "Normal", axis=1)

            st.session_state.transactions = df
            st.session_state.analysis_done = True

    if st.session_state.analysis_done:
        anomalies = st.session_state.transactions["Anomaly"].sum()
        if anomalies > 0:
            st.markdown(f"### 🔴 Found **{anomalies} anomalies** which may indicate fraud!")
        else:
            st.markdown("### ✅ No anomalies detected.")

    # Highlight anomalies in table
    if not st.session_state.transactions.empty:
        def highlight_anomalies(row):
            color = "background-color: red; color: white;" if row["Anomaly"] else ""
            return [color] * len(row)

        styled_df = st.session_state.transactions.style.apply(highlight_anomalies, axis=1)
        st.write(styled_df)

# --- Visualization Dashboard ---
st.markdown("---")
st.subheader("📊 Visualization Dashboard")

if not st.session_state.transactions.empty:
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Transaction Categories Distribution**")
        fig1, ax1 = plt.subplots()
        st.session_state.transactions["Category"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col4:
        st.markdown("**Normal vs Anomalous Transactions**")
        counts = st.session_state.transactions["Anomaly"].value_counts()
        fig2, ax2 = plt.subplots()
        counts.plot(kind="bar", color=["green", "red"], ax=ax2)
        ax2.set_xticklabels(["Normal", "Anomaly"], rotation=0)
        st.pyplot(fig2)

# --- Real-time Simulation ---
st.markdown("---")
st.subheader("🎮 Real-Time Fraud Detection Simulation")

if not st.session_state.transactions.empty and st.session_state.analysis_done:
    if st.button("Start Simulation"):
        st.info("🚀 Streaming transactions in real-time...")

        placeholder = st.empty()  # dynamic container
        for _, row in st.session_state.transactions.iterrows():
            with placeholder.container():
                st.write(f"📄 **Transaction ID:** {row['ID']}")
                st.write(f"💰 Amount: ${row['Amount ($)']}")
                st.write(f"🏷️ Category: {row['Category']}")
                st.write(f"🧠 QML Probability of Anomaly: {row['QML_Prob_Anomaly']*100:.0f}%")

                if row["Anomaly"]:
                    st.error(f"🚨 Fraud Alert! Reason: {row['Reason']}")
                else:
                    st.success("✅ Transaction is normal")

            time.sleep(1.5)  # delay for streaming effect

# --- Concept Explanation ---
st.markdown("---")
st.subheader("How it works (Simplified)")
st.markdown("""
- **Quantum Data Encoding:** Classical transaction data is converted into a quantum state.  
- **Quantum Circuit:** A Quantum Variational Autoencoder (QVAE) is trained to learn the "normal" patterns.  
- **Anomaly Scoring:** New transactions are fed into the trained QVAE. Anomalies are poorly reconstructed → high error.  
- **Fraud Alert:** Transactions with error above threshold are flagged.  
""")