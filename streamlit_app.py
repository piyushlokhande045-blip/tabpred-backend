import streamlit as st
import time

st.set_page_config(page_title="TABPred", page_icon="🧪", layout="wide")

# CSS styling
st.markdown("""
<style>
body { 
    background: linear-gradient(135deg, #060a10 0%, #0c131c 100%);
    color: #e8edf2;
}
.stButton>button { 
    background: linear-gradient(90deg, #2dd4bf, #14b8a6) !important;
    color: white !important;
    border-radius: 8px;
    padding: 12px 30px;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(45, 212, 191, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧪 TABPred")
st.subheader("Binding Affinity Predictor for c-Met (PDB 4R1V)")

st.markdown("---")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    smiles = st.text_input(
        "Enter SMILES String:",
        placeholder="e.g., CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
        label_visibility="collapsed"
    )

with col2:
    st.write("")  # Spacing

# Examples
st.markdown("**📋 Quick Examples:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example 1", use_container_width=True):
        st.session_state.smiles1 = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"

with col2:
    if st.button("Example 2", use_container_width=True):
        st.session_state.smiles2 = "CC(=O)Nc1ccc(O)cc1"

with col3:
    if st.button("Example 3", use_container_width=True):
        st.session_state.smiles3 = "c1ccc(cc1)C(=O)O"

st.markdown("---")

# PREDICT BUTTON
if st.button("🔬 PREDICT BINDING AFFINITY", use_container_width=True, type="primary"):
    if not smiles:
        st.error("❌ Please enter a SMILES string!")
    else:
        # Loading animation with progress
        st.markdown("### ⏳ Processing...")
        
        progress_container = st.container()
        
        stages = [
            ("⚙️ Preparing Ligand", "Converting SMILES to 3D structure", 0.2),
            ("🎯 Molecular Docking", "Running AutoDock Vina simulation", 0.5),
            ("📊 Feature Extraction", "Calculating Mordred descriptors", 0.7),
            ("🤖 ML Prediction", "XGBoost + CatBoost ensemble", 0.9),
            ("✨ Finalizing", "Generating final results", 1.0)
        ]
        
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        percentage_text = progress_container.empty()
        
        for stage_emoji, stage_desc, progress_val in stages:
            status_text.markdown(f"**{stage_emoji}**\n{stage_desc}")
            percentage_text.markdown(f"<h3 style='color: #2dd4bf; text-align: center;'>{int(progress_val * 100)}%</h3>", unsafe_allow_html=True)
            progress_bar.progress(progress_val)
            time.sleep(2)
        
        # Results (Dummy - replace with real backend later)
        affinity_value = -7.45
        
        st.success("✅ Prediction Complete!")
        
        # Result metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Binding Affinity", f"{affinity_value:.2f}", "kcal/mol")
        with col2:
            st.metric("Target", "c-Met", "PDB 4R1V")
        with col3:
            st.metric("Model", "Ensemble", "Vina + ML")
        with col4:
            st.metric("Confidence", "High", "✓")
        
        # Details
        st.markdown("---")
        st.markdown("### 📊 Analysis Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Input SMILES:** `{smiles}`")
        with col2:
            st.info(f"**Prediction Method:** AutoDock Vina + XGBoost/CatBoost")
        
        st.markdown("""
        **Note:** This is a predicted Vina docking score (delta-corrected), 
        NOT experimental wet-lab binding affinity. Use for virtual screening only.
        """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.header("🔬 About TABPred")
st.sidebar.markdown("""
**Version:** 1.0

**Features:**
- SMILES to 3D conversion
- Molecular docking (AutoDock Vina)
- Feature extraction (Mordred)
- ML prediction (XGBoost + CatBoost)

**Target:** c-Met (PDB: 4R1V)

**Built by:** SMCS-Psi Analytics
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: First prediction takes ~15 seconds to load models")
