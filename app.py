import streamlit as st
import ats_scorer

st.set_page_config(page_title="ATS Score Detector", layout="centered")

st.title("🎯 Smart ATS Score Detector")
st.markdown("Upload your resume and paste the job description to get a semantic matching score powered by Machine Learning.")

# Job Description Input
st.subheader("1. Job Description")
job_description = st.text_area("Paste the job description here:", height=200, 
                               placeholder="e.g. We are looking for a Software Engineer with experience in Python, Machine Learning...")

# Resume Upload
st.subheader("2. Upload Resume")
uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])

# Score Calculation
if st.button("Calculate ATS Score"):
    if not job_description.strip():
        st.warning("Please paste the job description.")
    elif uploaded_file is None:
        st.warning("Please upload a resume (PDF).")
    else:
        with st.spinner("Analyzing resume against job description..."):
            try:
                # Extract text from the uploaded PDF
                resume_text = ats_scorer.extract_text_from_pdf(uploaded_file)
                
                # Calculate score
                score = ats_scorer.calculate_ats_score(resume_text, job_description)
                
                # Display Results
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Use columns to center the metric
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(label="ATS Match Score", value=f"{score}%")
                
                # Provide feedback based on score
                if score >= 75:
                    st.success("Excellent Match! Your resume aligns very well with the job description.")
                elif score >= 50:
                    st.info("Good Match. Your resume has decent alignment, but you may want to highlight specific keywords from the JD.")
                else:
                    st.error("Low Match. Consider tailoring your resume more closely to the skills and experiences mentioned in the job description.")
                    
            except Exception as e:
                st.error(f"An error occurred while processing: {e}")

st.markdown("---")
st.caption("Powered by Sentence-Transformers and Streamlit")
