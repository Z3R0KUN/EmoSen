import streamlit as st
import pandas as pd
import cv2
from PIL import Image
import tempfile
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from analyzer import analyze_text
from face_emotion import analyze_face
from realtimedetection import extract_features, model, face_cascade, labels

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="EmoSenAI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CSS & ANIMATIONS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%);
    color: white;
}

.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    text-align: center;
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    text-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.subtitle {
    text-align: center;
    color: #a0a0c0;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

/* Card styling with animations */
.glass-card {
    background: rgba(28, 31, 38, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 30px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    margin-bottom: 20px;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    border-color: rgba(102, 126, 234, 0.5);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 30px;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
    margin: 10px 0;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
}

/* Mode selection cards */
.mode-card {
    background: rgba(28, 31, 38, 0.8);
    border-radius: 15px;
    padding: 25px;
    border: 2px solid transparent;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: center;
}

.mode-card:hover {
    border-color: #667eea;
    transform: scale(1.02);
    background: rgba(102, 126, 234, 0.1);
}

.mode-card.selected {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.2);
}

/* Metrics styling */
.metric-card {
    background: rgba(28, 31, 38, 0.6);
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    border-left: 4px solid #667eea;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateX(5px);
    border-left-width: 6px;
}

/* Progress bars */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Camera feed styling */
.camera-container {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    border: 3px solid rgba(102, 126, 234, 0.4);
    margin: 20px 0;
}

.camera-feed {
    width: 100%;
    display: block;
}

.camera-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.7);
    padding: 15px;
    color: white;
    font-size: 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

/* Emotion badge styling */
.emotion-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    margin: 5px;
    font-weight: 600;
    font-size: 0.9rem;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header {
        font-size: 2.5rem;
    }
    
    .glass-card {
        padding: 20px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "started" not in st.session_state:
    st.session_state.started = False

if "mode" not in st.session_state:
    st.session_state.mode = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "emotion_counter" not in st.session_state:
    st.session_state.emotion_counter = {}

if "last_emotions" not in st.session_state:
    st.session_state.last_emotions = []

if "camera_placeholder" not in st.session_state:
    st.session_state.camera_placeholder = None

# ---------------- HELPER FUNCTIONS ----------------
def create_emotion_gauge(emotion, score):
    """Create a gauge chart for emotion confidence"""
    colors = {
        'angry': '#FF4757',
        'disgust': '#2ed573',
        'fear': '#ffa502',
        'happy': '#ffa502',
        'sad': '#5352ed',
        'surprise': '#ff7f50',
        'neutral': '#747d8c',
        '0': '#FF4757',    # angry
        '1': '#2ed573',    # disgust
        '2': '#ffa502',    # fear
        '3': '#ffa502',    # happy
        '4': '#5352ed',    # sad
        '5': '#ff7f50',    # surprise
        '6': '#747d8c'     # neutral
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': str(emotion).upper(), 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': colors.get(str(emotion).lower(), "#667eea")},
            'steps': [
                {'range': [0, 33], 'color': 'rgba(255, 0, 0, 0.1)'},
                {'range': [33, 66], 'color': 'rgba(255, 255, 0, 0.1)'},
                {'range': [66, 100], 'color': 'rgba(0, 255, 0, 0.1)'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Poppins"},
        height=250,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# ---------------- SIMPLE LIVE CAMERA ----------------
def run_live_camera():
    """Simple live camera function that just works"""
    try:
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("⚠️ Camera not found or not accessible")
            return
        
        # Get labels - handle both string and integer labels
        if labels:
            if isinstance(labels[0], (int, np.integer)):
                emotion_names = {i: str(label) for i, label in enumerate(labels)}
            else:
                emotion_names = {i: label for i, label in enumerate(labels)}
        else:
            # Default emotions if labels not provided
            emotion_names = {
                0: 'ANGRY', 1: 'DISGUST', 2: 'FEAR', 
                3: 'HAPPY', 4: 'SAD', 5: 'SURPRISE', 6: 'NEUTRAL'
            }
        
        # Create placeholders
        camera_placeholder = st.empty()
        
        # Initialize counters if not exists
        if "emotion_counter" not in st.session_state:
            st.session_state.emotion_counter = {name: 0 for name in emotion_names.values()}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read frame from camera")
                break
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            current_emotions = []
            
            for (x, y, w, h) in faces:
                # Extract and process face ROI
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = extract_features(roi)
                
                # Predict emotion
                try:
                    pred = model.predict(roi)
                    emotion_idx = pred.argmax()
                    
                    # Get emotion name
                    emotion_name = emotion_names.get(emotion_idx, f"EMOTION {emotion_idx}")
                    
                    # Update counter
                    if emotion_name in st.session_state.emotion_counter:
                        st.session_state.emotion_counter[emotion_name] += 1
                    else:
                        st.session_state.emotion_counter[emotion_name] = 1
                    
                    current_emotions.append(emotion_name)
                    
                    # Color mapping
                    color_map = {
                        'ANGRY': (255, 50, 50),
                        'DISGUST': (50, 200, 50),
                        'FEAR': (200, 50, 200),
                        'HAPPY': (50, 255, 50),
                        'SAD': (50, 50, 255),
                        'SURPRISE': (255, 200, 50),
                        'NEUTRAL': (150, 150, 150)
                    }
                    
                    color = color_map.get(emotion_name, (102, 126, 234))
                    
                    # Draw rectangle around face - make it thicker
                    cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), color, 3)
                    
                    # ============ FIXED LABEL RENDERING ============
                    # Create label background (make sure it's inside frame)
                    label_height = 40
                    label_top = max(0, y - label_height)  # Don't go above frame
                    
                    # Draw solid background for label
                    cv2.rectangle(rgb_frame, (x, label_top), (x+w, y), color, -1)
                    
                    # Set font properties
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    thickness = 2
                    
                    # Calculate text size and position
                    (text_width, text_height), baseline = cv2.getTextSize(
                        emotion_name, font, font_scale, thickness
                    )
                    
                    # Center text in label box
                    text_x = x + (w - text_width) // 2
                    text_y = y - (label_height - text_height) // 2 - 5
                    
                    # Draw text with black outline for visibility
                    cv2.putText(rgb_frame, emotion_name, (text_x, text_y),
                               font, font_scale, (0, 0, 0), thickness + 1)  # Black outline
                    cv2.putText(rgb_frame, emotion_name, (text_x, text_y),
                               font, font_scale, (255, 255, 255), thickness)  # White text
                    # ================================================
                    
                except Exception as e:
                    continue
            
            # Display camera feed
            camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Small delay
            time.sleep(0.03)
            
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
    finally:
        try:
            cap.release()
        except:
            pass

# ---------------- MAIN APP ----------------
st.markdown('<h1 class="main-header">🎭 EmoSenAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Emotion & Sentiment Evaluation Platform</p>', unsafe_allow_html=True)

# ---------------- STEP 1: START ----------------
if not st.session_state.started:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h2 style="color: #ffffff; margin-bottom: 20px;">Welcome to EmoSenAI</h2>
            <p style="color: #a0a0c0; margin-bottom: 30px;">
                A powerful platform for emotion and sentiment analysis across multiple modalities.
                Choose your analysis type and gain insights into emotional states.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Evaluation", key="start_btn", use_container_width=True):
            st.session_state.started = True
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- STEP 2: MODE SELECTION ----------------
elif st.session_state.mode is None:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    
    st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">Choose Analysis Type</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Text Analysis", key="text_btn", use_container_width=True):
            st.session_state.mode = "Text"
            st.rerun()
    
    with col2:
        if st.button("🖼️ Image Analysis", key="image_btn", use_container_width=True):
            st.session_state.mode = "Image"
            st.rerun()
    
    with col3:
        if st.button("🎥 Live Camera", key="live_btn", use_container_width=True):
            st.session_state.mode = "Live Camera"
            st.session_state.camera_active = True
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- STEP 3: ANALYSIS ----------------
else:
    # Back button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("↩ Back", use_container_width=True):
            st.session_state.mode = None
            st.session_state.camera_active = False
            st.rerun()
    
    # -------- TEXT ANALYSIS --------
    if st.session_state.mode == "Text":
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white; margin-bottom: 20px;">📝 Text Emotion & Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        text = st.text_area("Enter your text here:", height=150, 
                          placeholder="Type or paste your text to analyze emotions and sentiment...")
        
        if st.button("🔍 Analyze Text", use_container_width=True, type="primary"):
            if text.strip():
                with st.spinner("Analyzing text..."):
                    result = analyze_text(text)
                    st.session_state.analysis_results = result
            
                # Display results
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"### Sentiment: **{result['sentiment']['label'].upper()}**")
                    sentiment_scores = result["sentiment"]["all_scores"]
                    sent_df = pd.DataFrame(
                        sentiment_scores.items(),
                        columns=["Sentiment", "Score"]
                    )
                    st.dataframe(sent_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"### Emotion: **{result['emotion']['label'].upper()}**")
                    emotion_scores = result["emotion"]["all_scores"]
                    emo_df = pd.DataFrame(
                        emotion_scores.items(),
                        columns=["Emotion", "Score"]
                    )
                    st.dataframe(emo_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Gauge chart
                st.markdown("### 📊 Confidence Gauge")
                fig = create_emotion_gauge(
                    result['emotion']['label'],
                    result['emotion']['all_scores'].get(result['emotion']['label'], 0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ Please enter some text to analyze.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # -------- IMAGE ANALYSIS --------
    elif st.session_state.mode == "Image":
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white; margin-bottom: 20px;">🖼️ Image Emotion Detection</h2>', unsafe_allow_html=True)
        
        img_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        
        if img_file is not None:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
                with st.spinner("Detecting emotions..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        img.save(tmp.name)
                        dominant, scores = analyze_face(tmp.name)
                
                # Display results
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"### Dominant Emotion")
                    st.markdown(f'<h1 style="color: #667eea; text-align: center; font-size: 3rem; margin: 20px 0;">{dominant.upper()}</h1>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_b:
                    # Bar chart
                    fig = px.bar(
                        x=list(scores.keys()),
                        y=list(scores.values()),
                        labels={'x': 'Emotion', 'y': 'Confidence'},
                        color=list(scores.values()),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Please upload an image to begin analysis")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # -------- LIVE CAMERA ANALYSIS --------
    elif st.session_state.mode == "Live Camera":
        # Clean, minimal interface
        st.markdown('<h2 style="color: white; margin-bottom: 20px;">🎥 Live Camera Emotion Detection</h2>', unsafe_allow_html=True)
        
        # Simple instructions
        st.info("""
        **How to use:** 
        - Camera starts automatically
        - Position your face in the camera view
        - See emotion labels appear on detected faces
        - Multiple faces can be detected at once
        """)
        
        # Start camera automatically
        if st.session_state.camera_active:
            # Just show the camera feed - nothing else
            run_live_camera()
        else:
            st.info("🎥 Starting camera...")
            st.session_state.camera_active = True
            st.rerun()

# ---------------- FOOTER ----------------
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #a0a0c0; border-top: 1px solid rgba(255,255,255,0.1)">
    <p>EmoSenAI v1.0 | Advanced Emotion Detection Platform</p>
    <p style="font-size: 0.9rem;">Powered by Deep Learning & Computer Vision</p>
</div>
""", unsafe_allow_html=True)