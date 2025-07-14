import os
import shutil
import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import streamlit as st

# --- Environment and UI Setup ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics_config'
os.makedirs(os.environ['STREAMLIT_CONFIG_DIR'], exist_ok=True)
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)
os.environ['STREAMLIT_GATHER_USAGE_STATS'] = "false"

# --- Custom CSS for Dark Theme ---
custom_css = """
<style>
body, .stApp { background: #23243a !important; color: #f4f4f4 !important; }
.stSidebar { background: #2c2f3a !important; color: #f4f4f4 !important; border-radius: 16px; padding: 1.5rem 1rem; }
.stSidebarContent { background: #2c2f3a !important; border-radius: 16px; }
.stExpander { background: #26283a !important; border-radius: 12px; margin-bottom: 1.2rem; border: 1px solid #33354a; box-shadow: 0 2px 8px rgba(44,47,58,0.10); }
.stExpanderHeader { color: #f4f4f4 !important; font-weight: 700 !important; font-size: 1.1rem !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p, .stMarkdown span, .stMarkdown div { color: #f4f4f4 !important; }
.stSlider > div[data-baseweb="slider"] { background: transparent !important; }
.stSlider .rc-slider-track { background-color: #ff4b4b !important; }
.stSlider .rc-slider-handle { border: solid 2px #ff4b4b !important; background: #fff !important; }
.stSlider .rc-slider-dot-active { border-color: #ff4b4b !important; }
.stRadio label { color: #f4f4f4 !important; }
.stRadio [data-baseweb="radio"] > div[role="radio"]:checked { border-color: #ff4b4b !important; background-color: #ff4b4b !important; }
.stSelectbox, .stSelectbox > div { background: #181926 !important; color: #f4f4f4 !important; border-radius: 8px !important; }
.stSelectbox label { color: #f4f4f4 !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.set_page_config(
    page_title="🌾 AgriVision - Smart Crop Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "AI-Powered Crop Monitoring System"}
)

# --- Sidebar: Model and Detection Settings ---
with st.sidebar:
    st.title("⚙️ Model Configuration")
    with st.expander("Model Settings", expanded=True):
        model_options = [
            "Fruits Counting Model",
            "Plants Counting Model",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "Custom"
        ]
        model_type = st.selectbox("Select Model", model_options, index=0)
        # Model loading logic will be handled later
    with st.expander("Detection Parameters", expanded=True):
        confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)
        iou = st.slider("IoU threshold", 0.1, 1.0, 0.5, 0.05)
    with st.expander("Input Options"):
        input_type = st.radio("Input type", ["Image", "Video", "Webcam"])
    st.markdown("---")
    st.caption("Developed Under IIITDMJ | v1.2.0")

st.title("🌾 AgriVision: AI-Powered Crop Analysis")
st.markdown("Upload agricultural images, videos, or use your webcam to identify and count fruits or plants using YOLOv8. Fast, accurate, and easy to use!")

# --- File Upload Section ---
upload_container = st.container()
if input_type == "Image":
    uploaded_file = upload_container.file_uploader("📷 Upload agricultural image", type=["jpg", "jpeg", "png"])
elif input_type == "Video":
    uploaded_file = upload_container.file_uploader("🎬 Upload field video", type=["mp4", "mov", "avi"])
else:
    uploaded_file = upload_container.camera_input("📸 Capture field image")

results_container = st.container()
model = None

# --- Model Loading Logic (including safe_globals for custom) ---
if model_type == "Fruits Counting Model":
    model_path = "weights/best.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
elif model_type == "Plants Counting Model":
    model_path = "weights/best (2).pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
elif model_type == "Custom":
    custom_model = st.file_uploader("Upload your model", type=["pt"])
    if custom_model:
        os.makedirs("weights", exist_ok=True)
        with open("weights/custom.pt", "wb") as f:
            f.write(custom_model.getbuffer())
        try:
            with torch.serialization.safe_globals([DetectionModel]):
                model = torch.load("weights/custom.pt", weights_only=True)
            st.toast("Custom model loaded using safe_globals.", icon="✅")
        except Exception as e:
            st.error(f"Failed to load custom model: {e}")
            st.stop()
    else:
        model_path = "weights/yolov8s.pt"
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
else:
    model_path = f"weights/{model_type}"
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# --- Main Detection Logic ---
if uploaded_file is not None and model is not None:
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with results_container:
        st.subheader("🔍 Input Preview")
        if input_type == "Image":
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original", use_container_width=True)
        elif input_type == "Video":
            st.video(uploaded_file)
        else:
            st.image(uploaded_file, caption="Captured Image", use_container_width=True)

    with results_container:
        st.subheader("📊 Detection Results")
        progress_bar = st.progress(0, text="Analyzing content...")

        if input_type == "Image" or input_type == "Webcam":
            try:
                if model_type == "Custom" and isinstance(model, torch.nn.Module):
                    # For custom model loaded with torch.load()
                    st.warning("Direct inference with custom PyTorch model requires manual preprocessing and postprocessing.")
                else:
                    results = model.predict(
                        source=file_path,
                        conf=confidence,
                        iou=iou,
                        save=True,
                        save_txt=True,
                        project="temp",
                        exist_ok=True
                    )
                    progress_bar.progress(100, "Analysis complete!")
                    detected_image_path = os.path.join("temp", "predict", uploaded_file.name)
                    if os.path.exists(detected_image_path):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(uploaded_file, caption="Original", use_container_width=True)
                        with col2:
                            st.image(detected_image_path, caption="Detected Objects", use_container_width=True)
                        with st.expander("🔬 Detailed Analysis", expanded=True):
                            if results and results[0].boxes:
                                class_counts = {}
                                for box in results[0].boxes:
                                    class_id = int(box.cls)
                                    class_name = model.names[class_id]
                                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                                st.subheader("📈 Object Distribution")
                                for cls, count in class_counts.items():
                                    st.metric(label=cls, value=count)
                                st.subheader("🌐 Field Insights")
                                st.markdown(f"- **Total detections:** {sum(class_counts.values())}")
                                st.markdown(f"- **Dominant species:** {max(class_counts, key=class_counts.get)}")
                            else:
                                st.warning("No agricultural objects detected in the image. Try lowering the confidence threshold.")
                        with open(detected_image_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=f,
                                file_name=f"agri_analysis_{uploaded_file.name}",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                    else:
                        st.warning("Detection completed, but the output image was not saved. Please check file permissions.")
            except Exception as e:
                st.error(f"Detection failed: {e}")

        else:
            try:
                unique_objects = {}
                cap = cv2.VideoCapture(file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                results = model.track(
                    source=file_path,
                    conf=confidence,
                    iou=iou,
                    save=True,
                    project="temp",
                    exist_ok=True,
                    stream=True,
                    tracker="botsort.yaml"
                )
                processed_frames = 0
                start_time = time.time()
                for result in results:
                    processed_frames += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_frame = elapsed_time / processed_frames
                    remaining_frames = total_frames - processed_frames
                    remaining_time = avg_time_per_frame * remaining_frames
                    minutes = int(remaining_time // 60)
                    seconds = int(remaining_time % 60)
                    progress = int(100 * processed_frames / total_frames)
                    progress_bar.progress(progress, f"Processing frames... {progress}% | Time left: {minutes}m {seconds}s")
                    if result.boxes and result.boxes.id is not None:
                        for box, box_id in zip(result.boxes, result.boxes.id):
                            class_id = int(box.cls)
                            class_name = model.names[class_id]
                            if class_name not in unique_objects:
                                unique_objects[class_name] = set()
                            unique_objects[class_name].add(int(box_id))
                progress_bar.progress(100, "Analysis complete!")
                base_name = os.path.splitext(uploaded_file.name)[0]
                output_video_track = os.path.join("temp", "track", base_name + '.avi')
                output_video_predict = os.path.join("temp", "predict", base_name + '.avi')
                if os.path.exists(output_video_track):
                    output_video = output_video_track
                elif os.path.exists(output_video_predict):
                    output_video = output_video_predict
                else:
                    output_video = None
                if output_video:
                    st.subheader("🎬 Processed Video")
                    st.video(output_video)
                    if unique_objects:
                        st.subheader("📈 Unique Objects Detected (Video)")
                        for cls, ids in unique_objects.items():
                            st.metric(label=cls, value=len(ids))
                        st.subheader("🌐 Field Insights (Video)")
                        st.markdown(f"- **Total unique objects:** {sum(len(ids) for ids in unique_objects.values())}")
                        st.markdown(f"- **Dominant species:** {max(unique_objects, key=lambda k: len(unique_objects[k]))}")
                    with open(output_video, "rb") as f:
                        st.download_button(
                            label="📥 Download Analyzed Video",
                            data=f,
                            file_name=f"agri_analysis_{base_name}.avi",
                            mime="video/avi",
                            use_container_width=True
                        )
                else:
                    st.error("Video detection completed, but the output video was not saved. Please check file permissions.")
            except Exception as e:
                st.error(f"Video detection failed: {e}")

    shutil.rmtree(os.path.join("temp", "track"), ignore_errors=True)
    shutil.rmtree(os.path.join("temp", "predict"), ignore_errors=True)
else:
    with results_container:
        st.info(" Upload an image or video to begin analysis")

st.markdown("---")
st.caption("🌐 AI-powered agricultural analysis | Identify Fruits And Plants And Count Them | Also Accepts Users Own Model For Testing")
