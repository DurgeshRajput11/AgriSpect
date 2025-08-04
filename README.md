# 🌾 AgriVision: AI-Powered Crop Analysis
Live WebApp : (https://huggingface.co/spaces/DurgeshRajput11/AgriVision)

AgriVision is an interactive web app for smart crop and fruit detection using YOLOv8. Upload images, videos, or use your webcam to identify and count fruits and crops with a beautiful, modern UI.
![Plant Model Result](https://github.com/DurgeshRajput11/AgriVision/blob/a58da8f3b69fd531f47e4cab5c57c156be69f377/agri_analysis_IMG-20250801-WA0007%20(2).jpg)
![Live WebCam Results](https://github.com/DurgeshRajput11/AgriVision/blob/c88ffb0b3e1b9f0261b280195183ea2875a2545b/Screenshot%202025-07-01%20085748.png)
## 🚀 Features
 
- **Model selection:** Choose from system-trained, YOLOv8 variants, or upload your own model.
- **Flexible input:** Supports images, videos, and live webcam.
- **Customizable detection:** Adjust confidence and IoU thresholds.
- **Modern UI:** Dark mode, accent colors, and responsive design.
- **Download results:** Save processed images/videos with detections.


## 🛠️ Installation

1. **Clone the repository:**

    ```
    git clone https://github.com/your-username/agrivision-app.git
    cd agrivision-app
    ```

2. **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

3. **(Optional) Set up your own YOLOv8 model:**

    - Place your `.pt` file in the `weights/` folder.

## ▶️ Usage


- Open your browser at the displayed local URL.
- Use the sidebar to configure the model, detection parameters, and input type.

## 💡 Customization

- **Change theme/colors:** Edit the CSS in `app.py` for different sidebar or main area looks.
- **Add new models:** Place new YOLOv8 `.pt` files in the `weights/` directory.





## 📁 Project Structure




agrivision-app/   
│
├── app.py     
├── requirements.txt         
├── weights/    
│   └── best.pt    
├── temp/          
├── .gitignore     
└── README.md    





## ->
 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
