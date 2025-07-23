# 🌾 AgriVision: AI-Powered Crop Analysis
Live WebApp : (https://huggingface.co/spaces/DurgeshRajput11/AgriVision)

AgriVision is an interactive web app for smart crop and fruit detection using YOLOv8. Upload images, videos, or use your webcam to identify and count fruits and crops with a beautiful, modern UI.

![Live WebCam Results ]([[https://github.com/user-attachments/assets/1be3f3b0-a91a-4387-a9b9-c53f1cff5ce0]](https://github.com/DurgeshRajput11/AgriVision/blob/main/Screenshot%202025-07-01%20085748.png))
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
