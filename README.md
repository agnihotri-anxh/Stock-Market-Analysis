# Zidio_Work

## Overview
This repository contains two main projects:

1. **Stock Analysis & Time Series Forecasting**
2. **Image Captioning & Segmentation**

---

## 1. Stock Analysis & Time Series Forecasting
A comprehensive dashboard and toolkit for analyzing stock data and forecasting prices using multiple machine learning models.

### Features
- Real-time data fetching from Yahoo Finance
- Interactive visualizations using Plotly
- 10+ prediction models: Prophet, LSTM, ARIMA, SARIMA, XGBoost, LightGBM, CatBoost, Random Forest, SVR, and more
- Performance metrics: MSE, MAE, R² Score
- Data features: Closing Price, Returns, Moving Averages, Volatility

### Setup
```bash
cd Stock_Analysis
pip install -r requirements.txt
```

### Usage
```bash
python stock_analysis.py
```
Or, if using Streamlit:
```bash
streamlit run app.py
```

---

## 2. Image Captioning & Segmentation
A deep learning-based system for generating captions for images and performing image segmentation using OpenCV and TensorFlow.

### Features
- Image caption generation using deep learning
- Image segmentation using OpenCV (basic) and optional DeepLabV3+ (TensorFlow Hub)
- K-means color segmentation
- Pre-trained models and feature extractors
- Utilities for processing COCO-style datasets

### Setup
```bash
cd image_caption
pip install -r requirements.txt
```

### Usage
- Run the main app or use the provided notebooks for feature extraction and model training.
- Example (Python):
```python
from image_segmentation import opencv_basic_colored_segmentation
# ...
```

---

## Directory Structure
```
Zidio_Work/
│
├── Stock_Analysis/
│   ├── stock_analysis.py
│   ├── requirements.txt
│   └── README.md
│
├── image_caption/
│   ├── app.py
│   ├── image_segmentation.py
│   ├── object_detection.py
│   ├── feature_extractor.ipynb
│   ├── model_trainer.ipynb
│   ├── requirements.txt
│   ├── image_caption_sub/
│   │   └── output/
│   │       ├── model.h5
│   │       ├── sample_caption.png
│   │       └── tokenizer.pkl
│   └── ...
│
├── dataset/           # (Ignored in git, contains images and annotations)
└── README.md
```

---

## Notes
- The `dataset/` directory is excluded from version control due to its large size.
- Some large model and data files are also excluded from git.
- For large file support, consider using [Git LFS](https://git-lfs.github.com/).

---

## License
This repository is for educational and research purposes. Please check individual files for specific licenses or requirements. 