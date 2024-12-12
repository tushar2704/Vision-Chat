# VisionChat AI Image Analysis 🖼️🤖

## Overview
VisionChat is an intelligent Streamlit application that leverages advanced AI models for comprehensive image analysis. The application provides multiple modules for processing and understanding visual content using cutting-edge vision and language models.

## Features 🌟
- **Image Analysis Module**: Upload and analyze images with detailed AI insights
- **Vision LLM Module**: Perform advanced image classification and interpretation
- **Camera Capture Module**: Real-time image capture and analysis

## Prerequisites 📋
- Python 3.8+
- Groq API Key

## Installation 🛠️

### 1. Clone the Repository
```bash
git clone https://github.com/tushar2704/visionchat.git
cd visionchat-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file or use Streamlit secrets to store your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage 🚀

### Run the Application
```bash
streamlit run app.py
```

### Modules
1. **Image Analysis**: Upload and analyze images
2. **Vision LLM**: Advanced image classification
3. **Capture Analysis**: Camera-based image capture and analysis

## Technologies Used 💻
- Streamlit
- Groq AI
- Plotly
- PIL (Python Imaging Library)
- LangChain

## Configuration 🔧
- Supports PNG, JPG, JPEG image formats
- Configurable AI analysis parameters

## Contributing 🤝
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄
[Specify your project's license here]

## Contact 📧
[Tushar Aggarwal](https://www.linkedin.com/in/tusharaggarwalinseec/)

## Acknowledgments 🙏
- Groq for providing advanced AI models
- Streamlit for the amazing web application framework