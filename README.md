# Face Recognition Analysis Comparison

This project is an advanced face recognition analysis tool designed to compare different recognition techniques, specifically focused on performance with low-quality and blurred images. It includes initial image quality analysis, an Eigenface-based recognition method, and future support for other techniques.

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Features

- **Initial Image Quality Analysis**
  - Assesses quality metrics like blur, brightness, and contrast for input images.

- **Recognition Techniques**
  - Eigenface Analysis for face recognition.
  - *(Planned support for Local Binary Pattern (LBP) Analysis and Deep Learning-based techniques)*

- **Interactive Visualization**
  - Real-time quality analysis with visual charts and quality metrics.

## 🛠️ Installation

### Prerequisites

Ensure you have the following installed:

- **Python** 3.8 or higher
- **Git**
- **pip** (Python package installer)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/dedipya001/Face-Recognition-Analysis-Comparison.git
   cd Face-Recognition-Analysis-Comparison
Create and Activate a Virtual Environment

Windows:

```bash

python -m venv venv
venv\Scripts\activate
```
Linux/Mac:

```bash

python3 -m venv venv
source venv/bin/activate
```

Install Dependencies

```bash

pip install streamlit opencv-python numpy Pillow scikit-learn matplotlib seaborn
Run the Application
```

Run the Application
```bash

streamlit run app.py
```
