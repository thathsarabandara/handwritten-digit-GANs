# GAN Handwritten Digits Project

This repository contains a Generative Adversarial Network (GAN) implementation for generating handwritten digit images using the MNIST dataset, along with a Streamlit application for digit recognition.

## Project Structure
```
gan_handwritten_digits/
├── data/                # (optional) place for cached MNIST data
├── models/              # GAN and optional classifier
│   ├── __init__.py
│   ├── gan.py           # Generator & Discriminator
│   └── classifier.py    # Simple CNN classifier (optional)
├── notebooks/           # Jupyter notebook for training & visualization
│   └── gan_training.ipynb
├── app/                 # Streamlit UI
│   ├── __init__.py
│   ├── app.py
│   └── utils.py
├── utils/               # Shared utilities (e.g., seed setting)
│   └── __init__.py
├── requirements.txt     # Python dependencies
└── README.md            # Project overview & setup
```

## Quick Start
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the GAN** (run the notebook `notebooks/gan_training.ipynb`).
3. **Run the Streamlit app**
   ```bash
   streamlit run app/app.py
   ```

