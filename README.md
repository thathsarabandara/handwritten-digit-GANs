# Handwritten Digit Generator (cGAN)

Generate synthetic handwritten digits using a Conditional Generative Adversarial Network trained on MNIST.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
# Open and run: notebooks/gan_training.ipynb

# Launch the app
streamlit run app/app.py
```

## Features

- **Conditional digit generation**: Generate specific digits (0-9)
- **Random generation**: Mix of all digit classes
- **Interactive UI**: Simple controls for sampling and visualization

## Project Structure

```
├── app/                    # Streamlit application
│   ├── app.py             # Main UI
│   └── models/            # GAN architecture
├── notebooks/             # Training notebook
│   └── gan_training.ipynb
├── models/                # Saved model weights
└── requirements.txt
```

