# app/app.py
"""Streamlit application for handwritten digit generation using conditional GANs.
Users can select a digit class (0-9) and generate synthetic examples of that digit.
"""

import streamlit as st
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from seed import set_seed
from models.gan import Generator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
set_seed(SEED)

# Paths to model checkpoint
GENERATOR_PATH = Path(__file__).parent.parent / "models" / "generator.pth"

# Load generator (if available)
if GENERATOR_PATH.is_file():
    generator = Generator(latent_dim=100, num_classes=10).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()
else:
    generator = None

# ---------------------------------------------------------------------------
# Streamlit UI Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Generator (cGAN)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    ### Conditional GAN for Handwritten Digits
    
    This application demonstrates a **Conditional Generative Adversarial Network (cGAN)** trained on MNIST.
    
    **Features:**
    - Generate synthetic handwritten digits for any class (0-9)
    - Control the number of samples to generate
    - Explore digit variability across different classes
    
    **Model Details:**
    - Architecture: Conditional GAN (Generator + Discriminator)
    - Dataset: MNIST (60,000 training images)
    - Classes: 10 digits (0-9)
    - Status: ✅ Loaded if `models/generator.pth` exists
    
    **Technologies:**
    - PyTorch for deep learning
    - Streamlit for interactive UI
    """)
    
    st.divider()
    st.markdown("""
    **How to use:**
    1. Select a digit class (0-9)
    2. Choose how many samples to generate
    3. Click "Generate Digits" to see the results
    4. Each generation is unique and random!
    """)

# ---------------------------------------------------------------------------
# Main Title and Description
# ---------------------------------------------------------------------------
st.title("🎨 Handwritten Digit Generator")

st.markdown("""
This application uses a **Conditional GAN (cGAN)** to generate synthetic handwritten digits.
Simply select a digit class and the model will create realistic examples of that digit.
""")

st.divider()

# ---------------------------------------------------------------------------
# Main Feature: Conditional Digit Generation
# ---------------------------------------------------------------------------
st.header("📊 Generate Digits by Class")

if generator is not None:
    # Two columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Controls")
        selected_digit = st.selectbox(
            "Select a digit to generate:",
            options=list(range(10)),
            format_func=lambda x: f"Digit {x}",
            key="digit_selector"
        )
        
        num_samples = st.select_slider(
            "Number of samples to generate:",
            options=[4, 9, 16, 25, 36],
            value=16,
            key="num_samples"
        )
    
    with col2:
        st.subheader("Generation Settings")
        if st.button("🔄 Generate Digits", use_container_width=True, key="generate_btn"):
            st.session_state.generate = True
        
        if st.button("🗑️ Clear Results", use_container_width=True, key="clear_btn"):
            if "generated_images" in st.session_state:
                del st.session_state.generated_images
            st.rerun()
    
    # Generate samples
    if "generate" in st.session_state and st.session_state.generate:
        with st.spinner(f"🎲 Generating {num_samples} samples of digit {selected_digit}..."):
            with torch.no_grad():
                # Generate random noise
                z = torch.randn(num_samples, 100, device=DEVICE)
                
                # Create labels for the selected digit
                labels = torch.full((num_samples,), selected_digit, dtype=torch.long, device=DEVICE)
                
                # Generate images
                fake_images = generator(z, labels)
                
                # Denormalize from [-1, 1] to [0, 1]
                fake_images_display = (fake_images + 1) / 2
                
                # Store in session state for display
                st.session_state.generated_images = fake_images_display.cpu().numpy()
        
        st.success(f"✅ Generated {num_samples} samples of digit {selected_digit}!")
        st.session_state.generate = False
    
    # Display generated images
    if "generated_images" in st.session_state:
        st.subheader(f"Generated Samples of Digit {selected_digit}")
        
        images = st.session_state.generated_images
        
        # Determine grid layout
        if num_samples == 4:
            cols_per_row = 2
        elif num_samples == 9:
            cols_per_row = 3
        elif num_samples == 16:
            cols_per_row = 4
        elif num_samples == 25:
            cols_per_row = 5
        else:  # 36
            cols_per_row = 6
        
        # Create grid
        for row_idx in range(0, num_samples, cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, col in enumerate(cols):
                if row_idx + col_idx < num_samples:
                    with col:
                        img = images[row_idx + col_idx, 0]  # Get first channel
                        st.image(img, use_column_width=True)
        
        # Show matplotlib visualization
        st.subheader("Grid Visualization")
        
        fig, axes = plt.subplots(
            int(np.sqrt(num_samples)),
            int(np.sqrt(num_samples)) if num_samples != 36 else 6,
            figsize=(10, 10)
        )
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = images[i, 0]
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        
        plt.suptitle(f'Generated Samples - Digit {selected_digit}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.error(
        """
        ⚠️ **Generator model not loaded!**
        
        Please ensure you have:
        1. Trained the GAN using the notebook (`notebooks/gan_training.ipynb`)
        2. Saved the generator model to `models/generator.pth`
        
        **Steps to train the model:**
        - Open `notebooks/gan_training.ipynb`
        - Run all cells to train the conditional GAN
        - The model will be automatically saved to the correct location
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Random Generation (bonus feature)
# ---------------------------------------------------------------------------
st.header("🎲 Random Digit Generation")

if generator is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Generate random digits from all classes at once.")
    
    with col2:
        if st.button("🔄 Generate Random Mix", use_container_width=True, key="random_gen_btn"):
            st.session_state.random_gen = True
    
    if "random_gen" in st.session_state and st.session_state.random_gen:
        with st.spinner("🎲 Generating random digits..."):
            with torch.no_grad():
                # Generate with random labels
                z = torch.randn(16, 100, device=DEVICE)
                labels = torch.randint(0, 10, (16,), device=DEVICE)
                fake_images = generator(z, labels)
                fake_images_display = (fake_images + 1) / 2
                
                st.session_state.random_images = fake_images_display.cpu().numpy()
                st.session_state.random_labels = labels.cpu().numpy()
        
        st.session_state.random_gen = False
    
    if "random_images" in st.session_state:
        st.subheader("Random Generated Digits")
        
        images = st.session_state.random_images
        labels = st.session_state.random_labels
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(16):
            img = images[i, 0]
            label = labels[i]
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Class: {label}', fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Random Mix of Generated Digits', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("Load the generator model to use random generation feature.")

