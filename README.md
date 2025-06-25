# 🎨 Sketch-to-Image AI Generator

Transform your sketches into stunning realistic images using state-of-the-art deep learning technology. This project implements an **Attention-based UNet** architecture that converts hand-drawn sketches into photorealistic artwork.

![Sketch to Image Banner](https://pplx-res.cloudinary.com/image/upload/v1750264916/gpt4o_images/d3lbvspbric3wkcfqeub.png)

## ✨ Features

- **🔥 Attention-based UNet Architecture** - Advanced neural network with selective feature propagation
- **🎯 Residual Connections** - Enhanced gradient flow for deeper networks
- **📈 Progressive Training** - Scale from 256px to 1024px+ resolutions
- **🚀 Multi-GPU Support** - Accelerated training on multiple GPUs
- **💾 Smart Checkpointing** - Resume training from any point
- **🎨 Data Augmentation** - Robust training with advanced transformations

## 🏗️ Architecture

Our model features a sophisticated **Attention UNet** with the following components:

- **AttentionGate**: Selective feature propagation mechanism
- **ResidualBlock**: Skip-connection blocks for better gradient flow
- **Progressive Scaling**: Transfer weights between different resolutions
- **Instance Normalization**: Stable training dynamics

```
Input Sketch (256x256) → Encoder → Bottleneck → Attention Decoder → Generated Image (256x256)
                ↓                                        ↑
            Skip Connections ←→ Attention Gates ←→ Feature Selection
```


## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision albumentations opencv-python pillow numpy
```

### 📁 Project Structure

```
sketch-to-image/
├── models/
│   ├── unet.py              # Main UNet architecture
│   └── weights/             # Model checkpoints
├── dataset/
│   └── sketch_dataset.py    # Custom dataset loader
├── data/
│   ├── train/
│   │   ├── sketches/        # Training sketches
│   │   └── originals/       # Target images
│   └── test/                # Validation data
├── train.py                 # Training script
├── inference.py             # Generate images
└── utils.py                 # Training utilities
```


### 🎯 Training

```bash
# Start training from scratch
python train.py

# Training will automatically:
# ✓ Resume from checkpoints
# ✓ Apply data augmentation
# ✓ Save best models
# ✓ Log training progress
```

### 🎨 Generate Images

```bash
# Generate from a sketch
python inference.py --input sketch.jpg --output generated.png

# Batch processing
python inference.py --input_dir sketches/ --output_dir results/
```

## 📊 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Image Size** | 256×256 | Input/output resolution |
| **Batch Size** | 32 | Training batch size |
| **Learning Rate** | 2e-4 | Adam optimizer rate |
| **Epochs** | 250 | Training iterations |
| **Loss Function** | L1 Loss | Pixel-level reconstruction |

## 🎨 Data Augmentation

Our training pipeline includes robust augmentations:

- **Geometric**: Horizontal flip, rotation, elastic transforms
- **Color**: Brightness, contrast, saturation adjustments
- **Noise**: Gaussian noise, coarse dropout
- **Normalization**: [-1, 1] range for stable training

## 🔧 Model Features

### **Attention Mechanism**
```python
# Selective feature propagation
attention_output = AttentionGate(gating_signal, skip_connection)
```

### **Progressive Training**
```python
# Scale from 256px → 512px → 1024px
model_512 = UNet(depth=6)
model_512.transfer_weights(model_256)
```

### **Residual Blocks**
```python
# Enhanced gradient flow
output = ResidualBlock(input) + shortcut_connection
```

## 📈 Performance

- **Training Time**: ~6 hours on RTX 3080
- **Memory Usage**: ~8GB VRAM for 256px images
- **Inference Speed**: ~50ms per image
- **Model Size**: ~45MB compressed

## 🎯 Results

Transform sketches into photorealistic images:

| Input Sketch | Generated Image | Style |
|--------------|-----------------|-------|
| 🖊️ Line art | 🎨 Photorealistic | Portrait |
| ✏️ Rough sketch | 🌅 Detailed landscape | Nature |
| 🖍️ Simple drawing | 🏢 Architectural render | Buildings |

## 🛠️ Advanced Usage

### **Custom Training**
```python
# Initialize model with custom depth
model = UNet(in_channels=3, out_channels=3, depth=6, base_channels=64)

# Apply custom transformations
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
```

### **Progressive Scaling**
```python
# Train at multiple resolutions
model_256 = UNet(depth=5)  # 256px
model_512 = UNet(depth=6)  # 512px
model_1024 = UNet(depth=7) # 1024px

# Transfer weights between scales
model_512.transfer_weights(model_256)
model_1024.transfer_weights(model_512)
```

## 🤝 Contributing

We welcome contributions! Please feel free to:

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📖 Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UNet Architecture**: Ronneberger et al.
- **Attention Mechanism**: Oktay et al.
- **PyTorch Team**: For the amazing framework
- **Community**: For datasets and inspiration

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

[🔗 Demo](https://your-demo-link.com) • [📖 Files](https://github.com/RFahmi99/Sketch-To-Art) • [🐛 Issues](https://github.com/RFahmi99/Sketch-To-Art/issues)

</div>
