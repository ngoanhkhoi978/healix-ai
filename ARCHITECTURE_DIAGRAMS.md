# Healix AI - Model Architecture Diagrams

## Overview of Both Models

```
┌─────────────────────────────────────────────────────────────┐
│                      HEALIX AI SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────┐         ┌─────────────────────┐    │
│  │   X-ray Module     │         │    MRI Module       │    │
│  │                    │         │                     │    │
│  │  ┌──────────────┐  │         │  ┌───────────────┐  │    │
│  │  │   RFDETR     │  │         │  │ TransformerU  │  │    │
│  │  │   Medium     │  │         │  │     Net       │  │    │
│  │  └──────────────┘  │         │  └───────────────┘  │    │
│  │                    │         │                     │    │
│  │  Input: X-ray     │         │  Input: MRI Scan   │    │
│  │  Output: Boxes    │         │  Output: Mask      │    │
│  └────────────────────┘         └─────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 1. X-ray Detection Model (RFDETR) Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     RFDETR Architecture                       │
└──────────────────────────────────────────────────────────────┘

Input Image (H×W×3)
      │
      ▼
┌─────────────────┐
│  Backbone CNN   │  Extract features
│  (ResNet-like)  │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Transformer   │  Self-attention on features
│    Encoder      │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│   Transformer   │  Query-based detection
│    Decoder      │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Prediction     │  → Bounding Boxes (x1,y1,x2,y2)
│    Heads        │  → Class Labels (11 lung diseases)
└─────────────────┘  → Confidence Scores (0-1)
```

## 2. MRI Segmentation Model (TransformerUNet) Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   TransformerUNet Architecture                       │
└─────────────────────────────────────────────────────────────────────┘

Input (224×224×3)
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ENCODER PATH                                 │
└─────────────────────────────────────────────────────────────────────┘
      │
      ├─────→ Skip1 (3→32)  ──────────────────────────┐
      ▼                                                │
   MaxPool                                             │
      │                                                │
      ├─────→ Skip2 (32→64) ──────────────────────┐   │
      ▼                                            │   │
   MaxPool                                         │   │
      │                                            │   │
      ├─────→ Skip3 (64→128) ──────────────────┐  │   │
      ▼                                         │  │   │
   MaxPool                                      │  │   │
      │                                         │  │   │
      ├─────→ Skip4 (128→256) ──────────────┐  │  │   │
      ▼                                      │  │  │   │
   MaxPool                                   │  │  │   │
      │                                      │  │  │   │
      ▼                                      │  │  │   │
┌─────────────────────────┐                 │  │  │   │
│     BOTTLENECK          │                 │  │  │   │
│   ConvBlock (256→512)   │                 │  │  │   │
│          +              │                 │  │  │   │
│  Positional Encoding    │                 │  │  │   │
│          +              │                 │  │  │   │
│  Multi-Head Self-Attn   │                 │  │  │   │
│      (4 heads)          │                 │  │  │   │
└─────────────────────────┘                 │  │  │   │
      │                                      │  │  │   │
      ▼                                      │  │  │   │
┌─────────────────────────────────────────────────────────────────────┐
│                         DECODER PATH                                 │
└─────────────────────────────────────────────────────────────────────┘
      │                                      │  │  │   │
      │◄─────Cross-Attention────────────────┘  │  │   │
      ▼                                         │  │   │
   ConvTranspose (512→256)                      │  │   │
      │                                         │  │   │
      │◄─────Cross-Attention─────────────────────┘  │   │
      ▼                                              │   │
   ConvTranspose (256→128)                           │   │
      │                                              │   │
      │◄─────Cross-Attention──────────────────────────┘   │
      ▼                                                    │
   ConvTranspose (128→64)                                  │
      │                                                    │
      │◄─────Cross-Attention───────────────────────────────┘
      ▼
   ConvTranspose (64→32)
      │
      ▼
┌─────────────────┐
│  Final Conv 1×1 │  Output: Binary Mask (224×224×1)
└─────────────────┘
```

## 3. Attention Mechanisms in TransformerUNet

### Multi-Head Self-Attention (Bottleneck)
```
Input Features (B×512×H×W)
      │
      ├─→ Query (Q)  ─┐
      ├─→ Key   (K)  ─┤
      └─→ Value (V)  ─┘
            │
            ▼
    ┌─────────────────┐
    │   Attention     │   Attention(Q,K,V) = softmax(QK^T/√d)V
    │   (4 heads)     │
    └─────────────────┘
            │
            ▼
    Output Features (B×512×H×W)
```

### Multi-Head Cross-Attention (Decoder)
```
Skip Connection (S)          Decoder Features (Y)
       │                             │
       ▼                             ▼
   Conv+Pool                     Conv 1×1
       │                             │
       ├─→ Key (K), Value (V)        └─→ Query (Q)
       │                                   │
       └───────────────┬───────────────────┘
                       │
                       ▼
               ┌─────────────────┐
               │ Cross-Attention │   Cross-Attn(Q,K,V)
               │   (4 heads)     │
               └─────────────────┘
                       │
                       ▼
                  ConvTranspose
                       │
                       ▼
               Gated Multiply with S
                       │
                       ▼
               Enhanced Features
```

## 4. Complete Data Flow

### X-ray Analysis Pipeline
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Upload Image │ --> │  RFDETR      │ --> │  Draw Boxes  │ --> │   Return     │
│  (JPEG/PNG)  │     │  Inference   │     │  & Labels    │     │  Annotated   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     11 Lung Diseases:
                     - Aortic enlargement, Atelectasis,
                       Cardiomegaly, Consolidation, ILD,
                       Infiltration, Lung Opacity, Other lesion,
                       Pleural effusion, Pneumothorax, Pulmonary fibrosis
                     - Confidence scores
                     - Bounding boxes
```

### MRI Segmentation Pipeline
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Upload Image │ --> │ Resize 224²  │ --> │ TransformerU │ --> │ Post-process │
│  (JPEG/PNG)  │     │  Normalize   │     │     Net      │     │   Cleaning   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                  │                      │
                                                  ▼                      ▼
                                           Binary Mask            Largest Component
                                           (224×224)              (Connected Comp)
                                                                         │
                                                                         ▼
                                                                  ┌──────────────┐
                                                                  │   Overlay    │
                                                                  │  on Original │
                                                                  └──────────────┘
```

## 5. Model Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL COMPARISON                                    │
├──────────────┬─────────────────────────┬────────────────────────────────────┤
│  Feature     │  RFDETR (X-ray)         │  TransformerUNet (MRI)             │
├──────────────┼─────────────────────────┼────────────────────────────────────┤
│  Task        │  Object Detection       │  Semantic Segmentation             │
│  Complexity  │  Medium                 │  High (Attention)                  │
│  Parameters  │  ~20-30M                │  ~15-25M                           │
│  Input       │  Variable Size          │  Fixed 224×224                     │
│  Output      │  Boxes + Classes        │  Pixel-wise Mask                   │
│  Speed       │  Fast (Real-time)       │  Medium (~1-2s)                    │
│  GPU Memory  │  2-4 GB                 │  3-5 GB                            │
│  Attention   │  Transformer (built-in) │  Multi-head (custom)               │
└──────────────┴─────────────────────────┴────────────────────────────────────┘
```

## 6. Key Components Summary

### RFDETR Components
1. **Backbone**: Feature extraction (CNN-based)
2. **Transformer Encoder**: Self-attention on features
3. **Transformer Decoder**: Query-based object detection
4. **Prediction Heads**: Box regression + classification

### TransformerUNet Components
1. **Encoder Layers**: 
   - ConvBlock (Conv → BN → ReLU)
   - MaxPooling for downsampling
   - Skip connections

2. **Bottleneck**:
   - ConvBlock (512 channels)
   - Positional Encoding (sinusoidal)
   - Multi-Head Self-Attention (4 heads)

3. **Decoder Layers**:
   - Multi-Head Cross-Attention (skip × decoder)
   - ConvTranspose for upsampling
   - Gated attention (sigmoid)

4. **Output Layer**:
   - Conv 1×1 (32 → 1 channel)
   - Sigmoid activation
   - Binary threshold (0.5)

## 7. Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                   Technology Stack                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Deep Learning:                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   PyTorch    │  │    rfdetr    │  │  torch.nn    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  Image Processing:                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     PIL      │  │     cv2      │  │ Albumentations│     │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  API & Server:                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │   FastAPI    │  │   Uvicorn    │                        │
│  └──────────────┘  └──────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

**Note**: These diagrams provide a visual overview of the model architectures. For detailed implementation, see the source code in `models/` directory.
