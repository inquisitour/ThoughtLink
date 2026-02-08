# ThoughtLink: Brain-to-Robot Control

**Hack Nation 4th Global AI Hackathon | Feb 7-8, 2026**

Real-time motor intent decoding from EEG+fNIRS brain signals.

---

## Results

- **94.6% accuracy** (5-class motor imagery)
- **3.5ms average latency** (14x under 50ms limit)
- All classes >92% balanced performance

---

## Quick Start

```bash
pip install -r requirements.txt
python data/download.py
python train.py
python predict.py data/cache/robot_control/data/<sample>.npz
```

---

## Approach

**Competitive Edges:**
- M2 variance moment extraction (highest cortical selectivity)
- Short-channel regression (removes scalp noise)
- Sliding window augmentation (10x training data)

**Architecture:**
- Simple RBF-SVM (fast, scalable)
- Multimodal fusion: EEG bandpower + fNIRS moments
- PCA dimension reduction + confidence thresholding

---

## Dataset

5 classes: Right Fist, Left Fist, Both Fists, Tongue Tapping, Relax

- **EEG**: 6 channels @ 500Hz (frontal placement: FCz, CPz critical)
- **TD-NIRS**: 40 modules @ 4.76Hz (full head coverage)

---

## Integration

The `predict()` function outputs:
```python
{
    'command': 'RIGHT_FIST',
    'confidence': 0.98,
    'latency_ms': 3.4
}
```

Ready to plug into robot simulation using a connector script.

---

## Why It Works

- The 6 EEG channels miss motor cortex (C3/C4), so treating the 40-module fNIRS array as primary signal source. 
- Short-channel regression + M2 variance extracts cortical signals. 
- Sliding windows give the simple SVM enough data to learn robust patterns without overfitting.