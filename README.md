# ThoughtLink: Brain-to-Robot Intent Decoding

**Hack Nation 4th Global AI Hackathon - Feb 7-8, 2026**

Multimodal EEG+TD-NIRS brain signal classification for real-time robot control.

---

## Strategy

**Hybrid Approach:**
- **Research advantages**: M2 variance moments, short-channel regression, motor-specific EEG bands
- **ThoughtLink constraints**: Start simple, stay fast (<50ms latency), scale only if justified

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python data/download.py
```

### 3. Train Model
```bash
python train.py
```

### 4. Test Prediction
```bash
python predict.py data/cache/robot_control/<sample_file>.npz
```

### 5. Benchmark Latency
```bash
python benchmark.py
```

---

## Project Structure

```
thoughtlink/
├── config.yaml           # All hyperparameters
├── train.py             # Training pipeline
├── predict.py           # Prediction interface (for hackathon integration)
├── benchmark.py         # Latency testing
├── src/
│   ├── preprocessing.py # EEG + fNIRS preprocessing
│   ├── features.py      # Feature extraction
│   ├── models.py        # SVM, RF, Ensemble
│   ├── inference.py     # Real-time predictor
│   └── utils.py         # Metrics, visualization
└── data/
    └── download.py      # HuggingFace downloader
```

---

## Key Technical Decisions

### 1. TD-NIRS is Primary Signal (Competitive Edge)
- 6 EEG channels miss motor cortex (C3/C4)
- 40 fNIRS modules provide full head coverage
- **M2 variance moment**: Highest cortical selectivity
- **Short-channel regression**: Removes scalp contamination (most teams miss this)

### 2. Simple Baseline First (ThoughtLink Constraint)
- Linear SVM: <10ms inference, proven effective
- Binary (Left vs Right) → 5-class progression
- Only scale complexity if justified

### 3. Confidence + Smoothing (Prevent False Triggers)
- Threshold: 0.6
- Temporal smoothing: Moving average over 3 predictions
- Critical for robot stability

---

## Dataset

**Classes**: Right Fist, Left Fist, Both Fists, Tongue Tapping, Relax

**EEG**: (7499, 6) @ 500 Hz
- Channels: AFF6, AFp2, AFp1, AFF5, FCz, CPz

**TD-NIRS**: (72, 40, 3, 2, 3) @ 4.76 Hz
- 72 samples, 40 modules, 3 SDS, 2 wavelengths, 3 moments

**Timing**:
- t=0-3s: Baseline
- t=3s: Stimulus presented
- t=8-12s: Peak hemodynamic response

---

## Performance Targets

- **Accuracy**: >40% (vs 20% chance)
- **Latency**: <50ms (real-time constraint)
- **Stability**: No oscillation or false triggers

---

## Hackathon Integration

Your model outputs:
```python
{
    'command': 'RIGHT_FIST',  # or LEFT_FIST, BOTH_FISTS, TONGUE_TAPPING, RELAX
    'confidence': 0.85,
    'latency_ms': 12.3
}
```

Their simulation script:
- Calls your `predict()` function
- Renders robot action in real-time

---

## Tips for Success

1. **Preprocessing is critical**: Short-channel regression on fNIRS
2. **Feature selection matters**: Focus on M2 variance, peak window
3. **Keep it simple**: SVM beats complex models in hackathons
4. **Track latency from day 1**: Don't optimize accuracy then discover it's too slow

---

## Citation

If using this approach, please reference:
- Kernel Flow TD-NIRS technology
- ThoughtLink challenge framework
- Short-channel regression methodology (Gagnon et al., 2011)
