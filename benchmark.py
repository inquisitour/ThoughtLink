"""
Benchmark inference latency: must be <50ms
"""
import time
import numpy as np
from pathlib import Path

from predict import BrainRobotPredictor
from src.utils import load_npz_data


def benchmark_latency(predictor, samples, n_trials=100):
    """
    Measure average inference latency
    
    Returns:
        dict with latency statistics
    """
    latencies = []
    
    print(f"Running {n_trials} inference trials...")
    
    for i in range(min(n_trials, len(samples))):
        eeg, fnirs, _ = samples[i]
        
        result = predictor.predict(eeg, fnirs)
        latencies.append(result['latency_ms'])
    
    latencies = np.array(latencies)
    
    return {
        'mean': latencies.mean(),
        'std': latencies.std(),
        'min': latencies.min(),
        'max': latencies.max(),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }


def main():
    # Load predictor
    predictor = BrainRobotPredictor(model_dir='./models')
    
    # Load test samples
    data_dir = Path('./data/cache/robot_control')
    samples = load_npz_data(data_dir)
    
    # Benchmark
    stats = benchmark_latency(predictor, samples, n_trials=100)
    
    print("\n" + "="*50)
    print("LATENCY BENCHMARK RESULTS")
    print("="*50)
    print(f"Mean:   {stats['mean']:.2f} ms")
    print(f"Std:    {stats['std']:.2f} ms")
    print(f"Min:    {stats['min']:.2f} ms")
    print(f"Max:    {stats['max']:.2f} ms")
    print(f"P50:    {stats['p50']:.2f} ms")
    print(f"P95:    {stats['p95']:.2f} ms")
    print(f"P99:    {stats['p99']:.2f} ms")
    print("="*50)
    
    # Evaluation
    target_latency = 50.0
    if stats['p95'] < target_latency:
        print(f"\n✓ PASSED: 95th percentile ({stats['p95']:.1f}ms) < {target_latency}ms")
    else:
        print(f"\n✗ FAILED: 95th percentile ({stats['p95']:.1f}ms) > {target_latency}ms")
        print("  Consider simplifying model or reducing features")


if __name__ == "__main__":
    main()
