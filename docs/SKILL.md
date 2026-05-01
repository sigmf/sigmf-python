# SKILL.md - SigMF Python

## Description
SigMF Python library skill for AI agents to work with Signal Metadata Format recordings. This skill enables AI agents to read, write, and process RF time series data using the SigMF standard.

## Use Cases
- Reading and analyzing RF signal recordings
- Converting between different RF data formats
- Processing and validating SigMF metadata
- Creating automated RF signal analysis workflows
- Generating reports from RF signal data

## Setup
```python
import sigmf
```

## Core Operations

### Read SigMF Recordings
```python
# Read SigMF recording
meta = sigmf.fromfile("recording.sigmf-meta")
samples = meta[0:1024]  # Get first 1024 samples
sample_rate = meta.sample_rate  # Get sample rate

# Read other formats as SigMF
meta = sigmf.fromfile("recording.wav")   # WAV files
meta = sigmf.fromfile("recording.cdif")  # BLUE/Platinum files  
meta = sigmf.fromfile("recording.xml")   # Signal Hound Spike files
```

### Signal Analysis
```python
# Basic signal properties
sample_rate = meta.sample_rate
center_freq = meta.core:frequency
datetime = meta.core:datetime
annotation = meta.annotations[0] if meta.annotations else None

# Process signal data
samples = meta.read_samples(start_index=0, count=1024)
```

### Format Conversion
```python
# Convert between RF formats
# WAV to SigMF
sigmf.fromfile("input.wav").tofile("output.sigmf-meta")

# CDIF to SigMF  
sigmf.fromfile("input.cdif").tofile("output.sigmf-meta")

# Signal Hound to SigMF
sigmf.fromfile("input.xml").tofile("output.sigmf-meta")
```

### Metadata Operations
```python
# Access metadata
global_fields = meta.global_fields
global_keys = list(meta.global_fields.keys())

# Validate SigMF files
sigmf.validate.validate_file("recording.sigmf-meta")
```

## Advanced Features

### Signal Generation
```python
# Generate test signals
from sigmf import siggen

# Create sine wave
samples = siggen.sinewave(frequency=1e6, sample_rate=10e6, duration=1.0)

# Create noise
samples = siggen.white_noise(sample_rate=10e6, num_samples=1024)
```

### Archive Operations
```python
# Handle SigMF archives
archive = sigmf.Archive("archive.sigmf-meta")
archive.add_file("recording1.sigmf-meta")
archive.add_file("recording2.sigmf-meta")
archive.write("combined.sigmf-meta")
```

## Best Practices
1. Always validate SigMF files before processing
2. Use appropriate sample rates for your analysis
3. Include proper metadata annotations
4. Handle large files in chunks when memory is constrained
5. Use proper units for frequency (Hz) and time (seconds)

## Common Tasks

### RF Signal Analysis Workflow
```python
def analyze_rf_signal(file_path):
    """Complete RF signal analysis workflow"""
    # Load signal
    meta = sigmf.fromfile(file_path)
    
    # Get basic properties
    sample_rate = meta.sample_rate
    center_freq = meta.core.get('frequency', 0)
    
    # Read samples
    samples = meta.read_samples(count=1024)
    
    # Basic analysis
    max_amplitude = max(abs(samples))
    avg_amplitude = sum(abs(samples)) / len(samples)
    
    return {
        'sample_rate': sample_rate,
        'center_frequency': center_freq,
        'max_amplitude': max_amplitude,
        'avg_amplitude': avg_amplitude
    }
```

### Batch Processing
```python
def batch_convert_rf_files(input_dir, output_dir, input_format='wav'):
    """Convert multiple RF files to SigMF format"""
    import glob
    
    pattern = f"{input_dir}/*.{input_format}"
    for input_file in glob.glob(pattern):
        output_file = input_file.replace(input_format, 'sigmf-meta')
        meta = sigmf.fromfile(input_file)
        meta.tofile(output_file)
        print(f"Converted {input_file} to {output_file}")
```

## Error Handling
```python
try:
    meta = sigmf.fromfile("signal.sigmf-meta")
    samples = meta.read_samples()
except sigmf.SigMFError as e:
    print(f"SigMF error: {e}")
except FileNotFoundError:
    print("File not found")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## References
- [SigMF Specification](https://sigmf.org/)
- [SigMF Python Documentation](https://sigmf.readthedocs.io/)
- [Signal Metadata Format](https://github.com/sigmf/SigMF)

## Notes
- Compatible with Python 3.7-3.14
- Licensed under GNU Lesser GPL v3
- Supports various RF recording formats
- Includes validation and conversion utilities