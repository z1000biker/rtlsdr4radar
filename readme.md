# RTLSDR Passive Radar 📡🛰️

A passive motion detection system using FM radio signals and a single RTL-SDR dongle. This project detects human motion by analyzing phase and amplitude variations in reflected FM broadcasts, turning a $25 USB dongle into a sophisticated radar sensor.
This project reflects my interest in hardware-adjacent systems, where timing, signal integrity, and data pipelines matter. The same principles I apply in backend and crypto systems — determinism, observability, and explicit state handling — apply here at the RF and device boundary.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![SDR](https://img.shields.io/badge/SDR-RTL--SDR-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Key Achievement
Human motion detection with a **single antenna setup** - typically considered extremely difficult in passive radar applications.

## Overview
This project explores real-time signal acquisition and processing using RTL-SDR
hardware for radar-like applications.

The focus is on:
- Real-time data ingestion
- Signal processing pipelines
- Performance constraints
- Hardware–software interaction

## Why this matters for software engineers
Although hardware-assisted, this project demonstrates:
- Streaming data pipelines
- Real-time constraints
- Low-level performance trade-offs
- System reliability considerations

## 🚀 Quick Start (Windows/Linux/macOS)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/z1000biker/rtlsdr4radar.git
cd rtlsdr4radar

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage
```bash
# Run the standard detector (uses 103.7 MHz by default)
python radar.py

# Or use the Makefile
make run
```

### 3. Advanced Usage (Dual-Frequency)
```bash
# Run the dual-frequency analyzer for better robustness
python df.py --freq1 102.5 --freq2 105.2
```

## 📋 Overview
This project demonstrates passive radar motion detection using consumer hardware (RTL-SDR). Unlike traditional radar systems that transmit their own signals, this system uses existing FM radio broadcasts as an "illuminator of opportunity" and detects motion by observing how moving objects disturb the received signal pattern.
## How It Works
### The Physics
#### 1. FM Signal as Ambient Illumination
FM radio stations continuously broadcast at 88-108 MHz with high power. These signals:
*   Penetrate buildings and fill indoor spaces
*   Reflect off all surfaces: walls, furniture, human bodies
*   Create complex interference patterns (standing waves)
The wavelength at 103.7 MHz is approximately 2.9 meters, making it ideal for detecting human-sized objects (1.5-1.8m).
#### 2. The Multipath Environment
When an FM signal enters a room, it doesn't travel in a straight line. Instead:
```
FM Tower (XX kW) → Your Room:
  ├─ Direct path → Antenna
  ├─ Wall reflection → Antenna
  ├─ Floor reflection → Antenna
  ├─ Ceiling reflection → Antenna
  ├─ **YOUR BODY** reflection → Antenna (this changes when you move!)
  └─ Furniture reflections → Antenna
```
All these paths have different lengths, so the waves arrive at different phases. They constructively and destructively interfere, creating a unique signal pattern.
#### 3. Motion Detection Principle
When you move:
*   **Path Length Changes:**
    *   A person moving 10 cm changes the reflection path by 20 cm (there and back)
    *   At λ = 2.9m, this is a phase shift of: Δφ = (2π × 0.2m) / 2.9m ≈ 0.43 radians
*   **Interference Pattern Shifts:**
    *   Waves that were canceling (destructive interference) might now add (constructive)
    *   The total received power fluctuates
    *   The phase of the received signal changes rapidly
*   **Human Body as Reflector:**
    *   Radar Cross Section (RCS) of human: ~0.5-1.5 m²
    *   At 103.7 MHz, a standing human is approximately λ/2 tall (efficient reflector)
    *   Moving limbs create additional micro-Doppler signatures
#### 4. Detection Algorithm
The system uses a multi-method approach:
*   **Method 1: Phase Variance Detection**
    *   `Phase(t) = atan2(Q(t), I(t))`  # Extract instantaneous phase
    *   `dφ/dt = diff(unwrap(Phase))`   # Phase derivative
    *   `Motion Metric = var(dφ/dt)`     # Variance indicates motion
    *   When stationary: Phase changes slowly and smoothly (just the carrier)
    *   When moving: Phase fluctuates rapidly and irregularly
*   **Method 2: Power Variance Detection**
    *   `Power(t) = |I(t) + jQ(t)|²`    # Signal magnitude squared
    *   `Recent Variance = var(Power[-20:])`     # Last 2 seconds
    *   `Baseline Variance = var(Power[-40:-20])` # Previous 2 seconds
    *   `Motion Score = Recent / Baseline`
*   **Method 3: Temporal Filtering**
    *   Requires 3 consecutive detections to confirm motion (reduce false positives)
    *   Requires 5 consecutive "still" readings to clear alarm
    *   Prevents flickering and noise-induced false alarms
### Why Single-Antenna Detection Works
Traditional passive radar requires:
*   Reference antenna: Points at FM tower (direct signal)
*   Surveillance antenna: Points at target area (reflected signals)
*   Physical separation: 10+ meters apart
This system works with one antenna because:
*   We're not trying to separate signals - We embrace the mixture!
*   Motion creates disturbances - We detect the change in the interference pattern
*   Room acts as a resonant cavity - The enclosed space amplifies small reflections
*   Statistical approach - We compare variance over time, not absolute measurements
*   Wavelength advantage - 3-meter waves interact strongly with human-sized objects
*Think of it like dropping a pebble in a pond - you don't need to see the pebble, you just need to detect the ripples.*
### Mathematical Foundation
**Doppler Shift from Human Motion:**
`f_doppler = (2 × v × f_carrier) / c`
For v = 1 m/s (walking), f = 103.7 MHz:
`f_doppler = (2 × 1 × 103.7×10⁶) / 3×10⁸ ≈ 0.69 Hz`
**Signal Model:**
Received Signal = Direct + Σ(Reflections)
`s(t) = A₀·e^(j2πft) + Σ Aᵢ·e^(j2πf(t-τᵢ(t)))`
                      i=1
Where τᵢ(t) changes with motion
**Phase Sensitivity:**
For a target at range R:
`Phase = 4πR / λ`
A 1 cm movement creates:
`Δφ = 4π × 0.01 / 2.9 ≈ 0.043 radians ≈ 2.5°`
This high sensitivity allows detection of even small movements.
## Hardware Requirements
### Required
*   **RTL-SDR dongle** (RTL-SDR V3 or V4 recommended)
    *   Any RTL2832U-based SDR will work
    *   Tested with RTL-SDR Blog V4
*   **Antenna** (usually included with RTL-SDR)
    *   Dipole antenna works well
    *   Position near window for best FM reception
*   **Computer with USB port**
    *   Linux, Windows, or macOS
    *   Python 3.7 or higher
### Recommended
*   Strong local FM station (check 88-108 MHz)
*   Indoor environment (works better in enclosed spaces)
*   Clear line of sight between antenna and detection area
## Software Installation (Windows 11)
### 1. Install RTL-SDR Drivers
1.  Plug in your RTL-SDR v4 dongle.
2.  Download **Zadig** from [zadig.akeo.ie](https://zadig.akeo.ie/).
3.  Open Zadig and select **Options -> List All Devices**.
4.  Select "RTL2838UHIDIR" (or similar "Bulk-In, Interface (Interface 0)").
    *   *Note: Windows might auto-install a default driver. We need to replace it.*
5.  Ensure the target driver is **WinUSB**.
6.  Click **Replace Driver** (or Install Driver).
### 2. Install Python Dependencies
Open PowerShell or Command Prompt and run:
```powershell
pip install numpy scipy matplotlib pyrtlsdr
```
### 3. (Optional) Install RTL-SDR Command Line Tools
To use commands like `rtl_test` or `rtl_fm`:
1.  Download the **RTL-SDR Blog V4 Windows Release** zip from the [RTL-SDR Blog V4 Drivers](https://github.com/rtlsdrblog/rtl-sdr-blog/releases) page.
2.  Extract the zip file.
3.  Copy the files (including `rtlsdr.dll`) to a folder in your system PATH, or place them in the same folder as this script.
    *   *Crucial for V4 users: You need the specific V4-compatible `rtlsdr.dll` if you want to use these tools, though the Python script often uses the one installed by `pyrtlsdr` or system libs.*
## Usage
### Basic Usage
```powershell
# Use default settings (103.7 MHz, auto gain)
python radar.py
# Specify custom FM frequency
python radar.py --freq 98.5
# Use auto gain (recommended for RTL-SDR V4)
python radar.py --freq 103.7 --gain auto
```
The program will:
1.  Initialize RTL-SDR at 103.7 MHz (hardcoded or via arg)
2.  Collect baseline for 5 seconds (**stay still!**)
3.  Start monitoring for motion
4.  Display real-time graphs
### Understanding the Display
The program shows 4 plots:
1.  **Signal Strength Over Time (Top)**
    *   **Y-axis:** Power (dB) - Radio frequency signal power in decibels
    *   **X-axis:** Time (HH:MM:SS) - Real clock time
    *   **Blue line:** Instantaneous signal power measurement
    *   **Red shading:** Periods when motion is confirmed
    *   **Yellow banner:** Appears when motion is currently detected
    *   *Typical values:* -40 to -30 dB (strong signal), -60 to -50 dB (weak signal)
2.  **Signal Variance (Middle-Left)**
    *   **Y-axis:** Variance (power units squared)
    *   **Green bar:** Baseline variance (last 2-4 seconds of historical data)
    *   **Blue bar:** Recent variance (last 2 seconds) when stationary
    *   **Red bar:** Recent variance when motion detected
    *   *Interpretation:* Higher recent variance indicates signal fluctuation from motion
    *   *Typical values:* Baseline ~0.001-0.01, Motion ~0.01-0.1 (depends on signal strength)
3.  **Motion Detection Score (Middle-Right)**
    *   **Y-axis:** Motion Score (unitless ratio, 0-10 scale)
    *   **X-axis:** Time (HH:MM:SS)
    *   **Green line:** (Recent Variance / Baseline Variance) × (Recent Rate of Change / Baseline Rate of Change)
    *   **Red dashed line:** Detection threshold = 3.0 (or 1.5 in improved version)
    *   *Interpretation:* Score > Threshold triggers motion detection
    *   *Typical values:* Still ~0.5-2.0, Motion ~3.0-8.0
4.  **Current Spectrum (Bottom)**
    *   **Y-axis:** Power (dB) - Spectral power density
    *   **X-axis:** Frequency Offset (kHz) - Offset from center frequency
    *   **Shows:** FFT (Fast Fourier Transform) of captured samples
    *   **Purpose:** Diagnose signal strength and interference
    *   **What to look for:**
        *   Strong peak at center = good FM signal
        *   Flat/noisy = weak signal or no station
        *   Multiple peaks = interference or adjacent stations
### Terminal Output
```
[14:35:22] ✓ Still (score: 1.23)
[14:35:23] ⚠️  Possible motion (confirming... 1/3)
[14:35:24] ⚠️  Possible motion (confirming... 2/3)
[14:35:25] 🚨 MOTION (score: 4.56, consecutive: 3)
[14:35:27] 🚨 MOTION (score: 5.12, consecutive: 5)
   → Motion lasted 3.2 seconds
```
## Tips for Best Performance
### 🚨 CRITICAL: During Baseline Collection (First 4-5 Seconds)
*   **Stay COMPLETELY still** - Don't even breathe heavily
*   Don't move furniture or objects in the room
*   Close doors/windows to minimize air currents
*   Turn off fans/AC if possible during baseline
*   Wait for "Collecting baseline..." message before moving
*   *This establishes what the "quiet" signal looks like*
*   **Moving during baseline = system won't detect motion properly**
### For Best Detection After Baseline
*   Walk directly toward or away from antenna (radial motion)
*   Wave arms or move objects
*   Stay within 2-5 meters of antenna
*   Works best in enclosed rooms (multipath enhancement)
### If Not Detecting Motion
*   Check signal strength (should be > -50 dB)
*   Try different FM frequency
*   Move antenna to window
*   Increase detection sensitivity (modify motion_score threshold)
## Configuration
The improved version supports command-line arguments:
```bash
# Change FM frequency
python radar.py --freq 98.5
# Set manual gain (0-49 dB)
python radar.py --gain 45
# Use auto gain (recommended)
python radar.py --gain auto
```
To modify detection sensitivity, edit `radar.py` (or `rf_motion_detector.py`):
```python
# Line 50: Detection Threshold
motion = motion_score > 1.5     # Lower = more sensitive (default: 1.5)
                                # Increase to 2.0 or 2.5 if too many false positives
                                # Decrease to 1.0 for maximum sensitivity
# Line 72: Temporal Filter
if consecutive_motion >= 3:     # Detections needed to confirm (default: 3)
if consecutive_still >= 5:      # Still readings needed to clear (default: 5)
# Line 23: Signal Smoothing
def _smooth(self, values, window=5):  # Moving average window (default: 5)
                                       # Increase for more smoothing (6-10)
                                       # Decrease for faster response (3-4)
```
### Key Parameters Explained
| Parameter | Default | Effect |
| :--- | :--- | :--- |
| `--freq` | 103.7 | FM station frequency (MHz) |
| `--gain` | auto | RF gain (0-49 dB or 'auto') |
| Detection threshold | 1.5 | Motion score needed (lower = more sensitive) |
| Smoothing window | 5 | Samples averaged (higher = smoother) |
| Consecutive detections | 3 | Confirmations needed before alarm |
| Consecutive still | 5 | Still readings to clear alarm |
## Troubleshooting
*   **"No RTL-SDR device found"**
    *   Check USB connection
    *   Verify drivers installed: `rtl_test`
    *   Try different USB port
    *   Check permissions (Linux): `sudo usermod -a -G plugdev $USER`
*   **"Signal too weak"**
    *   Move antenna to window
    *   Try different FM frequency
    *   Increase gain in code
    *   Check antenna connection
*   **"Too many false positives"**
    *   Increase threshold: `motion_score > 4.0` or `5.0`
    *   Close windows (reduces external motion)
    *   Increase consecutive detection requirement
    *   Move away from fans, AC vents
*   **"Not detecting my motion"**
    *   Check baseline: Did you move during the first 5 seconds? If yes, restart the program
    *   Move toward/away from antenna (not side-to-side)
    *   Stay within 5 meters
    *   Make larger movements
    *   Check signal strength is adequate
    *   Lower threshold: `motion_score > 1.0` or even `0.8`
    *   Reduce smoothing window to 3 for faster response
## Technical Specifications
| Parameter | Value |
| :--- | :--- |
| **Frequency** | 103.7 MHz (FM broadcast, configurable) |
| **Wavelength** | 2.9 meters |
| **Sample Rate** | 2.4 MHz |
| **Bandwidth** | 2.4 MHz |
| **Gain** | Auto (or 0-49 dB manual) |
| **Detection Threshold** | 1.5 (configurable) |
| **Signal Smoothing** | 5-sample moving average |
| **Detection Range** | 0.5-5 meters (typical) |
| **Velocity Range** | 0.1-5 m/s (0.4-18 km/h) |
| **Update Rate** | ~10 Hz |
| **Latency** | ~200 ms |
## Limitations
*   **Single Antenna Constraint:** Cannot determine range or direction; only detects presence/absence of motion. Less sensitive than dual-antenna systems.
*   **Environmental Factors:** Works best indoors (multipath enhancement). Affected by large metal objects. Windows, fans, curtains can cause false positives.
*   **Detection Characteristics:** Most sensitive to radial motion (toward/away). Less sensitive to tangential motion (side-to-side). Cannot distinguish between multiple people. Cannot identify specific individuals.
*   **Signal Dependency:** Requires strong FM broadcast station. Performance degrades with weak signals. May not work in RF-shielded rooms.
## Scientific Applications
This simple system demonstrates principles used in:
*   **WiFi Sensing** - Detecting motion through WiFi signal disruption
*   **Through-Wall Radar** - Military/rescue applications
*   **Passive Bistatic Radar** - Aircraft detection using TV/radio broadcasts
*   **Healthcare Monitoring** - Non-contact breathing/heartbeat detection
*   **Smart Home Systems** - Occupancy and activity recognition
## Future Enhancements
Potential improvements for advanced users:
*   Direction of arrival estimation using antenna arrays
*   Range estimation using time-domain analysis
*   Multiple target tracking using particle filters
*   Activity classification (walking, sitting, running) using ML
*   Breathing/heartbeat detection at closer ranges
*   Web interface for remote monitoring
*   Data logging and historical analysis
*   Multi-frequency operation for improved detection
## Contributing
Contributions welcome! Areas of interest:
*   Algorithm improvements
*   Support for different SDR hardware
*   Machine learning integration
*   Documentation improvements
*   Bug reports and fixes
## References
*   **Academic Papers:**
    *   Griffiths, H. D., & Baker, C. J. (2017). "An Introduction to Passive Radar"
    *   Colone, F., et al. (2016). "WiFi-Based Passive Bistatic Radar"
    *   Adib, F., & Katabi, D. (2013). "See Through Walls with WiFi!"
*   **Technical Resources:**
    *   RTL-SDR Documentation: https://www.rtl-sdr.com/
    *   Passive Radar Theory: https://en.wikipedia.org/wiki/Passive_radar
    *   PyRTLSDR Library: https://pyrtlsdr.readthedocs.io/
## License
MIT License - See LICENSE file for details
## Acknowledgments
*   RTL-SDR developers and community
*   PyRTLSDR library maintainers
*   FM radio stations providing free illumination!
## Author
Created by z1000biker
*Disclaimer: This project is for educational purposes. Ensure compliance with local regulations regarding RF reception and monitoring. Do not use for surveillance without proper authorization.*
---
### Quick Start Summary
```powershell
# 1. Install dependencies
pip install numpy scipy matplotlib pyrtlsdr
# 2. (Optional) Find strong FM station using SDR# or similar GUI tool
#    Or just try known local stations (e.g., 103.7, 98.5)
# 3. Run detector
python radar.py --freq 103.7 --gain auto
# 4. IMPORTANT: When you see "Collecting baseline..."
#    → FREEZE! Don't move for 5 seconds!

```
**Common Mistake:** Moving during baseline collection → Restart if you moved!
**It works! Human motion detection with a $25 USB dongle! 🎉**

## Possible Extensions
- Web-based visualization dashboard
- Backend API for processed data
- Cloud-based processing pipeline
- 
