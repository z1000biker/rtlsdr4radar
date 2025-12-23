import argparse
import numpy as np
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import time

class DualFrequencyMotionAnalyzer:
    """
    Dual-Frequency Motion Analysis with Single RTL-SDR
    
    CONCEPT: Rapidly alternate between two frequencies to:
    - Compare motion signatures across frequencies
    - Estimate relative range changes (not absolute distance)
    - Improve motion detection robustness
    - Detect frequency-dependent reflections
    
    LIMITATIONS:
    - Cannot measure absolute distance (need reference channel)
    - Frequency hopping causes gaps in temporal coherence
    - Different transmitters aren't phase-locked
    
    WHAT IT PROVIDES:
    - Differential Doppler analysis
    - Multi-frequency motion correlation
    - Improved false positive rejection
    - Frequency-dependent target signatures
    """
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="Dual-frequency motion analyzer")
        parser.add_argument("--freq1", type=float, default=102.5,
                            help="First frequency in MHz (default 102.5)")
        parser.add_argument("--freq2", type=float, default=105.2,
                            help="Second frequency in MHz (default 105.2)")
        parser.add_argument("--gain", type=str, default="auto",
                            help="Gain in dB or 'auto'")
        parser.add_argument("--hop-interval", type=float, default=0.1,
                            help="Seconds between frequency hops (default 0.1)")
        self.args = parser.parse_args()
        
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        self.sdr.gain = self.args.gain
        
        # Dual frequency setup
        self.freq1 = self.args.freq1 * 1e6
        self.freq2 = self.args.freq2 * 1e6
        self.freq_delta = abs(self.freq2 - self.freq1)
        
        # Wavelengths
        self.lambda1 = 3e8 / self.freq1
        self.lambda2 = 3e8 / self.freq2
        
        # Synthetic wavelength for ambiguity resolution
        self.lambda_synthetic = 3e8 / self.freq_delta
        
        # Separate histories for each frequency
        self.freq1_power = []
        self.freq1_phase = []
        self.freq2_power = []
        self.freq2_phase = []
        
        # Combined analysis
        self.phase_diff = []  # φ₁ - φ₂
        self.relative_range = []  # Relative distance changes
        self.correlation = []  # Cross-frequency correlation
        self.time_history = []
        self.motion_history = []
        
        # Frequency hopping state
        self.current_freq = 1  # Toggle between 1 and 2
        self.last_hop_time = time.time()
        self.hop_interval = self.args.hop_interval
        
        print("=" * 75)
        print("DUAL-FREQUENCY MOTION ANALYZER")
        print("=" * 75)
        print(f"Frequency 1: {self.args.freq1} MHz (λ = {self.lambda1:.3f} m)")
        print(f"Frequency 2: {self.args.freq2} MHz (λ = {self.lambda2:.3f} m)")
        print(f"Frequency Δ: {self.freq_delta/1e6:.2f} MHz")
        print(f"Synthetic λ: {self.lambda_synthetic:.2f} m (unambiguous range)")
        print(f"Hop interval: {self.hop_interval}s")
        print()
        print("Capabilities:")
        print("  ✓ Differential Doppler analysis")
        print("  ✓ Multi-frequency motion correlation")
        print("  ✓ Frequency-selective target detection")
        print("  ✓ Relative range change estimation")
        print()
        print("Limitations:")
        print("  ✗ No absolute distance (need reference antenna)")
        print("  ✗ Temporal gaps during frequency hopping")
        print("  ✗ Transmitters not phase-synchronized")
        print("=" * 75)
        print()
    
    def hop_frequency(self):
        """Switch between the two frequencies."""
        current_time = time.time()
        if current_time - self.last_hop_time >= self.hop_interval:
            self.current_freq = 2 if self.current_freq == 1 else 1
            freq = self.freq1 if self.current_freq == 1 else self.freq2
            self.sdr.center_freq = freq
            self.last_hop_time = current_time
            return True
        return False
    
    def extract_phase_power(self, samples):
        """Extract power and mean phase from samples."""
        power = np.mean(np.abs(samples) ** 2)
        power_db = 10 * np.log10(power + 1e-10)
        
        phases = np.angle(samples)
        phases_unwrapped = np.unwrap(phases)
        mean_phase = np.mean(phases_unwrapped)
        
        return power_db, mean_phase
    
    def estimate_relative_range_change(self):
        """
        Estimate relative range change using phase difference.
        This is RELATIVE motion, not absolute distance!
        
        ΔR = (Δφ × λ_synthetic) / (4π)
        """
        if len(self.phase_diff) < 2:
            return 0
        
        # Change in phase difference over time
        delta_phase_diff = self.phase_diff[-1] - self.phase_diff[-2]
        
        # Convert to relative range change
        range_change = (delta_phase_diff * self.lambda_synthetic) / (4 * np.pi)
        
        return range_change
    
    def cross_frequency_correlation(self):
        """
        Calculate correlation between motion at two frequencies.
        High correlation = consistent motion signature.
        Low correlation = frequency-dependent scattering or interference.
        """
        if len(self.freq1_power) < 10 or len(self.freq2_power) < 10:
            return 0
        
        # Get recent power variations
        f1_recent = np.diff(self.freq1_power[-10:])
        f2_recent = np.diff(self.freq2_power[-10:])
        
        # Normalize and correlate
        if np.std(f1_recent) > 0 and np.std(f2_recent) > 0:
            f1_norm = (f1_recent - np.mean(f1_recent)) / np.std(f1_recent)
            f2_norm = (f2_recent - np.mean(f2_recent)) / np.std(f2_recent)
            corr = np.corrcoef(f1_norm, f2_norm)[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0
    
    def detect_motion_dual_freq(self):
        """
        Enhanced motion detection using dual-frequency analysis.
        """
        if len(self.freq1_power) < 20 or len(self.freq2_power) < 20:
            return False, 0, 0
        
        # Variance at each frequency
        f1_var = np.var(np.diff(self.freq1_power[-20:]))
        f2_var = np.var(np.diff(self.freq2_power[-20:]))
        
        # Baseline variance
        f1_baseline = np.var(np.diff(self.freq1_power[-40:-20])) if len(self.freq1_power) >= 40 else 1e-10
        f2_baseline = np.var(np.diff(self.freq2_power[-40:-20])) if len(self.freq2_power) >= 40 else 1e-10
        
        # Motion scores
        f1_score = f1_var / (f1_baseline + 1e-10)
        f2_score = f2_var / (f2_baseline + 1e-10)
        
        # Cross-frequency correlation
        corr = self.cross_frequency_correlation()
        
        # Combined score (weighted by correlation)
        # High correlation = both frequencies see same motion = more confident
        combined_score = (f1_score + f2_score) / 2 * (0.5 + 0.5 * abs(corr))
        
        motion = combined_score > 1.5
        
        return motion, combined_score, corr
    
    def run(self):
        plt.ion()
        fig = plt.figure("Dual-Frequency Motion Analyzer", figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])   # Power at both frequencies
        ax2 = fig.add_subplot(gs[1, :])   # Phase at both frequencies
        ax3 = fig.add_subplot(gs[2, 0])   # Phase difference
        ax4 = fig.add_subplot(gs[2, 1])   # Relative range change
        ax5 = fig.add_subplot(gs[3, 0])   # Cross-frequency correlation
        ax6 = fig.add_subplot(gs[3, 1])   # Motion detection score
        
        print("Initializing... collecting baseline data...")
        
        motion_confirmed = False
        consecutive_motion = 0
        consecutive_still = 0
        iteration = 0
        
        try:
            while True:
                iteration += 1
                now = datetime.now()
                
                # Read samples at current frequency
                samples = self.sdr.read_samples(16384)  # Smaller for faster hopping
                power_db, mean_phase = self.extract_phase_power(samples)
                
                # Store data for current frequency
                if self.current_freq == 1:
                    self.freq1_power.append(power_db)
                    self.freq1_phase.append(mean_phase)
                else:
                    self.freq2_power.append(power_db)
                    self.freq2_phase.append(mean_phase)
                
                # Calculate phase difference when both frequencies have data
                if len(self.freq1_phase) > 0 and len(self.freq2_phase) > 0:
                    # Use most recent measurements from each frequency
                    phase_diff = self.freq1_phase[-1] - self.freq2_phase[-1]
                    self.phase_diff.append(phase_diff)
                    
                    # Estimate relative range change
                    range_change = self.estimate_relative_range_change()
                    self.relative_range.append(range_change)
                    
                    # Calculate correlation
                    corr = self.cross_frequency_correlation()
                    self.correlation.append(corr)
                
                self.time_history.append(now)
                
                # Trim histories
                max_history = 200
                for hist in [self.freq1_power, self.freq1_phase, 
                           self.freq2_power, self.freq2_phase,
                           self.phase_diff, self.relative_range,
                           self.correlation, self.time_history]:
                    if len(hist) > max_history:
                        hist.pop(0)
                
                # Motion detection
                motion_raw, combined_score, corr = self.detect_motion_dual_freq()
                
                # Confirmation logic - REDUCED LAG
                if motion_raw:
                    consecutive_motion += 1
                    consecutive_still = 0
                    if consecutive_motion >= 2:  # Faster response (was 5)
                        motion_confirmed = True
                else:
                    consecutive_still += 1
                    consecutive_motion = 0
                    if consecutive_still >= 3:  # Faster clearing (was 5)
                        motion_confirmed = False
                
                self.motion_history.append(1 if motion_confirmed else 0)
                if len(self.motion_history) > max_history:
                    self.motion_history.pop(0)
                
                # Console output
                time_str = now.strftime('%H:%M:%S')
                freq_str = f"F{self.current_freq}"
                if motion_confirmed:
                    print(f"[{time_str}] 🚨 MOTION | {freq_str} | Score: {combined_score:.2f} | "
                          f"Corr: {corr:+.2f} | ΔR: {self.relative_range[-1] if self.relative_range else 0:.3f}m")
                elif iteration % 15 == 0:
                    print(f"[{time_str}] ✓ Still | {freq_str} | Score: {combined_score:.2f}")
                
                # ===== PLOTTING =====
                
                # 1. Power at both frequencies
                ax1.clear()
                if len(self.freq1_power) > 0:
                    # Create time arrays for each frequency
                    times_f1 = [self.time_history[i] for i in range(len(self.time_history)) 
                               if i < len(self.freq1_power)]
                    times_f2 = [self.time_history[i] for i in range(len(self.time_history)) 
                               if i < len(self.freq2_power)]
                    
                    ax1.plot(times_f1[-len(self.freq1_power):], self.freq1_power, 
                            'b-', linewidth=1.5, label=f'F1: {self.args.freq1} MHz', alpha=0.7)
                    ax1.plot(times_f2[-len(self.freq2_power):], self.freq2_power,
                            'r-', linewidth=1.5, label=f'F2: {self.args.freq2} MHz', alpha=0.7)
                    
                    # Shade motion
                    motion_idxs = [i for i, m in enumerate(self.motion_history[-len(self.time_history):]) if m]
                    for i in motion_idxs:
                        if i < len(self.time_history) - 1:
                            ax1.axvspan(self.time_history[i], self.time_history[i+1],
                                       alpha=0.15, color='yellow')
                    
                    ax1.set_ylabel('Power (dB)')
                    ax1.set_title('Signal Power at Both Frequencies')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # 2. Phase at both frequencies
                ax2.clear()
                if len(self.freq1_phase) > 0 and len(self.freq2_phase) > 0:
                    times_f1 = self.time_history[-len(self.freq1_phase):]
                    times_f2 = self.time_history[-len(self.freq2_phase):]
                    
                    ax2.plot(times_f1, self.freq1_phase, 'b-', linewidth=1.5, 
                            label=f'Phase F1', alpha=0.7)
                    ax2.plot(times_f2, self.freq2_phase, 'r-', linewidth=1.5,
                            label=f'Phase F2', alpha=0.7)
                    ax2.set_ylabel('Phase (radians)')
                    ax2.set_title('Phase Evolution at Both Frequencies')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # 3. Phase difference
                ax3.clear()
                if len(self.phase_diff) > 0:
                    times_diff = self.time_history[-len(self.phase_diff):]
                    ax3.plot(times_diff, self.phase_diff, 'purple', linewidth=2)
                    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                    ax3.set_ylabel('Δφ (radians)')
                    ax3.set_xlabel('Time')
                    ax3.set_title('Phase Difference (F1 - F2)')
                    ax3.grid(True, alpha=0.3)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # 4. Relative range change
                ax4.clear()
                if len(self.relative_range) > 1:
                    times_range = self.time_history[-len(self.relative_range):]
                    cumulative_range = np.cumsum(self.relative_range)
                    ax4.plot(times_range, cumulative_range, 'green', linewidth=2)
                    ax4.fill_between(times_range, 0, cumulative_range, alpha=0.3, color='green')
                    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                    ax4.set_ylabel('Cumulative ΔR (m)')
                    ax4.set_xlabel('Time')
                    ax4.set_title('Relative Range Change (Not Absolute Distance!)')
                    ax4.grid(True, alpha=0.3)
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # 5. Cross-frequency correlation
                ax5.clear()
                if len(self.correlation) > 0:
                    times_corr = self.time_history[-len(self.correlation):]
                    colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' 
                             for c in self.correlation]
                    ax5.scatter(times_corr, self.correlation, c=colors, s=20)
                    ax5.axhline(y=0.5, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Good')
                    ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                    ax5.set_ylabel('Correlation')
                    ax5.set_xlabel('Time')
                    ax5.set_title('Cross-Frequency Correlation (High = Consistent Motion)')
                    ax5.set_ylim([-1, 1])
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # 6. Motion detection score
                ax6.clear()
                if len(self.freq1_power) >= 40 and len(self.freq2_power) >= 40:
                    scores = []
                    times_score = []
                    min_len = min(len(self.freq1_power), len(self.freq2_power))
                    for i in range(40, min_len):
                        f1_var = np.var(np.diff(self.freq1_power[i-20:i]))
                        f1_base = np.var(np.diff(self.freq1_power[i-40:i-20]))
                        f2_var = np.var(np.diff(self.freq2_power[i-20:i]))
                        f2_base = np.var(np.diff(self.freq2_power[i-40:i-20]))
                        
                        score = ((f1_var / (f1_base + 1e-10)) + (f2_var / (f2_base + 1e-10))) / 2
                        scores.append(score)
                        times_score.append(self.time_history[i])
                    
                    ax6.plot(times_score, scores, 'darkblue', linewidth=2, label='Combined Score')
                    ax6.axhline(y=1.5, color='r', linestyle='--', linewidth=2, label='Threshold')
                    ax6.fill_between(times_score, 0, scores, alpha=0.3, color='blue')
                    ax6.set_xlabel('Time')
                    ax6.set_ylabel('Score')
                    ax6.set_title('Dual-Frequency Motion Score')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)
                    ax6.set_ylim([0, 8])
                    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                if motion_confirmed:
                    fig.suptitle('🚨 MOTION DETECTED', fontsize=16, color='red', weight='bold')
                else:
                    fig.suptitle('Dual-Frequency Passive Radar', fontsize=14)
                
                plt.pause(0.01)
                
                # Hop frequency after plotting
                hopped = self.hop_frequency()
                if hopped:
                    time.sleep(0.05)  # Brief settling time after hop
                
        except KeyboardInterrupt:
            print("\n[STOP] Analyzer stopped")
        finally:
            self.sdr.close()
            plt.ioff()
            plt.show()

if __name__ == "__main__":import argparse
import numpy as np
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import signal
import time

class HighSensitivityMotionDetector:
    """
    HIGH SENSITIVITY Motion Detector - Optimized for Small/Close Objects
    
    Designed to detect:
    - Hand motion near antenna
    - Small objects (< 1m)
    - Subtle movements
    
    Key improvements:
    - Much lower detection threshold
    - Faster response time
    - Enhanced phase sensitivity
    - Real-time sensitivity adjustment
    """
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="High-sensitivity motion detector")
        parser.add_argument("--freq", type=float, default=102.5,
                            help="Frequency in MHz (default 102.5)")
        parser.add_argument("--gain", type=str, default="auto",
                            help="Gain in dB or 'auto' (default auto)")
        parser.add_argument("--sensitivity", type=float, default=1.1,
                            help="Detection threshold (lower=more sensitive, default 1.1)")
        self.args = parser.parse_args()
        
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        self.sdr.center_freq = self.args.freq * 1e6
        self.sdr.gain = self.args.gain
        
        self.wavelength = 3e8 / (self.args.freq * 1e6)
        
        # Detection parameters
        self.threshold = self.args.sensitivity
        
        # History buffers
        self.power_history = []
        self.phase_history = []
        self.phase_diff_history = []  # Instantaneous phase changes
        self.motion_intensity = []
        self.time_history = []
        self.motion_history = []
        
        # Real-time statistics
        self.baseline_power = None
        self.baseline_phase_std = None
        self.samples_collected = 0
        
        print("=" * 70)
        print("HIGH SENSITIVITY MOTION DETECTOR")
        print("=" * 70)
        print(f"Frequency: {self.args.freq} MHz")
        print(f"Wavelength: {self.wavelength:.3f} m")
        print(f"Sample Rate: {self.sdr.sample_rate/1e6} MHz")
        print(f"Sensitivity Threshold: {self.threshold}")
        print()
        print("OPTIMIZED FOR:")
        print("  ✓ Hand motion near antenna (10-50cm)")
        print("  ✓ Small objects")
        print("  ✓ Subtle movements")
        print("  ✓ Fast response (<0.2s)")
        print()
        print("TESTING TIPS:")
        print("  • Wave hand slowly 20-50cm from antenna")
        print("  • Try walking past antenna (1-3m away)")
        print("  • Avoid touching/blocking antenna directly")
        print("  • Wait 5 seconds for baseline calibration")
        print("=" * 70)
        print()
    
    def extract_detailed_metrics(self, samples, prev_samples=None):
        """
        Extract multiple motion-sensitive metrics from I/Q samples.
        """
        # 1. Power
        power = np.mean(np.abs(samples) ** 2)
        power_db = 10 * np.log10(power + 1e-10)
        
        # 2. Phase analysis
        phases = np.angle(samples)
        phases_unwrapped = np.unwrap(phases)
        mean_phase = np.mean(phases_unwrapped)
        phase_std = np.std(np.diff(phases_unwrapped))
        
        # 3. Instantaneous phase change (very sensitive to motion)
        if prev_samples is not None:
            prev_phases = np.unwrap(np.angle(prev_samples))
            
            # Phase velocity (simple difference)
            phase_change = mean_phase - np.mean(prev_phases)
            
            # Standard deviation of phase differences (detects chaos/motion)
            curr_diff = np.diff(phases_unwrapped)
            prev_diff = np.diff(prev_phases)
            phase_diff_change = np.std(curr_diff) - np.std(prev_diff)
        else:
            phase_change = 0
            phase_diff_change = 0
        
        # 4. Amplitude variation (sensitive to scattering)
        amplitude_std = np.std(np.abs(samples))
        
        return {
            'power_db': power_db,
            'mean_phase': mean_phase,
            'phase_std': phase_std,
            'phase_change': phase_change,
            'phase_diff_change': phase_diff_change,
            'amplitude_std': amplitude_std
        }
    
    def adaptive_detection(self, metrics):
        """
        Multi-metric motion detection with adaptive baseline.
        Returns: (motion_detected, confidence_score, trigger_reason)
        """
        if self.samples_collected < 30:
            # Still calibrating
            return False, 0, "Calibrating..."
        
        # Calculate deviations from baseline
        power_deviation = 0
        phase_deviation = 0
        phase_change_score = 0
        
        if self.baseline_power is not None:
            power_deviation = abs(metrics['power_db'] - self.baseline_power)
        
        if self.baseline_phase_std is not None:
            phase_deviation = metrics['phase_std'] / (self.baseline_phase_std + 1e-10)
        
        # Phase change is very sensitive to motion
        phase_change_score = abs(metrics['phase_change']) * 100
        
        # Amplitude variation
        amp_score = metrics['amplitude_std'] * 1000
        
        # Combined score with weights favoring phase metrics
        confidence = (
            power_deviation * 0.2 +
            phase_deviation * 0.3 +
            phase_change_score * 0.4 +
            amp_score * 0.1
        )
        
        # Determine trigger reason
        trigger_reason = "Still"
        if confidence > self.threshold:
            if phase_change_score > 0.3:
                trigger_reason = "Phase Change (High Sensitivity)"
            elif phase_deviation > self.threshold:
                trigger_reason = "Phase Variance"
            elif power_deviation > 1.0:
                trigger_reason = "Power Change"
            else:
                trigger_reason = "Amplitude Variation"
        
        motion = confidence > self.threshold
        
        return motion, confidence, trigger_reason
    
    def update_baseline(self, metrics):
        """
        Continuously update baseline using exponential moving average.
        This allows adaptation to slow environmental changes.
        """
        alpha = 0.05  # Smoothing factor (lower = more stable)
        
        if self.baseline_power is None:
            self.baseline_power = metrics['power_db']
            self.baseline_phase_std = metrics['phase_std']
        else:
            # Only update baseline when no motion is detected
            if len(self.motion_history) > 0 and self.motion_history[-1] == 0:
                self.baseline_power = alpha * metrics['power_db'] + (1 - alpha) * self.baseline_power
                self.baseline_phase_std = alpha * metrics['phase_std'] + (1 - alpha) * self.baseline_phase_std
        
        self.samples_collected += 1
    
    def run(self):
        plt.ion()
        fig = plt.figure("High-Sensitivity Motion Detector", figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])   # Power
        ax2 = fig.add_subplot(gs[1, :])   # Phase
        ax3 = fig.add_subplot(gs[2, 0])   # Phase change (most sensitive)
        ax4 = fig.add_subplot(gs[2, 1])   # Detection confidence
        
        print("🔧 CALIBRATING... Please stay still for 5 seconds!")
        print()
        
        motion_confirmed = False
        prev_samples = None
        iteration = 0
        trigger_reason = "Initializing"
        
        # For real-time display
        last_print_time = time.time()
        
        try:
            while True:
                iteration += 1
                now = datetime.now()
                
                # Read samples
                samples = self.sdr.read_samples(16384)  # Smaller buffer = faster
                
                # Extract metrics
                metrics = self.extract_detailed_metrics(samples, prev_samples)
                
                # Update baseline
                self.update_baseline(metrics)
                
                # Store data
                self.power_history.append(metrics['power_db'])
                self.phase_history.append(metrics['mean_phase'])
                self.phase_diff_history.append(abs(metrics['phase_change']))
                self.time_history.append(now)
                
                # Detect motion
                motion_raw, confidence, reason = self.adaptive_detection(metrics)
                
                # INSTANT detection (no lag!)
                motion_confirmed = motion_raw
                trigger_reason = reason
                
                self.motion_history.append(1 if motion_confirmed else 0)
                self.motion_intensity.append(confidence)
                
                # Trim histories
                max_history = 300
                for hist in [self.power_history, self.phase_history, 
                           self.phase_diff_history, self.time_history,
                           self.motion_history, self.motion_intensity]:
                    if len(hist) > max_history:
                        hist.pop(0)
                
                # Store for next iteration
                prev_samples = samples
                
                # Console output (rate limited for readability)
                current_time = time.time()
                if motion_confirmed or (current_time - last_print_time > 0.5):
                    time_str = now.strftime('%H:%M:%S')
                    if motion_confirmed:
                        print(f"[{time_str}] 🚨 MOTION! | Confidence: {confidence:.2f} | {trigger_reason}")
                    elif self.samples_collected > 30:
                        print(f"[{time_str}] ✓ Still | Confidence: {confidence:.2f} | Threshold: {self.threshold}")
                    last_print_time = current_time
                
                # ===== PLOTTING =====
                
                # 1. Power over time
                ax1.clear()
                ax1.plot(self.time_history, self.power_history, 'b-', linewidth=1.5, label='Signal Power')
                
                if self.baseline_power is not None:
                    ax1.axhline(y=self.baseline_power, color='green', linestyle='--', 
                               linewidth=1, alpha=0.5, label='Baseline')
                
                # Shade motion periods
                for i in range(len(self.motion_history)):
                    if self.motion_history[i] == 1 and i < len(self.time_history) - 1:
                        ax1.axvspan(self.time_history[i], self.time_history[i+1],
                                   alpha=0.3, color='red')
                
                ax1.set_ylabel('Power (dB)')
                ax1.set_title('Signal Power (Red = Motion)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                if motion_confirmed:
                    ax1.text(0.5, 0.95, f'🚨 MOTION: {trigger_reason}', 
                            transform=ax1.transAxes, fontsize=14, color='red',
                            weight='bold', ha='center', va='top',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
                
                # 2. Phase over time
                ax2.clear()
                ax2.plot(self.time_history, self.phase_history, 'purple', linewidth=1.5)
                
                # Shade motion
                for i in range(len(self.motion_history)):
                    if self.motion_history[i] == 1 and i < len(self.time_history) - 1:
                        ax2.axvspan(self.time_history[i], self.time_history[i+1],
                                   alpha=0.3, color='red')
                
                ax2.set_ylabel('Phase (radians)')
                ax2.set_title('Phase Evolution (Highly Sensitive to Motion)')
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # 3. Phase change rate (MOST SENSITIVE)
                ax3.clear()
                if len(self.phase_diff_history) > 1:
                    # Scale for visibility
                    scaled_changes = [p * 100 for p in self.phase_diff_history]
                    ax3.plot(self.time_history[-len(scaled_changes):], scaled_changes,
                            'red', linewidth=2, label='Phase Change Rate')
                    ax3.fill_between(self.time_history[-len(scaled_changes):], 0, scaled_changes,
                                    alpha=0.3, color='red')
                    
                    # Show sensitivity threshold
                    ax3.axhline(y=0.3, color='orange', linestyle='--', linewidth=2,
                               alpha=0.7, label='Sensitivity Threshold')
                    
                    ax3.set_ylabel('Phase Change (scaled)')
                    ax3.set_xlabel('Time')
                    ax3.set_title('Instantaneous Phase Change (Most Sensitive!)')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                # 4. Detection confidence score
                ax4.clear()
                if len(self.motion_intensity) > 1:
                    colors = ['red' if m == 1 else 'green' 
                             for m in self.motion_history[-len(self.motion_intensity):]]
                    ax4.scatter(self.time_history[-len(self.motion_intensity):], 
                               self.motion_intensity, c=colors, s=20, alpha=0.6)
                    
                    ax4.axhline(y=self.threshold, color='orange', linestyle='--',
                               linewidth=2, label=f'Threshold: {self.threshold}')
                    
                    ax4.set_ylabel('Confidence Score')
                    ax4.set_xlabel('Time')
                    ax4.set_title('Motion Confidence (Red = Detected, Green = Still)')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    ax4.set_ylim([0, max(5, max(self.motion_intensity) * 1.2) if self.motion_intensity else 5])
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                
                plt.pause(0.01)
                time.sleep(0.05)  # Very fast update
                
        except KeyboardInterrupt:
            print("\n[STOP] Detector stopped")
            print(f"\nStatistics:")
            print(f"  Total samples: {self.samples_collected}")
            print(f"  Baseline power: {self.baseline_power:.2f} dB")
            print(f"  Motion events: {sum(self.motion_history)}")
        finally:
            self.sdr.close()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    detector = HighSensitivityMotionDetector()
    detector.run()
    detector = DualFrequencyMotionAnalyzer()
    detector.run()