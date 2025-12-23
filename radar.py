import argparse
import numpy as np
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import time
class ImprovedMotionDetector:
    def __init__(self):
        parser = argparse.ArgumentParser(description="RF motion detector")
        parser.add_argument("--freq", type=float, default=103.7,
                            help="Center frequency in MHz (default 103.7)")
        parser.add_argument("--gain", type=str, default="auto",
                            help="Gain in dB or 'auto' (default auto)")
        self.args = parser.parse_args()
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        self.sdr.center_freq = self.args.freq * 1e6
        # Use auto gain for maximum sensitivity (especially with RTL‑SDR v4)
        self.sdr.gain = self.args.gain
        self.power_history = []          # raw power dB values
        self.smoothed_history = []       # moving‑average of last 5 values
        self.motion_history = []         # 1 = motion, 0 = still
        self.time_history = []
        print("IMPROVED MOTION DETECTOR")
        print("With false positive reduction and real-time display")
        print()
    def _smooth(self, values, window=5):
        """Return simple moving‑average of the last *window* samples.
        If fewer than *window* samples are available, return the mean of what we have.
        """
        if len(values) == 0:
            return 0.0
        if len(values) < window:
            return float(np.mean(values))
        return float(np.mean(values[-window:]))
    def detect_motion_improved(self):
        """Improved detection with temporal filtering.
        Returns (motion_bool, motion_score, baseline_variance).
        """
        if len(self.smoothed_history) < 40:
            return False, 0, 0
        # Use the smoothed power values for variance calculations
        recent = self.smoothed_history[-20:]      # last 2 s (approx.)
        baseline = self.smoothed_history[-40:-20]  # previous 2 s
        recent_variance = np.var(recent)
        baseline_variance = np.var(baseline)
        recent_diff = np.abs(np.diff(recent))
        baseline_diff = np.abs(np.diff(baseline))
        motion_score = (recent_variance / (baseline_variance + 1e-10)) * \
                      (np.mean(recent_diff) / (np.mean(baseline_diff) + 1e-10))
        # Lowered threshold for higher sensitivity (tune if needed)
        motion = motion_score > 1.5
        return motion, motion_score, baseline_variance
    def run(self):
        plt.ion()
        fig = plt.figure("RTLSDR simple Radar detector",figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])
        print("Collecting baseline... stay still for 5 seconds!")
        motion_confirmed = False
        motion_start_time = None
        consecutive_motion = 0
        consecutive_still = 0
        iteration = 0
        try:
            while True:
                iteration += 1
                now = datetime.now()
                samples = self.sdr.read_samples(32768)
                power = np.mean(np.abs(samples) ** 2)
                power_db = 10 * np.log10(power)
                self.power_history.append(power_db)
                # Store smoothed version for detection
                self.smoothed_history.append(self._smooth(self.power_history))
                self.time_history.append(now)
                # Trim histories to keep memory bounded
                if len(self.power_history) > 200:
                    self.power_history.pop(0)
                    self.smoothed_history.pop(0)
                    self.time_history.pop(0)
                motion_raw, motion_score, baseline_var = self.detect_motion_improved()
                if motion_raw:
                    consecutive_motion += 1
                    consecutive_still = 0
                    if consecutive_motion >= 3:
                        motion_confirmed = True
                        if motion_start_time is None:
                            motion_start_time = datetime.now()
                else:
                    consecutive_still += 1
                    consecutive_motion = 0
                    if consecutive_still >= 5:
                        motion_confirmed = False
                        if motion_start_time:
                            dur = (datetime.now() - motion_start_time).total_seconds()
                            print(f"   → Motion lasted {dur:.1f} seconds")
                            motion_start_time = None
                self.motion_history.append(1 if motion_confirmed else 0)
                if len(self.motion_history) > 200:
                    self.motion_history.pop(0)
                # ---- UI ----
                time_str = now.strftime('%H:%M:%S')
                if motion_confirmed:
                    print(f"[{time_str}] 🚨 MOTION (score: {motion_score:.2f}, consecutive: {consecutive_motion})")
                elif motion_raw:
                    print(f"[{time_str}] ⚠️  Possible motion (confirming... {consecutive_motion}/3)")
                else:
                    if iteration % 10 == 0:
                        print(f"[{time_str}] ✓ Still (score: {motion_score:.2f})")
                # Plot power over time
                ax1.clear()
                ax1.plot(self.time_history, self.power_history, 'b-', linewidth=1.5, label='Signal Power')
                # Shade motion periods
                motion_idxs = [i for i, m in enumerate(self.motion_history[-len(self.power_history):]) if m]
                for i in motion_idxs:
                    if i < len(self.time_history) - 1:
                        ax1.axvspan(self.time_history[i], self.time_history[i+1], alpha=0.2, color='red')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Power (dB)')
                ax1.set_title('Signal Strength Over Time (Red = Motion)')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate()
                if motion_confirmed:
                    ax1.text(0.5, 0.95, '🚨 MOTION DETECTED', transform=ax1.transAxes,
                             fontsize=16, color='red', weight='bold', ha='center', va='top',
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
                # Variance plot
                ax2.clear()
                if len(self.smoothed_history) >= 40:
                    recent_var = np.var(self.smoothed_history[-20:])
                    ax2.bar(['Baseline', 'Recent'], [baseline_var, recent_var],
                            color=['green', 'red' if motion_confirmed else 'blue'])
                    ax2.set_ylabel('Variance')
                    ax2.set_title('Signal Variance')
                    ax2.grid(True, axis='y', alpha=0.3)
                # Motion score plot
                ax3.clear()
                if len(self.smoothed_history) >= 40:
                    scores = []
                    times = []
                    for i in range(40, len(self.smoothed_history)):
                        recent = self.smoothed_history[i-20:i]
                        baseline = self.smoothed_history[i-40:i-20]
                        rv = np.var(recent)
                        bv = np.var(baseline)
                        scores.append(rv / (bv + 1e-10))
                        times.append(self.time_history[i])
                    ax3.plot(times, scores, 'g-', linewidth=2)
                    ax3.axhline(y=1.5, color='r', linestyle='--', linewidth=2, label='Threshold')
                    ax3.set_xlabel('Time')
                    ax3.set_ylabel('Motion Score')
                    ax3.set_title('Motion Detection Score')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_ylim([0, 10])
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
                # Spectrum plot
                ax4.clear()
                spectrum = np.abs(np.fft.fft(samples))[:len(samples)//2]
                freqs = np.fft.fftfreq(len(samples), 1/self.sdr.sample_rate)[:len(samples)//2]
                ax4.plot(freqs/1e3, 20*np.log10(spectrum + 1e-10), linewidth=0.5)
                ax4.set_ylabel('Power (dB)')
                ax4.set_xlabel('Frequency Offset (kHz)')
                ax4.set_title('Current Spectrum')
                ax4.grid(True, alpha=0.3)
                plt.pause(0.01)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[STOP] Detector stopped")
        finally:
            self.sdr.close()
            plt.ioff()
            plt.show()
if __name__ == "__main__":
    detector = ImprovedMotionDetector()
    detector.run()