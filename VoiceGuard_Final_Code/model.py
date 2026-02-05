import numpy as np
import librosa
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceClassifier:
    def __init__(self):
        # Enhanced thresholds (calibrated for demonstration)
        self.PITCH_STD_THRESHOLD = 20.0  # Hz
        self.JITTER_THRESHOLD = 0.03  # 3% - AI voices often have perfect timing
        self.SHIMMER_THRESHOLD = 0.05  # 5% - AI voices have consistent amplitude
        self.HNR_THRESHOLD = 15.0  # dB - Low HNR suggests noise/artifacts
        self.SILENCE_RATIO_THRESHOLD = 0.05
        
    def preemphasis(self, signal, coeff=0.97):
        """
        Applies a pre-emphasis filter to the signal.
        This balances the spectrum by boosting high frequencies, 
        helping to reduce the effect of low-frequency noise.
        """
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    def calculate_jitter(self, y, sr, f0):
        """
        Calculate jitter (pitch period variation).
        Jitter measures micro-variations in the fundamental frequency.
        AI voices often have unnaturally low jitter.
        """
        try:
            # Filter out non-voiced frames
            f0_voiced = f0[f0 > 0]
            
            if len(f0_voiced) < 2:
                return 0.0
            
            # Calculate period from frequency
            periods = 1.0 / f0_voiced
            
            # Jitter is the average absolute difference between consecutive periods
            period_diffs = np.abs(np.diff(periods))
            jitter = np.mean(period_diffs) / np.mean(periods) if np.mean(periods) > 0 else 0
            
            return float(jitter)
        except:
            return 0.0

    def calculate_shimmer(self, y, sr):
        """
        Calculate shimmer (amplitude variation).
        Shimmer measures variations in amplitude between periods.
        AI voices often have very consistent amplitude.
        """
        try:
            # Use RMS energy as amplitude proxy
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            if len(rms) < 2:
                return 0.0
            
            # Shimmer is the average absolute difference in amplitude
            rms_diffs = np.abs(np.diff(rms))
            shimmer = np.mean(rms_diffs) / np.mean(rms) if np.mean(rms) > 0 else 0
            
            return float(shimmer)
        except:
            return 0.0

    def calculate_hnr(self, y, sr):
        """
        Calculate Harmonic-to-Noise Ratio.
        High HNR indicates clear harmonic structure (typical in natural voice).
        Low HNR might indicate artifacts or processing.
        """
        try:
            # Use autocorrelation to estimate HNR
            autocorr = librosa.autocorrelate(y)
            
            # Find first peak (fundamental period)
            peaks = []
            for i in range(1, min(len(autocorr), sr // 50)):  # Search up to 50Hz
                if i > 0 and i < len(autocorr) - 1:
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        peaks.append((i, autocorr[i]))
            
            if not peaks:
                return 10.0  # Default moderate value
            
            # Get strongest peak
            peak_idx, peak_val = max(peaks, key=lambda x: x[1])
            
            # HNR estimate: ratio of signal to noise
            signal_power = peak_val
            noise_power = np.mean(autocorr[peak_idx+1:peak_idx+50]) if peak_idx+50 < len(autocorr) else np.mean(autocorr[peak_idx+1:])
            
            if noise_power > 0:
                hnr_db = 10 * np.log10(signal_power / noise_power)
                return float(np.clip(hnr_db, 0, 40))  # Clip to reasonable range
            
            return 10.0
        except:
            return 10.0

    def extract_features(self, y, sr):
        """Extracts comprehensive acoustic features from the audio time series."""
        
        # 0. Noise Resilience: Apply Pre-emphasis
        y_proc = self.preemphasis(y)

        # 1. Pitch (Fundamental Frequency - F0) - OPTIMIZED with pyin
        # pyin is faster and more accurate than piptrack for voice
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y_proc,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=2048
            )
            
            pitch_values = f0[voiced_flag]
            
            if len(pitch_values) == 0:
                pitch_std = 0
                pitch_mean = 0
            else:
                pitch_std = np.std(pitch_values)
                pitch_mean = np.mean(pitch_values)
                
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            pitch_std = 0
            pitch_mean = 0
            f0 = np.array([])
            
        # 2. Jitter (NEW)
        jitter = self.calculate_jitter(y, sr, f0 if len(f0) > 0 else np.array([]))
        
        # 3. Shimmer (NEW)
        shimmer = self.calculate_shimmer(y, sr)
        
        # 4. Harmonic-to-Noise Ratio (NEW)
        hnr = self.calculate_hnr(y, sr)
        
        # 5. Spectral Flatness
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # 6. Silence/Pause analysis
        rms = librosa.feature.rms(y=y)[0]
        silence_frames = np.sum(rms < 0.01)
        total_frames = len(rms)
        silence_ratio = silence_frames / total_frames if total_frames > 0 else 0
        
        return {
            "pitch_std": float(pitch_std),
            "pitch_mean": float(pitch_mean),
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
            "spectral_flatness": float(flatness),
            "silence_ratio": float(silence_ratio)
        }

    def predict(self, y, sr):
        """
        Classifies audio as AI_GENERATED or HUMAN based on advanced features.
        Returns: classification, confidence, explanation
        """
        features = self.extract_features(y, sr)
        
        score = 0
        explanations = []
        feature_scores = {}
        
        # --- Enhanced Heuristic Logic ---
        
        # 1. Pitch Consistency Check
        if features["pitch_std"] < self.PITCH_STD_THRESHOLD:
            contribution = 0.25
            score += contribution
            feature_scores['pitch'] = contribution
            explanations.append(f"Monotone pitch pattern (std: {features['pitch_std']:.1f}Hz)")
        else:
            score -= 0.15
            explanations.append(f"Natural pitch variation (std: {features['pitch_std']:.1f}Hz)")
        
        # 2. Jitter Check (NEW)
        if features["jitter"] < self.JITTER_THRESHOLD:
            contribution = 0.25
            score += contribution
            feature_scores['jitter'] = contribution
            explanations.append(f"Perfect pitch timing (jitter: {features['jitter']:.3f})")
        else:
            score -= 0.15
            
        # 3. Shimmer Check (NEW)
        if features["shimmer"] < self.SHIMMER_THRESHOLD:
            contribution = 0.2
            score += contribution
            feature_scores['shimmer'] = contribution
            explanations.append(f"Uniform amplitude (shimmer: {features['shimmer']:.3f})")
        else:
            score -= 0.1
            
        # 4. HNR Check (NEW)
        if features["hnr"] > 20.0:  # Unusually high HNR can indicate synthetic clarity
            contribution = 0.15
            score += contribution
            feature_scores['hnr'] = contribution
            explanations.append(f"Synthetic clarity (HNR: {features['hnr']:.1f}dB)")
        elif features["hnr"] < 10.0:  # Low HNR might indicate artifacts
            score += 0.1
            explanations.append(f"Unusual noise pattern (HNR: {features['hnr']:.1f}dB)")
            
        # 5. Spectral Flatness
        if features["spectral_flatness"] < 0.01:
            score += 0.1
            explanations.append("Overly tonal spectrum")
            
        # 6. Silence Ratio
        if features["silence_ratio"] < 0.02:
            score += 0.05
            explanations.append("Lack of natural pauses")
            
        # Normalize score to probability
        ai_probability = 0.5 + (score * 0.8)  # Base 50%, adjust by weighted features
        ai_probability = max(0.01, min(0.99, ai_probability))
        
        # Generate detailed explanation
        if ai_probability > 0.5:
            classification = "AI_GENERATED"
            confidence = ai_probability
            
            # Build detailed explanation
            if len(explanations) > 0:
                main_explanation = " | ".join(explanations[:3])  # Top 3 indicators
            else:
                main_explanation = "Multiple synthetic audio patterns detected"
        else:
            classification = "HUMAN"
            confidence = 1.0 - ai_probability
            
            # Highlight human characteristics
            human_traits = []
            if features["pitch_std"] >= self.PITCH_STD_THRESHOLD:
                human_traits.append("dynamic intonation")
            if features["jitter"] >= self.JITTER_THRESHOLD:
                human_traits.append("natural pitch variation")
            if features["shimmer"] >= self.SHIMMER_THRESHOLD:
                human_traits.append("organic amplitude changes")
                
            if human_traits:
                main_explanation = f"Natural speech characteristics: {', '.join(human_traits)}"
            else:
                main_explanation = "Natural pitch variation and prosody detected"
            
        # Log classification for monitoring
        logger.info(f"Classification: {classification} | Confidence: {confidence} | Features: {feature_scores}")
        
        return classification, round(confidence, 2), main_explanation

# Global instance
classifier = VoiceClassifier()
