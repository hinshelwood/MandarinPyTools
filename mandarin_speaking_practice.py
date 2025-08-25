import sys
import os
import pandas as pd
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from pypinyin import pinyin, Style
import tempfile
import threading
import queue
from difflib import SequenceMatcher
import torch
import unicodedata
import opencc  # For traditional/simplified conversion
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                            QTextEdit, QFileDialog, QMessageBox, QProgressBar,
                            QGroupBox, QSizePolicy, QSpacerItem, QLineEdit,
                            QComboBox, QRadioButton, QButtonGroup, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPainter, QPen
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl

# Add your Spark-TTS path
sys.path.append(os.path.expanduser("~/software/Spark-TTS/"))
sys.path.append(os.path.expanduser("~/software/SenseVoice/"))
from cli.SparkTTS import SparkTTS

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    SENSEVOICE_AVAILABLE = True
except ImportError:
    SENSEVOICE_AVAILABLE = False
    print("SenseVoice not available - falling back to Whisper only")

class AudioLevelWidget(QWidget):
    """Custom widget to display audio level meter"""
    def __init__(self):
        super().__init__()
        self.level = 0.0
        self.setMinimumSize(200, 30)
        self.setMaximumSize(200, 30)
    
    def set_level(self, level):
        """Update the audio level (0.0 to 1.0)"""
        self.level = max(0.0, min(1.0, level))
        self.update()
    
    def paintEvent(self, event):
        """Draw the level meter"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), Qt.GlobalColor.lightGray)
        
        # Level bar
        level_width = int(self.width() * self.level)
        level_rect = self.rect()
        level_rect.setWidth(level_width)
        
        # Color based on level
        if self.level < 0.3:
            color = Qt.GlobalColor.green
        elif self.level < 0.7:
            color = Qt.GlobalColor.yellow
        else:
            color = Qt.GlobalColor.red
        
        painter.fillRect(level_rect, color)
        
        # Border
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawRect(self.rect())

class ModelLoader(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def run(self):
        try:
            self.progress.emit("Loading Whisper model...")
            whisper_model = whisper.load_model("base")
            
            self.progress.emit("Loading SparkTTS model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_dir = "/home/richard/software/Spark-TTS/pretrained_models/Spark-TTS-0.5B/"
            spark_tts_model = SparkTTS(model_dir, device)
            
            # Load SenseVoice if available
            sensevoice_model = None
            if SENSEVOICE_AVAILABLE:
                try:
                    self.progress.emit("Loading SenseVoice model...")
                    sensevoice_model = AutoModel(
                        model="iic/SenseVoiceSmall",  # Use ModelScope identifier instead of local path
                        trust_remote_code=True,
                        remote_code="/home/richard/software/SenseVoice/model.py",
                        vad_model="fsmn-vad",
                        vad_kwargs={"max_single_segment_time": 30000},
                        device="cuda:0" if torch.cuda.is_available() else "cpu",
                        disable_update=True,  # Skip update checks for faster loading
                    )
                except Exception as e:
                    print(f"Failed to load SenseVoice: {e}")
                    sensevoice_model = None
            
            self.finished.emit(True, "Models loaded successfully!")
            # Store models in thread for main thread to access
            self.whisper_model = whisper_model
            self.spark_tts_model = spark_tts_model
            self.sensevoice_model = sensevoice_model
            
        except Exception as e:
            self.finished.emit(False, f"Error loading models: {str(e)}")

class AudioGenerator(QThread):
    finished = pyqtSignal(bool, str, str)  # success, message, audio_path
    progress = pyqtSignal(str)
    
    def __init__(self, spark_tts_model, text, speed='low'):
        super().__init__()
        self.spark_tts_model = spark_tts_model
        self.text = text
        self.speed = speed
    
    def run(self):
        try:
            self.progress.emit("Generating audio...")
            
            with torch.no_grad():
                wav = self.spark_tts_model.inference(
                    self.text,
                    "/home/richard/software/Spark-TTS/example/primsluer1.flac",
                    prompt_text=None,
                    gender='male',
                    pitch='moderate',
                    speed=self.speed
                )
            
            # Save to temporary file with proper sample rate
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_audio_path, wav, 16000)  # Updated sample rate
            
            self.finished.emit(True, "Audio generated successfully!", temp_audio_path)
            
        except Exception as e:
            self.finished.emit(False, f"Error generating audio: {str(e)}", "")

class AudioTranscriber(QThread):
    finished = pyqtSignal(bool, str, str, str)  # success, message, transcription, pinyin
    progress = pyqtSignal(str)
    
    def __init__(self, whisper_model, sensevoice_model, audio_data, sample_rate, selected_model):
        super().__init__()
        self.whisper_model = whisper_model
        self.sensevoice_model = sensevoice_model
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.selected_model = selected_model
    
    def preprocess_audio(self, audio_data, sample_rate):
        """Preprocess audio for better transcription"""
        # Normalize audio levels
        if len(audio_data) > 0:
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normalize to prevent clipping while maintaining dynamics
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Scale to 90% of maximum to avoid clipping
                audio_data = audio_data * (0.9 / max_val)
            
            # Apply simple noise gate (remove very quiet parts)
            noise_threshold = np.max(np.abs(audio_data)) * 0.01  # 1% of max volume
            audio_data[np.abs(audio_data) < noise_threshold] *= 0.1
            
        return audio_data
    
    def transcribe_with_sensevoice(self, audio_path):
        """Transcribe using SenseVoice model"""
        try:
            res = self.sensevoice_model.generate(
                input=audio_path,
                cache={},
                language="zh",  # Focus on Chinese
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            # Post-process the result
            if SENSEVOICE_AVAILABLE:
                transcribed_text = rich_transcription_postprocess(res[0]["text"])
            else:
                transcribed_text = res[0]["text"]
            
            return transcribed_text.strip()
            
        except Exception as e:
            raise Exception(f"SenseVoice transcription failed: {str(e)}")
    
    def transcribe_with_whisper(self, audio_path):
        """Transcribe using Whisper with multiple attempts"""
        attempts = [
            {"language": "zh", "task": "transcribe", "temperature": 0.0},
            {"task": "transcribe", "temperature": 0.0},
            {"language": "zh", "task": "transcribe", "temperature": 0.2},
            {"task": "transcribe", "temperature": 0.5}
        ]
        
        for i, params in enumerate(attempts):
            try:
                result = self.whisper_model.transcribe(audio_path, **params)
                candidate_text = result["text"].strip()
                
                # Basic quality check - prefer results with Chinese characters
                if candidate_text and any('\u4e00' <= char <= '\u9fff' for char in candidate_text):
                    return candidate_text
                elif candidate_text and i == len(attempts) - 1:
                    return candidate_text  # Last attempt fallback
                    
            except Exception as e:
                if i == len(attempts) - 1:  # Last attempt
                    raise e
                continue
        
        raise Exception("All Whisper transcription attempts failed")
    
    def run(self):
        try:
            self.progress.emit("Preparing audio for transcription...")
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(self.audio_data, self.sample_rate)
            
            # Save processed audio to temporary file
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_audio_path, processed_audio, self.sample_rate)
            
            # Transcribe based on selected model
            if self.selected_model == "sensevoice" and self.sensevoice_model is not None:
                self.progress.emit("Transcribing with SenseVoice...")
                transcribed_text = self.transcribe_with_sensevoice(temp_audio_path)
                print(f"SenseVoice transcription: {transcribed_text}")
            else:
                self.progress.emit("Transcribing with Whisper...")
                transcribed_text = self.transcribe_with_whisper(temp_audio_path)
                print(f"Whisper transcription: {transcribed_text}")
            
            if not transcribed_text:
                raise Exception("No transcription result obtained")
            
            # Generate pinyin
            pinyin_result = pinyin(transcribed_text, style=Style.TONE)
            pinyin_text = ' '.join([''.join(p) for p in pinyin_result])
            
            # Clean up
            os.unlink(temp_audio_path)
            
            self.finished.emit(True, f"Transcription complete ({self.selected_model})!", transcribed_text, pinyin_text)
            
        except Exception as e:
            self.finished.emit(False, f"Error transcribing with {self.selected_model}: {str(e)}", "", "")

class MandarinStudyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandarin Speaking Practice Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize variables
        self.whisper_model = None
        self.spark_tts_model = None
        self.sensevoice_model = None
        self.df = None
        self.current_index = 0
        self.is_recording = False
        self.recorded_audio = []
        self.temp_audio_path = None
        self.sample_rate = 16000
        self.selected_device = None
        self.device_sample_rates = {}  # Store supported sample rates per device
        self.csv_file_path = None  # Store CSV path for saving
        self.selected_asr_model = "whisper"  # Default ASR model
        self.current_speed = "low"  # Default TTS speed
        
        # Speed mapping for SparkTTS
        self.speed_options = ['very_low', 'low', 'moderate', 'high', 'very_high']
        
        # Chinese text normalization
        try:
            self.converter = opencc.OpenCC('t2s')  # Traditional to Simplified
        except:
            self.converter = None
            print("Warning: OpenCC not available, character normalization limited")
        
        # Audio level monitoring
        self.audio_level_timer = QTimer()
        self.audio_level_timer.timeout.connect(self.update_audio_level)
        self.current_audio_level = 0.0
        
        # Audio player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Setup UI
        self.setup_ui()
        self.apply_styles()
        
        # Load models
        self.load_models()
        
        # Start audio level monitoring
        self.start_audio_monitoring()
    
    def setup_ui(self):
        """Setup the main UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # File loading section
        file_group = QGroupBox("Load Study Data")
        file_layout = QHBoxLayout(file_group)
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select CSV file with columns: zh, eng (pinyin auto-generated)")
        file_layout.addWidget(self.file_path_edit)
        
        browse_btn = QPushButton("Browse CSV")
        browse_btn.clicked.connect(self.load_csv)
        file_layout.addWidget(browse_btn)
        
        main_layout.addWidget(file_group)
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left panel - Prompt
        left_group = QGroupBox("Prompt")
        left_layout = QVBoxLayout(left_group)
        left_group.setMinimumWidth(500)  # Increased from 450
        
        # Chinese text
        chinese_label = QLabel("Chinese:")
        chinese_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        left_layout.addWidget(chinese_label)
        
        self.chinese_text = QTextEdit()
        self.chinese_text.setFont(QFont("Noto Sans CJK SC", 16))  # Better Chinese font
        self.chinese_text.setMinimumHeight(100)
        self.chinese_text.setMaximumHeight(150)
        self.chinese_text.setReadOnly(True)
        left_layout.addWidget(self.chinese_text)
        
        # Pinyin
        pinyin_label = QLabel("Pinyin:")
        pinyin_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        left_layout.addWidget(pinyin_label)
        
        self.pinyin_text = QTextEdit()
        self.pinyin_text.setFont(QFont("Consolas", 14))  # Monospace for pinyin
        self.pinyin_text.setMinimumHeight(80)
        self.pinyin_text.setMaximumHeight(120)
        self.pinyin_text.setReadOnly(True)
        left_layout.addWidget(self.pinyin_text)
        
        # English
        english_label = QLabel("English:")
        english_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        left_layout.addWidget(english_label)
        
        self.english_text = QTextEdit()
        self.english_text.setFont(QFont("Arial", 12))
        self.english_text.setMinimumHeight(100)
        self.english_text.setMaximumHeight(150)
        self.english_text.setReadOnly(True)
        left_layout.addWidget(self.english_text)
        
        # Audio controls
        audio_layout = QHBoxLayout()
        
        self.play_button = QPushButton("🔊 Play Audio")
        self.play_button.clicked.connect(self.play_prompt_audio)
        audio_layout.addWidget(self.play_button)
        
        self.generate_audio_button = QPushButton("Generate Audio")
        self.generate_audio_button.clicked.connect(self.generate_audio)
        audio_layout.addWidget(self.generate_audio_button)
        
        left_layout.addLayout(audio_layout)
        
        # Speed control
        speed_layout = QVBoxLayout()
        speed_label = QLabel("Speech Speed:")
        speed_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        speed_layout.addWidget(speed_label)
        
        # Speed slider
        speed_slider_layout = QHBoxLayout()
        
        slow_label = QLabel("Slow")
        slow_label.setFont(QFont("Arial", 9))
        speed_slider_layout.addWidget(slow_label)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(4)
        self.speed_slider.setValue(1)  # Default to 'low' (index 1)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_slider_layout.addWidget(self.speed_slider)
        
        fast_label = QLabel("Fast")
        fast_label.setFont(QFont("Arial", 9))
        speed_slider_layout.addWidget(fast_label)
        
        speed_layout.addLayout(speed_slider_layout)
        
        # Speed value display
        self.speed_value_label = QLabel("Speed: low")
        self.speed_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speed_value_label.setFont(QFont("Arial", 10))
        speed_layout.addWidget(self.speed_value_label)
        
        left_layout.addLayout(speed_layout)
        
        # Add stretch to push everything to top
        left_layout.addStretch()
        
        content_layout.addWidget(left_group)
        
        # Right panel - Response
        right_group = QGroupBox("Your Response")
        right_layout = QVBoxLayout(right_group)
        right_group.setMinimumWidth(500)  # Increased from 450
        
        # Recording controls
        record_layout = QHBoxLayout()
        
        self.record_button = QPushButton("🎤 Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        record_layout.addWidget(self.record_button)
        
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.transcribe_audio)
        self.transcribe_button.setEnabled(False)
        record_layout.addWidget(self.transcribe_button)
        
        right_layout.addLayout(record_layout)
        
        # Playback controls for recorded audio
        playback_layout = QHBoxLayout()
        
        self.playback_button = QPushButton("🔊 Play Recording")
        self.playback_button.clicked.connect(self.play_recorded_audio)
        self.playback_button.setEnabled(False)
        playback_layout.addWidget(self.playback_button)
        
        right_layout.addLayout(playback_layout)
        
        # Microphone selection and level
        mic_layout = QVBoxLayout()
        
        # Microphone selector
        mic_select_layout = QHBoxLayout()
        mic_label = QLabel("Microphone:")
        mic_label.setFont(QFont("Arial", 10))
        mic_select_layout.addWidget(mic_label)
        
        self.mic_combo = QComboBox()
        self.populate_audio_devices()
        self.mic_combo.currentTextChanged.connect(self.on_mic_changed)
        mic_select_layout.addWidget(self.mic_combo)
        
        mic_layout.addLayout(mic_select_layout)
        
        # Audio level meter
        level_layout = QHBoxLayout()
        level_label = QLabel("Level:")
        level_label.setFont(QFont("Arial", 10))
        level_layout.addWidget(level_label)
        
        self.audio_level_widget = AudioLevelWidget()
        level_layout.addWidget(self.audio_level_widget)
        
        mic_layout.addLayout(level_layout)
        
        right_layout.addLayout(mic_layout)
        
        # Response text
        response_label = QLabel("Your Chinese:")
        response_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        right_layout.addWidget(response_label)
        
        self.response_text = QTextEdit()
        self.response_text.setFont(QFont("Noto Sans CJK SC", 16))
        self.response_text.setMinimumHeight(100)
        self.response_text.setMaximumHeight(150)
        right_layout.addWidget(self.response_text)
        
        # Response pinyin
        response_pinyin_label = QLabel("Your Pinyin:")
        response_pinyin_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        right_layout.addWidget(response_pinyin_label)
        
        self.response_pinyin = QTextEdit()
        self.response_pinyin.setFont(QFont("Consolas", 14))
        self.response_pinyin.setMinimumHeight(80)
        self.response_pinyin.setMaximumHeight(120)
        right_layout.addWidget(self.response_pinyin)
        
        # Add stretch
        right_layout.addStretch()
        
        content_layout.addWidget(right_group)
        
        # Controls panel
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_group.setMaximumWidth(300)
        
        # ASR Model Selection
        asr_group = QGroupBox("ASR Model")
        asr_layout = QVBoxLayout(asr_group)
        
        self.model_button_group = QButtonGroup()
        self.whisper_radio = QRadioButton("Whisper")
        self.whisper_radio.setChecked(True)
        self.whisper_radio.toggled.connect(lambda: self.set_asr_model("whisper"))
        asr_layout.addWidget(self.whisper_radio)
        self.model_button_group.addButton(self.whisper_radio)
        
        self.sensevoice_radio = QRadioButton("SenseVoice")
        self.sensevoice_radio.toggled.connect(lambda: self.set_asr_model("sensevoice"))
        asr_layout.addWidget(self.sensevoice_radio)
        self.model_button_group.addButton(self.sensevoice_radio)
        
        # Disable SenseVoice if not available
        if not SENSEVOICE_AVAILABLE:
            self.sensevoice_radio.setEnabled(False)
            self.sensevoice_radio.setText("SenseVoice (Not Available)")
        
        controls_layout.addWidget(asr_group)
        
        # Navigation
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(self.previous_sentence)
        nav_layout.addWidget(prev_btn)
        
        self.sentence_label = QLabel("0 / 0")
        self.sentence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sentence_label.setFont(QFont("Arial", 12))
        nav_layout.addWidget(self.sentence_label)
        
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.next_sentence)
        nav_layout.addWidget(next_btn)
        
        controls_layout.addLayout(nav_layout)
        
        # Score display
        score_group = QGroupBox("Accuracy Score")
        score_layout = QVBoxLayout(score_group)
        
        self.score_label = QLabel("No score yet")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        score_layout.addWidget(self.score_label)
        
        # Previous score and improvement display
        self.prev_score_label = QLabel("")
        self.prev_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_score_label.setFont(QFont("Arial", 10))
        score_layout.addWidget(self.prev_score_label)
        
        # Running average display
        self.running_avg_label = QLabel("")
        self.running_avg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.running_avg_label.setFont(QFont("Arial", 10))
        score_layout.addWidget(self.running_avg_label)
        
        calc_score_btn = QPushButton("Calculate Score")
        calc_score_btn.clicked.connect(self.calculate_score)
        score_layout.addWidget(calc_score_btn)
        
        controls_layout.addWidget(score_group)
        
        # Status
        self.status_label = QLabel("Loading models...")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.status_label)
        
        # Add stretch
        controls_layout.addStretch()
        
        content_layout.addWidget(controls_group)
        
        main_layout.addLayout(content_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
    
    def apply_styles(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QLineEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QLabel {
                color: #333;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
    
    def load_models(self):
        """Load models in background thread"""
        self.model_loader = ModelLoader()
        self.model_loader.progress.connect(self.update_status)
        self.model_loader.finished.connect(self.on_models_loaded)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.model_loader.start()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def normalize_chinese_text(self, text):
        """Normalize Chinese text for better comparison"""
        if not text:
            return text
            
        # Remove whitespace and punctuation
        normalized = ''.join(char for char in text if char.strip() and not unicodedata.category(char).startswith('P'))
        
        # Convert traditional to simplified if converter is available
        if self.converter:
            try:
                normalized = self.converter.convert(normalized)
            except:
                pass  # Continue with original if conversion fails
        
        # Normalize Unicode (NFC normalization)
        normalized = unicodedata.normalize('NFC', normalized)
        
        return normalized
    
    def calculate_enhanced_score(self, original, response):
        """Calculate score with multiple methods and show detailed comparison"""
        if not response:
            return 0, "No response to score", original, response
        
        # Normalize both texts
        norm_original = self.normalize_chinese_text(original)
        norm_response = self.normalize_chinese_text(response)
        
        # Method 1: Exact match after normalization
        if norm_original == norm_response:
            return 100, "Perfect match!", norm_original, norm_response
        
        # Method 2: Character-level similarity
        char_similarity = SequenceMatcher(None, norm_original, norm_response).ratio()
        
        # Method 3: Pinyin comparison (pronunciation-based)
        orig_pinyin = ''.join([''.join(p) for p in pinyin(norm_original, style=Style.NORMAL)])
        resp_pinyin = ''.join([''.join(p) for p in pinyin(norm_response, style=Style.NORMAL)])
        pinyin_similarity = SequenceMatcher(None, orig_pinyin, resp_pinyin).ratio()
        
        # Method 4: Character sequence matching (order-independent)
        orig_chars = set(norm_original)
        resp_chars = set(norm_response)
        char_overlap = len(orig_chars & resp_chars) / max(len(orig_chars), len(resp_chars), 1)
        
        # Weighted scoring
        char_weight = 0.5
        pinyin_weight = 0.3
        overlap_weight = 0.2
        
        final_score = (char_similarity * char_weight + 
                      pinyin_similarity * pinyin_weight + 
                      char_overlap * overlap_weight)
        
        score = int(final_score * 100)
        
        # Generate detailed feedback
        feedback_parts = []
        if char_similarity > 0.9:
            feedback_parts.append("字符匹配很好")
        elif char_similarity > 0.7:
            feedback_parts.append("字符基本正确")
        else:
            feedback_parts.append("字符需要改进")
            
        if pinyin_similarity > 0.9:
            feedback_parts.append("发音很准确")
        elif pinyin_similarity > 0.7:
            feedback_parts.append("发音基本正确")
        else:
            feedback_parts.append("发音需要练习")
        
        feedback = " | ".join(feedback_parts)
        
        return score, feedback, norm_original, norm_response
    
    def set_asr_model(self, model_name):
        """Set the selected ASR model"""
        self.selected_asr_model = model_name
        self.status_label.setText(f"Selected ASR model: {model_name}")
        print(f"ASR model changed to: {model_name}")
    
    def on_speed_changed(self, value):
        """Handle speed slider changes"""
        self.current_speed = self.speed_options[value]
        self.speed_value_label.setText(f"Speed: {self.current_speed}")
        print(f"TTS speed changed to: {self.current_speed}")
    
    def on_models_loaded(self, success, message):
        """Handle model loading completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        
        if success:
            self.whisper_model = self.model_loader.whisper_model
            self.spark_tts_model = self.model_loader.spark_tts_model
            self.sensevoice_model = self.model_loader.sensevoice_model
            
            # Enable/disable SenseVoice radio button based on successful loading
            if self.sensevoice_model is not None:
                self.sensevoice_radio.setEnabled(True)
                self.sensevoice_radio.setText("SenseVoice")
            else:
                self.sensevoice_radio.setEnabled(False)
                self.sensevoice_radio.setText("SenseVoice (Failed to Load)")
        else:
            QMessageBox.critical(self, "Error", message)
        """Handle model loading completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        
        if success:
            self.whisper_model = self.model_loader.whisper_model
            self.spark_tts_model = self.model_loader.spark_tts_model
        else:
            QMessageBox.critical(self, "Error", message)
    
    def get_supported_sample_rate(self, device_id):
        """Find a supported sample rate for the device"""
        # Expanded sample rates including common USB mic rates
        sample_rates = [
            44100, 48000, 22050, 16000,  # Most common
            96000, 88200,                # High quality
            32000, 24000, 12000, 8000,   # Lower quality
            11025, 6000, 4000            # Very low quality fallbacks
        ]
        
        if device_id in self.device_sample_rates:
            return self.device_sample_rates[device_id]
        
        # Get device info to help with selection
        try:
            device_info = sd.query_devices(device_id)
            default_sr = int(device_info.get('default_samplerate', 44100))
            
            # Try the device's default sample rate first
            if default_sr not in sample_rates:
                sample_rates.insert(0, default_sr)
            else:
                # Move default to front
                sample_rates.remove(default_sr)
                sample_rates.insert(0, default_sr)
                
        except Exception:
            pass
        
        for rate in sample_rates:
            try:
                # Test with minimal parameters for USB compatibility
                test_stream = sd.InputStream(
                    device=device_id,
                    samplerate=rate,
                    channels=1,
                    dtype=np.float32,
                    blocksize=1024,  # Larger block for USB devices
                    latency=None     # Let the system decide latency
                )
                test_stream.start()
                # Very brief test
                sd.sleep(50)  # 50ms test
                test_stream.stop()
                test_stream.close()
                
                # If we get here, the sample rate works
                self.device_sample_rates[device_id] = rate
                print(f"Device {device_id} supports {rate} Hz")
                return rate
                
            except Exception as e:
                # Print debug info for USB devices
                if "USB" in str(sd.query_devices(device_id).get('name', '')):
                    print(f"USB device {device_id} failed at {rate} Hz: {type(e).__name__}")
                continue
        
        # If no sample rate works, return most common one
        print(f"Warning: No working sample rate found for device {device_id}, using 44100 Hz")
        self.device_sample_rates[device_id] = 44100
        return 44100
    
    def populate_audio_devices(self):
        """Populate the microphone selection dropdown"""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            print("Available audio devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Input device
                    device_name = device['name']
                    print(f"  {i}: {device_name} (default SR: {device.get('default_samplerate', 'unknown')})")
                    input_devices.append((device_name, i))
            
            self.mic_combo.clear()
            for name, device_id in input_devices:
                self.mic_combo.addItem(name, device_id)
            
            # Set default device
            try:
                default_device = sd.query_devices(kind='input')
                default_id = default_device['index'] if 'index' in default_device else None
                print(f"Default input device: {default_id}")
                
                if default_id is not None:
                    for i in range(self.mic_combo.count()):
                        if self.mic_combo.itemData(i) == default_id:
                            self.mic_combo.setCurrentIndex(i)
                            self.selected_device = default_id
                            break
                else:
                    # Just select first device if no default found
                    if self.mic_combo.count() > 0:
                        self.selected_device = self.mic_combo.itemData(0)
                        print(f"Using first available device: {self.selected_device}")
                        
            except Exception as e:
                print(f"Error setting default device: {e}")
                # Fallback to first device
                if self.mic_combo.count() > 0:
                    self.selected_device = self.mic_combo.itemData(0)
                    print(f"Fallback to first device: {self.selected_device}")
                
        except Exception as e:
            print(f"Warning: Could not enumerate audio devices: {e}")
            # Add a dummy entry so the app doesn't crash
            self.mic_combo.addItem("No audio devices found", None)
    
    def on_mic_changed(self):
        """Handle microphone selection change"""
        current_index = self.mic_combo.currentIndex()
        if current_index >= 0:
            new_device = self.mic_combo.itemData(current_index)
            
            # Only restart if device actually changed
            if self.selected_device != new_device:
                # Stop current monitoring
                if hasattr(self, 'monitor_stream'):
                    try:
                        if self.monitor_stream.active:
                            self.monitor_stream.stop()
                        self.monitor_stream.close()
                    except:
                        pass
                
                self.selected_device = new_device
                print(f"Selected device: {self.selected_device}")
                
                # Short delay before restarting
                if hasattr(self, 'restart_timer'):
                    self.restart_timer.stop()
                
                self.restart_timer = QTimer()
                self.restart_timer.timeout.connect(self.start_audio_monitoring)
                self.restart_timer.setSingleShot(True)
                self.restart_timer.start(200)  # 200ms delay
    
    def start_audio_monitoring(self):
        """Start monitoring audio levels"""
        if self.selected_device is None:
            return
            
        # Don't start monitoring if we're currently recording
        if self.is_recording:
            return
            
        # Close existing monitor stream first
        if hasattr(self, 'monitor_stream'):
            try:
                if self.monitor_stream.active:
                    self.monitor_stream.stop()
                self.monitor_stream.close()
            except:
                pass
            
        try:
            # Get the supported sample rate for this device
            device_sample_rate = self.get_supported_sample_rate(self.selected_device)
            
            def level_callback(indata, frames, time, status):
                if status or not hasattr(self, 'current_audio_level'):
                    return  # Skip if there are issues
                try:
                    # Calculate RMS level
                    rms = np.sqrt(np.mean(indata**2))
                    self.current_audio_level = min(1.0, rms * 10)  # Scale for visibility
                except:
                    pass  # Ignore calculation errors
            
            # Use simple, conservative settings
            self.monitor_stream = sd.InputStream(
                device=self.selected_device,
                samplerate=device_sample_rate,
                channels=1,
                callback=level_callback,
                blocksize=1024,
                dtype=np.float32
            )
            
            self.monitor_stream.start()
            
            # Start timer to update UI
            if not self.audio_level_timer.isActive():
                self.audio_level_timer.start(100)  # Update every 100ms
            
            # Update the combo box to show the sample rate
            current_text = self.mic_combo.currentText()
            if "Hz)" not in current_text and "unavailable)" not in current_text:
                new_text = f"{current_text} ({device_sample_rate} Hz)"
                self.mic_combo.setItemText(self.mic_combo.currentIndex(), new_text)
            
            print(f"Audio monitoring started for device {self.selected_device} at {device_sample_rate} Hz")
            
        except Exception as e:
            print(f"Could not start audio monitoring: {e}")
            self.status_label.setText("Audio monitoring unavailable")
            
            # Update combo to show monitoring is unavailable
            try:
                current_text = self.mic_combo.currentText()
                if "unavailable)" not in current_text:
                    new_text = f"{current_text} (monitoring unavailable)"
                    self.mic_combo.setItemText(self.mic_combo.currentIndex(), new_text)
            except:
                pass
    
    def update_audio_level(self):
        """Update the audio level display"""
        self.audio_level_widget.set_level(self.current_audio_level)
        # Decay the level for visual effect
        self.current_audio_level *= 0.95
    
    def load_csv(self):
        """Load CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file", "", "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                required_columns = ['zh', 'eng']
                
                if not all(col in self.df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in self.df.columns]
                    QMessageBox.critical(self, "Error", f"Missing required columns: {missing}")
                    return
                
                # Add score column if it doesn't exist
                if 'score' not in self.df.columns:
                    self.df['score'] = None
                
                # Add previous_score column if it doesn't exist for tracking improvement
                if 'previous_score' not in self.df.columns:
                    self.df['previous_score'] = None
                
                # Add running_average column if it doesn't exist
                if 'running_average' not in self.df.columns:
                    self.df['running_average'] = None
                
                # Add pinyin column if it doesn't exist (will be populated as fallback)
                if 'pinyin' not in self.df.columns:
                    self.df['pinyin'] = None
                
                self.file_path_edit.setText(file_path)
                self.csv_file_path = file_path  # Store for saving later
                self.current_index = 0
                self.update_display()
                self.status_label.setText(f"Loaded {len(self.df)} sentences")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
    
    def play_recorded_audio(self):
        """Play the recorded audio"""
        if not hasattr(self, 'recorded_audio') or self.recorded_audio is None:
            QMessageBox.warning(self, "Warning", "No recorded audio to play")
            return
        
        try:
            # Save recorded audio to temporary file for playback
            temp_playback_path = tempfile.mktemp(suffix='.wav')
            sample_rate = getattr(self, 'current_recording_sample_rate', self.sample_rate)
            sf.write(temp_playback_path, self.recorded_audio, sample_rate)
            
            # Use QMediaPlayer for playback
            url = QUrl.fromLocalFile(os.path.abspath(temp_playback_path))
            self.media_player.setSource(url)
            self.media_player.play()
            
            # Store path for cleanup later
            self.temp_playback_path = temp_playback_path
            
            self.status_label.setText("Playing recorded audio...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to play recorded audio: {str(e)}")
    
    def update_display(self):
        """Update display with current sentence"""
        if self.df is None or len(self.df) == 0:
            return
        
        row = self.df.iloc[self.current_index]
        
        # Display Chinese text
        chinese_text = row['zh']
        self.chinese_text.setPlainText(chinese_text)
        
        # Get pinyin from CSV first, fallback to generation if empty/missing
        csv_pinyin = row.get('pinyin')
        if csv_pinyin and not pd.isna(csv_pinyin) and str(csv_pinyin).strip():
            # Use pinyin from CSV
            pinyin_text = str(csv_pinyin).strip()
        else:
            # Generate pinyin as fallback and store it
            pinyin_result = pinyin(chinese_text, style=Style.TONE)
            pinyin_text = ' '.join([''.join(p) for p in pinyin_result])
            # Store generated pinyin back to CSV for future use
            self.df.loc[self.current_index, 'pinyin'] = pinyin_text
        
        self.pinyin_text.setPlainText(pinyin_text)
        
        # Display English
        self.english_text.setPlainText(row['eng'])
        
        self.sentence_label.setText(f"{self.current_index + 1} / {len(self.df)}")
        
        # Clear response
        self.response_text.clear()
        self.response_pinyin.clear()
        self.score_label.setText("No score yet")
        self.score_label.setStyleSheet("")
        
        # Update previous score display
        self.update_score_display()
        self.update_running_average_display()
        
        # Reset recording-related buttons
        self.transcribe_button.setEnabled(False)
        self.playback_button.setEnabled(False)
    
    def update_running_average_display(self):
        """Update the running average display"""
        if self.df is None or len(self.df) == 0:
            self.running_avg_label.setText("")
            return
        
        running_avg = self.calculate_running_average(self.current_index)
        
        if running_avg is not None:
            self.running_avg_label.setText(f"Running Avg: {running_avg:.1f}%")
            
            # Color code based on average performance
            if running_avg >= 85:
                color = "#4CAF50"  # Green
            elif running_avg >= 70:
                color = "#2196F3"  # Blue
            elif running_avg >= 55:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
            
            self.running_avg_label.setStyleSheet(f"color: {color}; font-style: italic;")
        else:
            self.running_avg_label.setText("")
            self.running_avg_label.setStyleSheet("")
    
    def calculate_running_average(self, current_index):
        """Calculate running average score for the current sentence"""
        if self.df is None:
            return None
        
        # Get all non-null scores for this sentence (previous attempts)
        current_scores = []
        
        # Get current score if it exists
        current_score = self.df.iloc[current_index].get('score')
        if current_score is not None and not pd.isna(current_score):
            current_scores.append(current_score)
        
        # Get previous score if it exists
        previous_score = self.df.iloc[current_index].get('previous_score')
        if previous_score is not None and not pd.isna(previous_score):
            current_scores.append(previous_score)
        
        # Calculate average if we have scores
        if current_scores:
            return sum(current_scores) / len(current_scores)
        
        return None
    
    def update_score_display(self):
        """Update the score display to show current and previous scores"""
        if self.df is None or len(self.df) == 0:
            self.prev_score_label.setText("")
            return
        
        row = self.df.iloc[self.current_index]
        current_score = row.get('score')
        previous_score = row.get('previous_score')
        
        if current_score is not None and not pd.isna(current_score):
            # Show improvement/decline
            if previous_score is not None and not pd.isna(previous_score):
                diff = current_score - previous_score
                if diff > 0:
                    trend = f"↗ +{diff:.0f}"
                    color = "green"
                elif diff < 0:
                    trend = f"↘ {diff:.0f}"
                    color = "red"
                else:
                    trend = "→ same"
                    color = "blue"
                
                prev_text = f"Last: {previous_score:.0f}% | {trend}"
                self.prev_score_label.setText(prev_text)
                self.prev_score_label.setStyleSheet(f"color: {color};")
            else:
                self.prev_score_label.setText(f"Score: {current_score:.0f}%")
                self.prev_score_label.setStyleSheet("color: gray;")
        else:
            self.prev_score_label.setText("No previous attempts")
            self.prev_score_label.setStyleSheet("color: gray;")
    
    def previous_sentence(self):
        """Go to previous sentence"""
        if self.df is None or len(self.df) == 0:
            return
        self.current_index = (self.current_index - 1) % len(self.df)
        self.update_display()
    
    def next_sentence(self):
        """Go to next sentence"""
        if self.df is None or len(self.df) == 0:
            return
        self.current_index = (self.current_index + 1) % len(self.df)
        self.update_display()
    
    def generate_audio(self):
        """Generate audio for current sentence"""
        if self.df is None or self.spark_tts_model is None:
            return
        
        text = self.df.iloc[self.current_index]['zh']
        
        self.audio_generator = AudioGenerator(self.spark_tts_model, text, self.current_speed)
        self.audio_generator.progress.connect(self.update_status)
        self.audio_generator.finished.connect(self.on_audio_generated)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.audio_generator.start()
    
    def on_audio_generated(self, success, message, audio_path):
        """Handle audio generation completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        
        if success:
            self.temp_audio_path = audio_path
        else:
            QMessageBox.critical(self, "Error", message)
    
    def play_prompt_audio(self):
        """Play generated audio"""
        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            try:
                # Use QMediaPlayer for better compatibility
                url = QUrl.fromLocalFile(os.path.abspath(self.temp_audio_path))
                self.media_player.setSource(url)
                self.media_player.play()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to play audio: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please generate audio first")
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording"""
        if self.selected_device is None:
            QMessageBox.warning(self, "Warning", "No microphone selected")
            return
        
        # Stop monitoring to free up the device
        if hasattr(self, 'monitor_stream'):
            try:
                if self.monitor_stream.active:
                    self.monitor_stream.stop()
                self.monitor_stream.close()
                sd.sleep(50)  # Shorter wait
            except:
                pass
            
        self.is_recording = True
        self.record_button.setText("⏹ Stop Recording")
        self.recorded_audio = []
        self.status_label.setText("Recording...")
        
        # Get the supported sample rate for recording
        recording_sample_rate = self.get_supported_sample_rate(self.selected_device)
        
        def record_callback(indata, frames, time, status):
            if self.is_recording and hasattr(self, 'recorded_audio'):
                try:
                    self.recorded_audio.append(indata.copy())
                except:
                    pass  # Ignore callback errors
        
        try:
            # Simple recording setup
            self.stream = sd.InputStream(
                device=self.selected_device,
                samplerate=recording_sample_rate,
                channels=1,
                callback=record_callback,
                dtype=np.float32,
                blocksize=1024
            )
            self.stream.start()
            self.current_recording_sample_rate = recording_sample_rate
            print(f"Recording started at {recording_sample_rate} Hz")
            
        except Exception as e:
            self.is_recording = False
            self.record_button.setText("🎤 Start Recording")
            error_msg = f"Failed to start recording: {str(e)}"
            print(error_msg)
            
            # Restart monitoring
            self.start_audio_monitoring()
            
            if "Device unavailable" in str(e):
                error_msg += "\n\nTips:\n• Close other audio applications\n• Try a different microphone"
            
            QMessageBox.critical(self, "Recording Error", error_msg)
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.record_button.setText("🎤 Start Recording")
        
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        if self.recorded_audio and len(self.recorded_audio) > 0:
            try:
                self.recorded_audio = np.concatenate(self.recorded_audio, axis=0).flatten()
                self.transcribe_button.setEnabled(True)
                self.playback_button.setEnabled(True)  # Enable playback button
                self.status_label.setText("Recording stopped. Click Transcribe or Play Recording.")
                print(f"Recorded {len(self.recorded_audio)} samples")
            except Exception as e:
                print(f"Error processing recorded audio: {e}")
                self.status_label.setText("Error processing recording")
        else:
            self.status_label.setText("No audio recorded")
        
        # Restart audio monitoring after a delay
        if hasattr(self, 'restart_timer'):
            self.restart_timer.stop()
        
        self.restart_timer = QTimer()
        self.restart_timer.timeout.connect(self.start_audio_monitoring)
        self.restart_timer.setSingleShot(True)
        self.restart_timer.start(300)  # 300ms delay
    
    def transcribe_audio(self):
        """Transcribe recorded audio using selected model"""
        if self.recorded_audio is None:
            return
        
        # Check if selected model is available
        if self.selected_asr_model == "sensevoice" and self.sensevoice_model is None:
            QMessageBox.warning(self, "Model Not Available", "SenseVoice model is not loaded. Falling back to Whisper.")
            self.selected_asr_model = "whisper"
            self.whisper_radio.setChecked(True)
        
        if self.selected_asr_model == "whisper" and self.whisper_model is None:
            QMessageBox.critical(self, "Error", "No ASR model available for transcription")
            return
        
        # Use the sample rate that was actually used for recording
        actual_sample_rate = getattr(self, 'current_recording_sample_rate', self.sample_rate)
        
        self.transcriber = AudioTranscriber(
            self.whisper_model, 
            self.sensevoice_model, 
            self.recorded_audio, 
            actual_sample_rate,
            self.selected_asr_model
        )
        self.transcriber.progress.connect(self.update_status)
        self.transcriber.finished.connect(self.on_transcription_finished)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.transcriber.start()
    
    def on_transcription_finished(self, success, message, transcription, pinyin_text):
        """Handle transcription completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        
        if success:
            self.response_text.setPlainText(transcription)
            self.response_pinyin.setPlainText(pinyin_text)
            self.transcribe_button.setEnabled(False)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def calculate_score(self):
        """Calculate accuracy score with enhanced Chinese text comparison"""
        if self.df is None:
            return
        
        original = self.df.iloc[self.current_index]['zh']
        response = self.response_text.toPlainText().strip()
        
        # Use enhanced scoring
        score, feedback, norm_original, norm_response = self.calculate_enhanced_score(original, response)
        
        # Store previous score before updating
        current_score = self.df.iloc[self.current_index].get('score')
        if current_score is not None and not pd.isna(current_score):
            self.df.loc[self.current_index, 'previous_score'] = current_score
        
        # Update with new score
        self.df.loc[self.current_index, 'score'] = score
        
        # Calculate and store running average
        running_avg = self.calculate_running_average(self.current_index)
        if running_avg is not None:
            self.df.loc[self.current_index, 'running_average'] = running_avg
        
        # Color coding
        if score >= 90:
            color = "#4CAF50"  # Green
            grade = "Excellent!"
        elif score >= 75:
            color = "#2196F3"  # Blue
            grade = "Good"
        elif score >= 60:
            color = "#FF9800"  # Orange
            grade = "Fair"
        else:
            color = "#F44336"  # Red
            grade = "Needs Practice"
        
        # Show score and detailed feedback
        score_text = f"{score}% - {grade}"
        if feedback != grade:
            score_text += f"\n{feedback}"
        
        self.score_label.setText(score_text)
        self.score_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # Update score display to show improvement
        self.update_score_display()
        self.update_running_average_display()
        
        # Show comparison in status if texts were normalized differently
        if norm_original != original or norm_response != response:
            comparison = f"Normalized: '{norm_original}' vs '{norm_response}'"
            self.status_label.setText(comparison)
            print(f"Original: '{original}' -> '{norm_original}'")
            print(f"Response: '{response}' -> '{norm_response}'")
            print(f"Score: {score}% ({feedback})")
        
        # Auto-save after scoring
        self.save_csv()
    
    def save_csv(self):
        """Save the current dataframe back to CSV"""
        if self.df is None or self.csv_file_path is None:
            QMessageBox.warning(self, "Warning", "No data to save or file path not set")
            return
        
        try:
            self.df.to_csv(self.csv_file_path, index=False)
            self.status_label.setText("Progress saved!")
            print(f"CSV saved to {self.csv_file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV: {str(e)}")
            print(f"Save error: {e}")
    
    def closeEvent(self, event):
        """Handle application closing"""
        # Stop audio monitoring
        if hasattr(self, 'monitor_stream'):
            self.monitor_stream.stop()
            self.monitor_stream.close()
        
        if hasattr(self, 'stream') and self.is_recording:
            self.stream.stop()
            self.stream.close()
        
        self.audio_level_timer.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = MandarinStudyApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()