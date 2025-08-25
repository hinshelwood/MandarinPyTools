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

class GradingEngine:
    @staticmethod
    def calculate_similarity(user_text, correct_text):
        """Calculate similarity between user input and correct translation"""
        # Normalize texts
        user_normalized = user_text.lower().strip()
        correct_normalized = correct_text.lower().strip()
        
        # Calculate sequence similarity
        similarity = SequenceMatcher(None, user_normalized, correct_normalized).ratio()
        
        # Convert to percentage score
        score = int(similarity * 100)
        return score
    
    @staticmethod
    def get_feedback(score):
        """Generate feedback based on score"""
        if score >= 90:
            return "Excellent! Perfect translation!", "green"
        elif score >= 80:
            return "Very good! Minor differences.", "lightgreen"
        elif score >= 70:
            return "Good attempt! Some differences.", "orange"
        elif score >= 60:
            return "Okay. Significant differences.", "lightorange"
        else:
            return "Needs improvement. Try again!", "red"

class CSVManager:
    @staticmethod
    def load_csv(filepath):
        """Load CSV file and ensure required columns exist"""
        try:
            df = pd.read_csv(filepath)
            
            # Check for required columns
            if 'zh' not in df.columns or 'eng' not in df.columns:
                raise ValueError("CSV must contain 'zh' and 'eng' columns")
            
            # Add scoring columns if they don't exist
            required_columns = ['listen_current_score', 'listen_previous_score', 'listen_score_average']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan
            
            return df, None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def save_csv(df, filepath):
        """Save DataFrame to CSV"""
        try:
            df.to_csv(filepath, index=False)
            return True, None
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def update_scores(df, index, new_score):
        """Update scores for a given row"""
        # Move current to previous
        if not pd.isna(df.loc[index, 'listen_current_score']):
            df.loc[index, 'listen_previous_score'] = df.loc[index, 'listen_current_score']
        
        # Set new current score
        df.loc[index, 'listen_current_score'] = new_score
        
        # Calculate average (excluding NaN values)
        scores = []
        if not pd.isna(df.loc[index, 'listen_current_score']):
            scores.append(df.loc[index, 'listen_current_score'])
        if not pd.isna(df.loc[index, 'listen_previous_score']):
            scores.append(df.loc[index, 'listen_previous_score'])
        
        if scores:
            df.loc[index, 'listen_score_average'] = np.mean(scores)
        
        return df

class MandarinListeningPractice(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandarin Listening Practice")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize variables
        self.df = None
        self.csv_filepath = None
        self.current_index = 0
        self.models_loaded = False
        self.whisper_model = None
        self.spark_tts_model = None
        self.sensevoice_model = None
        self.current_audio_path = None
        
        # Audio player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        self.init_ui()
        self.load_models()
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Top section - File loading and model status
        top_group = QGroupBox("Setup")
        top_layout = QHBoxLayout(top_group)
        
        self.load_csv_btn = QPushButton("Load CSV File")
        self.load_csv_btn.clicked.connect(self.load_csv_file)
        top_layout.addWidget(self.load_csv_btn)
        
        self.csv_status_label = QLabel("No CSV loaded")
        top_layout.addWidget(self.csv_status_label)
        
        top_layout.addStretch()
        
        self.model_status_label = QLabel("Loading models...")
        top_layout.addWidget(self.model_status_label)
        
        main_layout.addWidget(top_group)
        
        # Progress bar for model loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(True)
        main_layout.addWidget(self.progress_bar)
        
        # Practice section
        self.practice_group = QGroupBox("Listening Practice")
        self.practice_group.setEnabled(False)
        practice_layout = QVBoxLayout(self.practice_group)
        
        # Chinese text display
        self.chinese_label = QLabel("Chinese text will appear here")
        self.chinese_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.chinese_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chinese_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        practice_layout.addWidget(self.chinese_label)
        
        # Audio controls
        audio_layout = QHBoxLayout()
        audio_layout.setSpacing(10)
        audio_layout.setContentsMargins(0, 10, 0, 10)
        
        self.generate_audio_btn = QPushButton("Generate Audio")
        self.generate_audio_btn.clicked.connect(self.generate_audio)
        self.generate_audio_btn.setMinimumHeight(35)
        audio_layout.addWidget(self.generate_audio_btn)
        
        self.play_audio_btn = QPushButton("Play Audio")
        self.play_audio_btn.clicked.connect(self.play_audio)
        self.play_audio_btn.setEnabled(False)
        self.play_audio_btn.setMinimumHeight(35)
        audio_layout.addWidget(self.play_audio_btn)
        
        # Speed control
        speed_label = QLabel("Speed:")
        audio_layout.addWidget(speed_label)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["slow", "low", "moderate", "high", "fast"])
        self.speed_combo.setCurrentText("low")
        self.speed_combo.setMinimumHeight(35)
        audio_layout.addWidget(self.speed_combo)
        
        audio_layout.addStretch()
        practice_layout.addLayout(audio_layout)
        
        # User input section
        input_layout = QVBoxLayout()
        
        input_label = QLabel("Enter English translation:")
        input_label.setFont(QFont("Arial", 12))
        input_layout.addWidget(input_label)
        
        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(100)
        self.user_input.setPlaceholderText("Type your English translation here...")
        input_layout.addWidget(self.user_input)
        
        submit_layout = QHBoxLayout()
        self.submit_btn = QPushButton("Submit Answer")
        self.submit_btn.clicked.connect(self.submit_answer)
        submit_layout.addWidget(self.submit_btn)
        
        self.show_answer_btn = QPushButton("Show Correct Answer")
        self.show_answer_btn.clicked.connect(self.show_correct_answer)
        submit_layout.addWidget(self.show_answer_btn)
        
        submit_layout.addStretch()
        input_layout.addLayout(submit_layout)
        
        practice_layout.addLayout(input_layout)
        
        # Feedback section
        self.feedback_label = QLabel("")
        self.feedback_label.setFont(QFont("Arial", 12))
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_label.setStyleSheet("QLabel { padding: 10px; border-radius: 5px; }")
        practice_layout.addWidget(self.feedback_label)
        
        # Correct answer display
        self.correct_answer_label = QLabel("")
        self.correct_answer_label.setFont(QFont("Arial", 11))
        self.correct_answer_label.setWordWrap(True)
        self.correct_answer_label.setStyleSheet("QLabel { background-color: #e8f4f8; padding: 8px; border-radius: 5px; }")
        self.correct_answer_label.setVisible(False)
        practice_layout.addWidget(self.correct_answer_label)
        
        main_layout.addWidget(self.practice_group)
        
        # Navigation and scoring
        nav_group = QGroupBox("Navigation & Scoring")
        nav_layout = QHBoxLayout(nav_group)
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_item)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_item)
        nav_layout.addWidget(self.next_btn)
        
        nav_layout.addStretch()
        
        self.position_label = QLabel("0 / 0")
        nav_layout.addWidget(self.position_label)
        
        nav_layout.addStretch()
        
        self.score_label = QLabel("Score: --")
        nav_layout.addWidget(self.score_label)
        
        main_layout.addWidget(nav_group)
        
        # Set initial state
        self.update_navigation_state()
    
    def load_models(self):
        """Load ML models in background thread"""
        self.model_loader = ModelLoader()
        self.model_loader.progress.connect(self.update_model_loading_progress)
        self.model_loader.finished.connect(self.on_models_loaded)
        self.model_loader.start()
    
    def update_model_loading_progress(self, message):
        """Update model loading progress"""
        self.model_status_label.setText(message)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
    
    def on_models_loaded(self, success, message):
        """Handle model loading completion"""
        if success:
            self.models_loaded = True
            self.whisper_model = self.model_loader.whisper_model
            self.spark_tts_model = self.model_loader.spark_tts_model
            self.sensevoice_model = self.model_loader.sensevoice_model
            self.model_status_label.setText("✓ Models ready")
            self.model_status_label.setStyleSheet("color: green;")
        else:
            self.model_status_label.setText("✗ Model loading failed")
            self.model_status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", f"Failed to load models:\n{message}")
        
        self.progress_bar.setVisible(False)
        self.update_ui_state()
    
    def load_csv_file(self):
        """Load CSV file with Chinese and English translations"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV files (*.csv)"
        )
        
        if filepath:
            df, error = CSVManager.load_csv(filepath)
            if error:
                QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{error}")
                return
            
            self.df = df
            self.csv_filepath = filepath
            self.current_index = 0
            
            filename = os.path.basename(filepath)
            self.csv_status_label.setText(f"✓ Loaded: {filename} ({len(df)} items)")
            self.csv_status_label.setStyleSheet("color: green;")
            
            self.update_ui_state()
            self.load_current_item()
    
    def update_ui_state(self):
        """Update UI state based on loaded models and CSV"""
        csv_loaded = self.df is not None
        models_ready = self.models_loaded
        
        self.practice_group.setEnabled(csv_loaded and models_ready)
        
        if csv_loaded and models_ready:
            self.update_navigation_state()
    
    def update_navigation_state(self):
        """Update navigation buttons and position label"""
        if self.df is None:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.position_label.setText("0 / 0")
            return
        
        total_items = len(self.df)
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < total_items - 1)
        self.position_label.setText(f"{self.current_index + 1} / {total_items}")
        
        # Update score display
        self.update_score_display()
    
    def update_score_display(self):
        """Update score display for current item"""
        if self.df is None:
            self.score_label.setText("Score: --")
            return
        
        current_score = self.df.iloc[self.current_index]['listen_current_score']
        avg_score = self.df.iloc[self.current_index]['listen_score_average']
        
        if pd.isna(current_score):
            score_text = "Score: --"
        else:
            score_text = f"Score: {current_score:.0f}"
            if not pd.isna(avg_score):
                score_text += f" (Avg: {avg_score:.1f})"
        
        self.score_label.setText(score_text)
    
    def load_current_item(self):
        """Load the current Chinese text"""
        if self.df is None:
            return
        
        current_row = self.df.iloc[self.current_index]
        chinese_text = current_row['zh']
        
        self.chinese_label.setText(chinese_text)
        self.user_input.clear()
        self.feedback_label.setText("")
        self.correct_answer_label.setVisible(False)
        self.play_audio_btn.setEnabled(False)
        self.current_audio_path = None
        
        self.update_navigation_state()
    
    def generate_audio(self):
        """Generate audio for current Chinese text"""
        if not self.models_loaded or self.df is None:
            return
        
        current_row = self.df.iloc[self.current_index]
        chinese_text = current_row['zh']
        speed = self.speed_combo.currentText()
        
        self.generate_audio_btn.setEnabled(False)
        self.generate_audio_btn.setText("Generating...")
        
        self.audio_generator = AudioGenerator(self.spark_tts_model, chinese_text, speed)
        self.audio_generator.progress.connect(lambda msg: None)  # Could show progress
        self.audio_generator.finished.connect(self.on_audio_generated)
        self.audio_generator.start()
    
    def on_audio_generated(self, success, message, audio_path):
        """Handle audio generation completion"""
        self.generate_audio_btn.setEnabled(True)
        self.generate_audio_btn.setText("Generate Audio")
        
        if success:
            self.current_audio_path = audio_path
            self.play_audio_btn.setEnabled(True)
            # Auto-play the generated audio
            self.play_audio()
        else:
            QMessageBox.critical(self, "Error", f"Failed to generate audio:\n{message}")
    
    def play_audio(self):
        """Play the generated audio"""
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try:
                url = QUrl.fromLocalFile(self.current_audio_path)
                self.media_player.setSource(url)
                self.media_player.play()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to play audio:\n{str(e)}")
    
    def submit_answer(self):
        """Submit and grade user's translation"""
        if self.df is None:
            return
        
        user_text = self.user_input.toPlainText().strip()
        if not user_text:
            QMessageBox.warning(self, "Warning", "Please enter a translation before submitting.")
            return
        
        current_row = self.df.iloc[self.current_index]
        correct_text = current_row['eng']
        
        # Calculate score
        score = GradingEngine.calculate_similarity(user_text, correct_text)
        feedback_text, color = GradingEngine.get_feedback(score)
        
        # Update scores in DataFrame
        self.df = CSVManager.update_scores(self.df, self.current_index, score)
        
        # Save to CSV
        success, error = CSVManager.save_csv(self.df, self.csv_filepath)
        if not success:
            QMessageBox.warning(self, "Warning", f"Failed to save scores:\n{error}")
        
        # Display feedback
        self.feedback_label.setText(f"{feedback_text} (Score: {score}%)")
        self.feedback_label.setStyleSheet(f"QLabel {{ background-color: {color}; padding: 10px; border-radius: 5px; }}")
        
        # Update score display
        self.update_score_display()
    
    def show_correct_answer(self):
        """Show the correct English translation"""
        if self.df is None:
            return
        
        current_row = self.df.iloc[self.current_index]
        correct_text = current_row['eng']
        
        self.correct_answer_label.setText(f"Correct answer: {correct_text}")
        self.correct_answer_label.setVisible(True)
    
    def previous_item(self):
        """Go to previous item"""
        if self.df is None or self.current_index <= 0:
            return
        
        self.current_index -= 1
        self.load_current_item()
    
    def next_item(self):
        """Go to next item"""
        if self.df is None or self.current_index >= len(self.df) - 1:
            return
        
        self.current_index += 1
        self.load_current_item()
    
    def closeEvent(self, event):
        """Handle application close"""
        # Clean up temporary audio files
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try:
                os.remove(self.current_audio_path)
            except:
                pass
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Mandarin Listening Practice")
    app.setApplicationVersion("1.0")
    
    window = MandarinListeningPractice()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()