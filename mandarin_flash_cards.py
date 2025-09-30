import sys
import os
import pandas as pd
import numpy as np
import tempfile
import torch
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                            QFileDialog, QMessageBox, QGroupBox, QCheckBox,
                            QButtonGroup, QRadioButton, QSpacerItem, QSizePolicy,
                            QProgressBar, QLineEdit, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl

# Add your Spark-TTS path if available
try:
    sys.path.append(os.path.expanduser("~/software/Spark-TTS/"))
    from cli.SparkTTS import SparkTTS
    SPARKTTS_AVAILABLE = True
except ImportError:
    SPARKTTS_AVAILABLE = False
    print("Spark-TTS not available - audio generation disabled")

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
            
            # Save to temporary file
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            import soundfile as sf
            sf.write(temp_audio_path, wav, 16000)
            
            self.finished.emit(True, "Audio generated successfully!", temp_audio_path)
            
        except Exception as e:
            self.finished.emit(False, f"Error generating audio: {str(e)}", "")

class ModelLoader(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def run(self):
        try:
            if SPARKTTS_AVAILABLE:
                self.progress.emit("Loading SparkTTS model...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_dir = "/home/richard/software/Spark-TTS/pretrained_models/Spark-TTS-0.5B/"
                spark_tts_model = SparkTTS(model_dir, device)
                self.spark_tts_model = spark_tts_model
                self.finished.emit(True, "Models loaded successfully!")
            else:
                self.spark_tts_model = None
                self.finished.emit(True, "Ready! (Audio generation unavailable)")
            
        except Exception as e:
            self.finished.emit(False, f"Error loading models: {str(e)}")

class MandarinFlashcardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandarin Flashcard Study Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.spark_tts_model = None
        self.df = None
        self.current_index = 0
        self.csv_file_path = None
        self.temp_audio_path = None
        self.current_speed = "low"
        
        # Study settings
        self.pinyin_hints_enabled = True
        self.study_direction = "zh_to_eng"  # "zh_to_eng" or "eng_to_zh"
        self.show_answer = False
        
        # Speed options for TTS
        self.speed_options = ['very_low', 'low', 'moderate', 'high', 'very_high']
        
        # Audio player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Setup UI
        self.setup_ui()
        self.apply_styles()
        
        # Load models if available
        if SPARKTTS_AVAILABLE:
            self.load_models()
        else:
            self.status_label.setText("Ready! (Audio generation unavailable)")
    
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
        self.file_path_edit.setPlaceholderText("Select CSV file with columns: zh, pinyin, eng")
        file_layout.addWidget(self.file_path_edit)
        
        browse_btn = QPushButton("Browse CSV")
        browse_btn.clicked.connect(self.load_csv)
        file_layout.addWidget(browse_btn)
        
        main_layout.addWidget(file_group)
        
        # Settings section
        settings_group = QGroupBox("Study Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        # Direction settings
        direction_group = QGroupBox("Study Direction")
        direction_layout = QVBoxLayout(direction_group)
        
        self.direction_button_group = QButtonGroup()
        
        self.zh_to_eng_radio = QRadioButton("Chinese → English")
        self.zh_to_eng_radio.setChecked(True)
        self.zh_to_eng_radio.toggled.connect(lambda: self.set_study_direction("zh_to_eng"))
        direction_layout.addWidget(self.zh_to_eng_radio)
        self.direction_button_group.addButton(self.zh_to_eng_radio)
        
        self.eng_to_zh_radio = QRadioButton("English → Chinese")
        self.eng_to_zh_radio.toggled.connect(lambda: self.set_study_direction("eng_to_zh"))
        direction_layout.addWidget(self.eng_to_zh_radio)
        self.direction_button_group.addButton(self.eng_to_zh_radio)
        
        settings_layout.addWidget(direction_group)
        
        # Hint settings
        hint_group = QGroupBox("Hints")
        hint_layout = QVBoxLayout(hint_group)
        
        self.pinyin_hints_checkbox = QCheckBox("Show Pinyin Hints")
        self.pinyin_hints_checkbox.setChecked(True)
        self.pinyin_hints_checkbox.toggled.connect(self.toggle_pinyin_hints)
        hint_layout.addWidget(self.pinyin_hints_checkbox)
        
        settings_layout.addWidget(hint_group)
        
        # Audio settings (if available)
        if SPARKTTS_AVAILABLE:
            audio_group = QGroupBox("Audio Settings")
            audio_layout = QVBoxLayout(audio_group)
            
            # Speed control
            speed_label = QLabel("Speech Speed:")
            audio_layout.addWidget(speed_label)
            
            speed_slider_layout = QHBoxLayout()
            slow_label = QLabel("Slow")
            speed_slider_layout.addWidget(slow_label)
            
            self.speed_slider = QSlider(Qt.Orientation.Horizontal)
            self.speed_slider.setMinimum(0)
            self.speed_slider.setMaximum(4)
            self.speed_slider.setValue(1)  # Default to 'low'
            self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.speed_slider.setTickInterval(1)
            self.speed_slider.valueChanged.connect(self.on_speed_changed)
            speed_slider_layout.addWidget(self.speed_slider)
            
            fast_label = QLabel("Fast")
            speed_slider_layout.addWidget(fast_label)
            audio_layout.addLayout(speed_slider_layout)
            
            self.speed_value_label = QLabel("Speed: low")
            self.speed_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            audio_layout.addWidget(self.speed_value_label)
            
            settings_layout.addWidget(audio_group)
        
        main_layout.addWidget(settings_group)
        
        # Main flashcard area
        card_layout = QHBoxLayout()
        
        # Question side
        question_group = QGroupBox("Question")
        question_layout = QVBoxLayout(question_group)
        question_group.setMinimumWidth(400)
        
        self.question_text = QTextEdit()
        self.question_text.setFont(QFont("Arial", 18))
        self.question_text.setMinimumHeight(150)
        self.question_text.setMaximumHeight(200)
        self.question_text.setReadOnly(True)
        self.question_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        question_layout.addWidget(self.question_text)
        
        # Pinyin hint (initially visible)
        self.pinyin_hint = QLabel("")
        self.pinyin_hint.setFont(QFont("Consolas", 14))
        self.pinyin_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pinyin_hint.setStyleSheet("color: #666; font-style: italic;")
        self.pinyin_hint.setWordWrap(True)
        question_layout.addWidget(self.pinyin_hint)
        
        # Audio button (if available)
        if SPARKTTS_AVAILABLE:
            audio_layout = QHBoxLayout()
            
            self.play_audio_btn = QPushButton("🔊 Play Audio")
            self.play_audio_btn.clicked.connect(self.play_audio)
            audio_layout.addWidget(self.play_audio_btn)
            
            self.generate_audio_btn = QPushButton("Generate Audio")
            self.generate_audio_btn.clicked.connect(self.generate_audio)
            audio_layout.addWidget(self.generate_audio_btn)
            
            question_layout.addLayout(audio_layout)
        
        card_layout.addWidget(question_group)
        
        # Answer side
        answer_group = QGroupBox("Answer")
        answer_layout = QVBoxLayout(answer_group)
        answer_group.setMinimumWidth(400)
        
        self.answer_text = QTextEdit()
        self.answer_text.setFont(QFont("Arial", 16))
        self.answer_text.setMinimumHeight(150)
        self.answer_text.setMaximumHeight(200)
        self.answer_text.setReadOnly(True)
        self.answer_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.answer_text.setVisible(False)  # Initially hidden
        answer_layout.addWidget(self.answer_text)
        
        # Show answer button
        self.show_answer_btn = QPushButton("Show Answer")
        self.show_answer_btn.clicked.connect(self.toggle_answer)
        answer_layout.addWidget(self.show_answer_btn)
        
        card_layout.addWidget(answer_group)
        
        main_layout.addLayout(card_layout)
        
        # Scoring section
        scoring_group = QGroupBox("Self-Assessment")
        scoring_layout = QVBoxLayout(scoring_group)
        
        scoring_info = QLabel("Did you know this card?")
        scoring_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scoring_layout.addWidget(scoring_info)
        
        score_buttons_layout = QHBoxLayout()
        
        # Add spacer for centering
        score_buttons_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        
        self.fail_btn = QPushButton("✗ No - Need Practice")
        self.fail_btn.clicked.connect(lambda: self.record_score(0))
        self.fail_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                min-width: 150px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)
        score_buttons_layout.addWidget(self.fail_btn)
        
        # Add spacer between buttons
        score_buttons_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))
        
        self.pass_btn = QPushButton("✓ Yes - I Knew It")
        self.pass_btn.clicked.connect(lambda: self.record_score(1))
        self.pass_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                min-width: 150px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        score_buttons_layout.addWidget(self.pass_btn)
        
        # Add spacer for centering
        score_buttons_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        
        scoring_layout.addLayout(score_buttons_layout)
        
        main_layout.addWidget(scoring_group)
        
        # Navigation and stats
        bottom_layout = QHBoxLayout()
        
        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        prev_btn = QPushButton("◀ Previous")
        prev_btn.clicked.connect(self.previous_card)
        nav_layout.addWidget(prev_btn)
        
        self.card_label = QLabel("0 / 0")
        self.card_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_label.setFont(QFont("Arial", 12))
        nav_layout.addWidget(self.card_label)
        
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self.next_card)
        nav_layout.addWidget(next_btn)
        
        bottom_layout.addWidget(nav_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.current_avg_label = QLabel("Current Average: N/A")
        self.current_avg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.current_avg_label)
        
        self.overall_avg_label = QLabel("Overall Average: N/A")
        self.overall_avg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.overall_avg_label)
        
        self.progress_label = QLabel("Cards Studied: 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.progress_label)
        
        bottom_layout.addWidget(stats_group)
        
        main_layout.addLayout(bottom_layout)
        
        # Status and progress
        self.status_label = QLabel("Load a CSV file to begin studying")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
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
                padding: 10px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 6px;
                min-height: 35px;
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
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: white;
                font-size: 16px;
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
        self.progress_bar.setRange(0, 0)
        self.model_loader.start()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def on_models_loaded(self, success, message):
        """Handle model loading completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        
        if success and hasattr(self.model_loader, 'spark_tts_model'):
            self.spark_tts_model = self.model_loader.spark_tts_model
        else:
            if not success:
                QMessageBox.warning(self, "Warning", f"Model loading failed: {message}")
    
    def on_speed_changed(self, value):
        """Handle speed slider changes"""
        self.current_speed = self.speed_options[value]
        self.speed_value_label.setText(f"Speed: {self.current_speed}")
    
    def set_study_direction(self, direction):
        """Set the study direction"""
        self.study_direction = direction
        self.show_answer = False
        self.update_display()
        print(f"Study direction changed to: {direction}")
    
    def toggle_pinyin_hints(self, enabled):
        """Toggle pinyin hints"""
        self.pinyin_hints_enabled = enabled
        self.update_display()
    
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
                
                # Add pinyin column if it doesn't exist
                if 'pinyin' not in self.df.columns:
                    self.df['pinyin'] = None
                
                # Initialize score tracking columns
                score_columns = ['score_zh_to_eng', 'score_eng_to_zh', 'attempts_zh_to_eng', 'attempts_eng_to_zh']
                for col in score_columns:
                    if col not in self.df.columns:
                        self.df[col] = None
                
                # Generate missing pinyin
                self.generate_missing_pinyin()
                
                self.file_path_edit.setText(file_path)
                self.csv_file_path = file_path
                self.current_index = 0
                self.show_answer = False
                self.update_display()
                self.update_statistics()
                self.status_label.setText(f"Loaded {len(self.df)} flashcards")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
    
    def generate_missing_pinyin(self):
        """Generate pinyin for entries that don't have it"""
        for i, row in self.df.iterrows():
            if pd.isna(row.get('pinyin')) or not str(row.get('pinyin')).strip():
                chinese_text = str(row['zh']).strip()
                if chinese_text:
                    pinyin_result = pinyin(chinese_text, style=Style.TONE)
                    pinyin_text = ' '.join([''.join(p) for p in pinyin_result])
                    self.df.loc[i, 'pinyin'] = pinyin_text
    
    def update_display(self):
        """Update the flashcard display"""
        if self.df is None or len(self.df) == 0:
            return
        
        row = self.df.iloc[self.current_index]
        
        # Update card counter
        self.card_label.setText(f"{self.current_index + 1} / {len(self.df)}")
        
        # Set question and answer based on study direction
        if self.study_direction == "zh_to_eng":
            question = str(row['zh'])
            answer = str(row['eng'])
            # Show pinyin hint for Chinese text
            if self.pinyin_hints_enabled:
                pinyin_text = str(row.get('pinyin', ''))
                self.pinyin_hint.setText(pinyin_text)
                self.pinyin_hint.setVisible(True)
            else:
                self.pinyin_hint.setVisible(False)
        else:  # eng_to_zh
            question = str(row['eng'])
            answer = str(row['zh'])
            # Show pinyin hint for Chinese answer (when revealed)
            if self.pinyin_hints_enabled and self.show_answer:
                pinyin_text = str(row.get('pinyin', ''))
                self.pinyin_hint.setText(pinyin_text)
                self.pinyin_hint.setVisible(True)
            else:
                self.pinyin_hint.setVisible(False)
        
        # Update question text
        self.question_text.setPlainText(question)
        
        # Set appropriate font for Chinese text
        if self.study_direction == "zh_to_eng":
            self.question_text.setFont(QFont("Noto Sans CJK SC", 20))
        else:
            self.question_text.setFont(QFont("Arial", 18))
        
        # Update answer display
        if self.show_answer:
            self.answer_text.setPlainText(answer)
            self.answer_text.setVisible(True)
            self.show_answer_btn.setText("Hide Answer")
            
            # Set appropriate font for answer
            if self.study_direction == "eng_to_zh":
                self.answer_text.setFont(QFont("Noto Sans CJK SC", 18))
            else:
                self.answer_text.setFont(QFont("Arial", 16))
        else:
            self.answer_text.setVisible(False)
            self.show_answer_btn.setText("Show Answer")
    
    def toggle_answer(self):
        """Toggle answer visibility"""
        self.show_answer = not self.show_answer
        self.update_display()
    
    def record_score(self, score):
        """Record the user's pass/fail score (0 = fail, 1 = pass)"""
        if self.df is None:
            return
        
        # Determine which score column to update
        score_col = f"score_{self.study_direction}"
        attempts_col = f"attempts_{self.study_direction}"
        
        # Get current scores and attempts
        current_scores = self.df.iloc[self.current_index].get(score_col)
        current_attempts = self.df.iloc[self.current_index].get(attempts_col)
        
        # Parse existing scores (stored as comma-separated values)
        if pd.isna(current_scores) or not str(current_scores).strip():
            scores_list = []
        else:
            scores_list = [int(x) for x in str(current_scores).split(',') if x.strip()]
        
        if pd.isna(current_attempts):
            attempts = 0
        else:
            attempts = int(current_attempts)
        
        # Add new score
        scores_list.append(score)
        attempts += 1
        
        # Update dataframe
        self.df.loc[self.current_index, score_col] = ','.join(map(str, scores_list))
        self.df.loc[self.current_index, attempts_col] = attempts
        
        # Save progress
        self.save_csv()
        
        # Update statistics
        self.update_statistics()
        
        # Move to next card automatically
        QTimer.singleShot(500, self.next_card)  # Small delay for user feedback
        
        result_text = "Correct!" if score == 1 else "Need more practice"
        self.status_label.setText(f"{result_text}")
    
    def calculate_pass_rate(self, index, direction):
        """Calculate pass rate (percentage) for a specific card and direction"""
        score_col = f"score_{direction}"
        scores_str = self.df.iloc[index].get(score_col)
        
        if pd.isna(scores_str) or not str(scores_str).strip():
            return None
        
        try:
            scores = [int(x) for x in str(scores_str).split(',') if x.strip()]
            if not scores:
                return None
            
            pass_count = sum(scores)  # Count of 1s (passes)
            total_attempts = len(scores)
            return (pass_count / total_attempts) * 100
        except:
            return None
    
    def update_statistics(self):
        """Update the statistics display"""
        if self.df is None or len(self.df) == 0:
            return
        
        # Current card pass rate
        current_pass_rate = self.calculate_pass_rate(self.current_index, self.study_direction)
        if current_pass_rate is not None:
            self.current_avg_label.setText(f"Current Card: {current_pass_rate:.0f}% pass rate")
            # Color code based on performance
            if current_pass_rate >= 80:
                color = "#4CAF50"  # Green
            elif current_pass_rate >= 60:
                color = "#2196F3"  # Blue
            elif current_pass_rate >= 40:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
            self.current_avg_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        else:
            self.current_avg_label.setText("Current Card: No attempts yet")
            self.current_avg_label.setStyleSheet("")
        
        # Overall pass rate
        all_pass_rates = []
        cards_studied = 0
        total_passes = 0
        total_attempts = 0
        
        for i in range(len(self.df)):
            pass_rate = self.calculate_pass_rate(i, self.study_direction)
            if pass_rate is not None:
                cards_studied += 1
                
                # Get raw scores for overall calculation
                score_col = f"score_{self.study_direction}"
                scores_str = self.df.iloc[i].get(score_col)
                if not pd.isna(scores_str) and str(scores_str).strip():
                    scores = [int(x) for x in str(scores_str).split(',') if x.strip()]
                    total_passes += sum(scores)
                    total_attempts += len(scores)
        
        if total_attempts > 0:
            overall_pass_rate = (total_passes / total_attempts) * 100
            self.overall_avg_label.setText(f"Overall: {overall_pass_rate:.0f}% pass rate")
        else:
            self.overall_avg_label.setText("Overall: No attempts yet")
        
        self.progress_label.setText(f"Cards Studied: {cards_studied}/{len(self.df)}")
    
    def previous_card(self):
        """Go to previous card"""
        if self.df is None or len(self.df) == 0:
            return
        self.current_index = (self.current_index - 1) % len(self.df)
        self.show_answer = False
        self.update_display()
        self.update_statistics()
    
    def next_card(self):
        """Go to next card"""
        if self.df is None or len(self.df) == 0:
            return
        self.current_index = (self.current_index + 1) % len(self.df)
        self.show_answer = False
        self.update_display()
        self.update_statistics()
    
    def generate_audio(self):
        """Generate audio for current Chinese text"""
        if not SPARKTTS_AVAILABLE or self.spark_tts_model is None:
            QMessageBox.warning(self, "Warning", "Audio generation not available")
            return
        
        if self.df is None:
            return
        
        # Always generate audio for the Chinese text
        text = str(self.df.iloc[self.current_index]['zh'])
        
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
    
    def play_audio(self):
        """Play generated audio"""
        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            try:
                url = QUrl.fromLocalFile(os.path.abspath(self.temp_audio_path))
                self.media_player.setSource(url)
                self.media_player.play()
                self.status_label.setText("Playing audio...")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to play audio: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please generate audio first")
    
    def save_csv(self):
        """Save the current dataframe back to CSV"""
        if self.df is None or self.csv_file_path is None:
            return
        
        try:
            self.df.to_csv(self.csv_file_path, index=False)
            print(f"Progress saved to {self.csv_file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save progress: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application closing"""
        # Save progress before closing
        if self.df is not None and self.csv_file_path is not None:
            self.save_csv()
        
        # Clean up temporary audio files
        if hasattr(self, 'temp_audio_path') and self.temp_audio_path:
            try:
                if os.path.exists(self.temp_audio_path):
                    os.unlink(self.temp_audio_path)
            except:
                pass
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = MandarinFlashcardApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()