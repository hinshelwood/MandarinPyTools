import sys
import os
import pandas as pd
import numpy as np
import tempfile
import torch
from datetime import datetime, timedelta
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                            QFileDialog, QMessageBox, QGroupBox, QCheckBox,
                            QButtonGroup, QRadioButton, QSpacerItem, QSizePolicy,
                            QProgressBar, QLineEdit, QSlider, QComboBox)
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
        self.setWindowTitle("Mandarin Flashcard Study Tool - SRS Edition")
        self.setGeometry(100, 100, 1200, 900)
        
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
        self.study_mode = "due_only"  # "due_only", "focus_level", "all"
        
        # SRS settings
        self.new_cards_per_day = 10
        self.mastery_threshold = 0.80
        self.min_reviews_for_mastery = 5
        
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
        self.file_path_edit.setPlaceholderText("Select CSV file with columns: zh, pinyin, eng, hsk_level, theme")
        file_layout.addWidget(self.file_path_edit)
        
        browse_btn = QPushButton("Browse CSV")
        browse_btn.clicked.connect(self.load_csv)
        file_layout.addWidget(browse_btn)
        
        main_layout.addWidget(file_group)
        
        # SRS Info Dashboard
        srs_info_group = QGroupBox("Spaced Repetition Status")
        srs_info_layout = QHBoxLayout(srs_info_group)
        
        self.focus_level_label = QLabel("Focus Level: N/A")
        self.focus_level_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.focus_level_label.setStyleSheet("color: #2196F3;")
        srs_info_layout.addWidget(self.focus_level_label)
        
        self.due_today_label = QLabel("Due Today: 0")
        self.due_today_label.setFont(QFont("Arial", 11))
        srs_info_layout.addWidget(self.due_today_label)
        
        self.new_today_label = QLabel("New Today: 0/10")
        self.new_today_label.setFont(QFont("Arial", 11))
        srs_info_layout.addWidget(self.new_today_label)
        
        self.mastery_progress_label = QLabel("Focus Mastery: 0%")
        self.mastery_progress_label.setFont(QFont("Arial", 11))
        srs_info_layout.addWidget(self.mastery_progress_label)
        
        main_layout.addWidget(srs_info_group)
        
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
        
        # Study mode settings
        mode_group = QGroupBox("Study Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Due Cards + New (Recommended)",
            "Focus Level Only", 
            "All Cards (Manual Review)"
        ])
        self.mode_combo.currentIndexChanged.connect(self.on_study_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        
        new_cards_layout = QHBoxLayout()
        new_cards_layout.addWidget(QLabel("New cards/day:"))
        self.new_cards_spinbox = QComboBox()
        self.new_cards_spinbox.addItems(['5', '10', '15', '20', '25'])
        self.new_cards_spinbox.setCurrentText('10')
        self.new_cards_spinbox.currentTextChanged.connect(self.on_new_cards_changed)
        new_cards_layout.addWidget(self.new_cards_spinbox)
        mode_layout.addLayout(new_cards_layout)
        
        settings_layout.addWidget(mode_group)
        
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
        
        # Card metadata (HSK level, theme, review info)
        self.card_metadata = QLabel("")
        self.card_metadata.setFont(QFont("Arial", 9))
        self.card_metadata.setStyleSheet("color: #666; font-style: italic;")
        self.card_metadata.setAlignment(Qt.AlignmentFlag.AlignCenter)
        question_layout.addWidget(self.card_metadata)
        
        # Two-panel display for Chinese and Pinyin
        question_display_layout = QHBoxLayout()
        
        # Left panel: Chinese characters
        # self.question_text = QTextEdit()
        self.question_text = QLabel() # manual_debug

        self.question_text.setFont(QFont("Noto Sans CJK SC", 24))
        self.question_text.setMinimumHeight(250)
        self.question_text.setMaximumHeight(300)
        #self.question_text.setReadOnly(True)
        self.question_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.question_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #self.question_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        question_display_layout.addWidget(self.question_text)
        
        # Right panel: Pinyin
        #self.pinyin_text = QTextEdit()
        self.pinyin_text = QLabel()
        self.pinyin_text.setFont(QFont("Consolas", 24))
        self.pinyin_text.setMinimumHeight(250)
        self.pinyin_text.setMaximumHeight(300)
        #self.pinyin_text.setReadOnly(True)
        self.pinyin_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.pinyin_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #self.pinyin_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pinyin_text.setStyleSheet("color: #555; background-color: #f9f9f9;")
        question_display_layout.addWidget(self.pinyin_text)
        
        question_layout.addLayout(question_display_layout)
        
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
        
        #self.answer_text = QTextEdit()
        self.answer_text = QLabel()
        self.answer_text.setFont(QFont("Arial", 64))
        self.answer_text.setMinimumHeight(200)
        self.answer_text.setMaximumHeight(250)
        #self.answer_text.setReadOnly(True)
        self.answer_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.answer_text.setVisible(False)  # Initially hidden
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
        
        # SRS interval info
        self.interval_info = QLabel("")
        self.interval_info.setFont(QFont("Arial", 9))
        self.interval_info.setStyleSheet("color: #666; font-style: italic;")
        self.interval_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scoring_layout.addWidget(self.interval_info)
        
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
        
        self.current_avg_label = QLabel("Current Card: N/A")
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
            QComboBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
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
    
    def on_study_mode_changed(self, index):
        """Handle study mode changes"""
        modes = ["due_only", "focus_level", "all"]
        self.study_mode = modes[index]
        self.update_study_set()
    
    def on_new_cards_changed(self, value):
        """Handle new cards per day setting change"""
        self.new_cards_per_day = int(value)
        self.update_srs_dashboard()
    
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
                
                # Add HSK level column if it doesn't exist
                if 'hsk_level' not in self.df.columns:
                    self.df['hsk_level'] = 1  # Default to HSK 1
                
                # Add theme column if it doesn't exist
                if 'theme' not in self.df.columns:
                    self.df['theme'] = 'general'
                
                # Initialize SRS columns
                srs_columns = {
                    'date_added': datetime.now().strftime('%Y-%m-%d'),
                    'date_last_reviewed': None,
                    'review_interval_days': 1.0,
                    'ease_factor': 2.5,
                    'next_review_date': None,
                    'times_reviewed': 0,
                    'times_correct': 0,
                    'studied_today': 0
                }
                
                for col, default in srs_columns.items():
                    if col not in self.df.columns:
                        self.df[col] = default
                
                # Initialize score tracking columns (legacy compatibility)
                score_columns = ['score_zh_to_eng', 'score_eng_to_zh', 
                               'attempts_zh_to_eng', 'attempts_eng_to_zh']
                for col in score_columns:
                    if col not in self.df.columns:
                        self.df[col] = None
                
                # Generate missing pinyin
                self.generate_missing_pinyin()
                
                self.file_path_edit.setText(file_path)
                self.csv_file_path = file_path
                
                # Update study set based on SRS
                self.update_study_set()
                
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
    
    def get_current_focus_level(self):
        """Determine which HSK level to draw new cards from"""
        if self.df is None:
            return 1
        
        for level in [1, 2, 3, 4, 5, 6]:
            level_cards = self.df[self.df['hsk_level'] == level]
            
            if len(level_cards) == 0:
                continue
            
            # Calculate mastery: cards with >= min_reviews and >= mastery_threshold pass rate
            mastered = level_cards[
                (level_cards['times_reviewed'] >= self.min_reviews_for_mastery) &
                (level_cards['times_correct'] / level_cards['times_reviewed'].replace(0, 1) >= self.mastery_threshold)
            ]
            
            mastery_rate = len(mastered) / len(level_cards) if len(level_cards) > 0 else 0
            
            # If less than 80% of this level is mastered, focus here
            if mastery_rate < 0.80:
                return level
        
        # If all levels mastered, continue with HSK 6
        return 6
    
    def get_focus_level_mastery(self):
        """Get mastery percentage for current focus level"""
        focus_level = self.get_current_focus_level()
        level_cards = self.df[self.df['hsk_level'] == focus_level]
        
        if len(level_cards) == 0:
            return 0, 0, 0
        
        mastered = level_cards[
            (level_cards['times_reviewed'] >= self.min_reviews_for_mastery) &
            (level_cards['times_correct'] / level_cards['times_reviewed'].replace(0, 1) >= self.mastery_threshold)
        ]
        
        mastery_rate = (len(mastered) / len(level_cards)) * 100
        return mastery_rate, len(mastered), len(level_cards)
    
    def update_study_set(self):
        """Update the current study set based on SRS algorithm and study mode"""
        if self.df is None:
            return
        
        today = pd.Timestamp.now().normalize()
        
        if self.study_mode == "due_only":
            # Get due cards (those that need review today)
            due_mask = (
                self.df['next_review_date'].isna() |
                (pd.to_datetime(self.df['next_review_date'], errors='coerce') <= today)
            )
            due_cards = self.df[due_mask].copy()
            
            # Add new cards from focus level
            focus_level = self.get_current_focus_level()
            never_studied = (self.df['times_reviewed'] == 0) | self.df['times_reviewed'].isna()
            new_from_focus = self.df[
                never_studied & 
                (self.df['hsk_level'] == focus_level) &
                (self.df['studied_today'] == 0)
            ].head(self.new_cards_per_day).copy()
            
            # Combine
            study_set = pd.concat([due_cards, new_from_focus]).drop_duplicates()
            
        elif self.study_mode == "focus_level":
            # Only cards from focus level
            focus_level = self.get_current_focus_level()
            study_set = self.df[self.df['hsk_level'] == focus_level].copy()
            
        else:  # "all"
            # All cards for manual review
            study_set = self.df.copy()
        
        # Shuffle and update indices
        if len(study_set) > 0:
            study_set = study_set.sample(frac=1).reset_index(drop=True)
            # Store original indices for saving back to main df
            study_set['original_index'] = study_set.index
            self.study_df = study_set
            self.current_index = 0
            self.show_answer = False
            self.update_display()
            self.update_statistics()
            self.update_srs_dashboard()
        else:
            self.study_df = None
            self.status_label.setText("No cards to study! Great job! 🎉")
    
    def update_srs_dashboard(self):
        """Update the SRS information dashboard"""
        if self.df is None:
            return
        
        today = pd.Timestamp.now().normalize()
        
        # Focus level
        focus_level = self.get_current_focus_level()
        self.focus_level_label.setText(f"Focus Level: HSK {focus_level}")
        
        # Due cards count
        due_mask = (
            self.df['next_review_date'].isna() |
            (pd.to_datetime(self.df['next_review_date'], errors='coerce') <= today)
        )
        due_count = due_mask.sum()
        self.due_today_label.setText(f"Due Today: {due_count}")
        
        # New cards studied today / target
        studied_today = (self.df['studied_today'] == 1).sum()
        self.new_today_label.setText(f"New Today: {studied_today}/{self.new_cards_per_day}")
        
        # Mastery progress for focus level
        mastery_pct, mastered_count, total_count = self.get_focus_level_mastery()
        self.mastery_progress_label.setText(f"HSK {focus_level} Mastery: {mastered_count}/{total_count} ({mastery_pct:.0f}%)")
    
    def update_display(self):
        """Update the flashcard display"""
        if self.study_df is None or len(self.study_df) == 0:
            return
        
        row = self.study_df.iloc[self.current_index]
        
        # Update card counter
        self.card_label.setText(f"{self.current_index + 1} / {len(self.study_df)}")
        
        # Display card metadata
        hsk_level = row.get('hsk_level', 'N/A')
        theme = row.get('theme', 'general')
        times_reviewed = row.get('times_reviewed', 0)
        times_correct = row.get('times_correct', 0)
        
        if times_reviewed > 0:
            accuracy = (times_correct / times_reviewed) * 100
            self.card_metadata.setText(f"HSK {hsk_level} | {theme} | Reviewed {times_reviewed}x ({accuracy:.0f}% accuracy)")
        else:
            self.card_metadata.setText(f"HSK {hsk_level} | {theme} | NEW CARD")
        
        # Set question and answer based on study direction
        if self.study_direction == "zh_to_eng":
            question = str(row['zh'])
            answer = str(row['eng'])
            pinyin_display = str(row.get('pinyin', ''))
            
            # Display Chinese in left panel, Pinyin in right panel
            #self.question_text.setPlainText(question)
            self.question_text.setText(question)
            self.question_text.setFont(QFont("Noto Sans CJK SC", 36))
            
            if self.pinyin_hints_enabled:
                #self.pinyin_text.setPlainText(pinyin_display)
                self.pinyin_text.setText(pinyin_display)
                self.pinyin_text.setFont(QFont("Consolas", 36))
                self.pinyin_text.setVisible(True)
            else:
                self.pinyin_text.setVisible(False)
                
        else:  # eng_to_zh
            question = str(row['eng'])
            answer = str(row['zh'])
            pinyin_display = str(row.get('pinyin', ''))
            
            # For English to Chinese, show English in main area
            #self.question_text.setPlainText(question)
            self.question_text.setText(question)
            self.question_text.setFont(QFont("Arial", 36))
            
            # Show pinyin hint when answer is revealed
            if self.pinyin_hints_enabled and self.show_answer:
                #self.pinyin_text.setPlainText(pinyin_display)
                self.pinyin_text.setText(pinyin_display)
                self.pinyin_text.setFont(QFont("Consolas", 36))
                self.pinyin_text.setVisible(True)
            else:
                self.pinyin_text.setVisible(False)
        
        # Update answer display
        if self.show_answer:
            #self.answer_text.setPlainText(answer)
            self.answer_text.setText(answer)
            self.answer_text.setVisible(True)
            self.show_answer_btn.setText("Hide Answer")
            
            # Set appropriate font for answer with larger size
            if self.study_direction == "eng_to_zh":
                self.answer_text.setFont(QFont("Noto Sans CJK SC", 36))
            else:
                self.answer_text.setFont(QFont("Arial", 36))
        else:
            self.answer_text.setVisible(False)
            self.show_answer_btn.setText("Show Answer")
        
        # Update interval info
        interval = row.get('review_interval_days', 1)
        next_review = row.get('next_review_date')
        if pd.notna(next_review):
            self.interval_info.setText(f"Current interval: {interval:.1f} days | Next review: {next_review}")
        else:
            self.interval_info.setText(f"Current interval: {interval:.1f} days")
    
    def toggle_answer(self):
        """Toggle answer visibility"""
        self.show_answer = not self.show_answer
        self.update_display()
    
    def record_score(self, score):
        """Record the user's pass/fail score using SRS algorithm"""
        if self.study_df is None or len(self.study_df) == 0:
            return
        
        today = pd.Timestamp.now().normalize()
        
        # Get current card from study set
        study_idx = self.current_index
        original_idx = self.study_df.iloc[study_idx].name
        
        # Get current SRS values
        old_interval = float(self.df.loc[original_idx, 'review_interval_days'])
        old_ease = float(self.df.loc[original_idx, 'ease_factor'])
        times_reviewed = int(self.df.loc[original_idx, 'times_reviewed']) if pd.notna(self.df.loc[original_idx, 'times_reviewed']) else 0
        times_correct = int(self.df.loc[original_idx, 'times_correct']) if pd.notna(self.df.loc[original_idx, 'times_correct']) else 0
        
        # Update review counts
        times_reviewed += 1
        if score == 1:
            times_correct += 1
        
        # Calculate new interval and ease using SM-2 algorithm
        if score == 1:  # Pass
            if times_reviewed == 1:
                new_interval = 1
            elif times_reviewed == 2:
                new_interval = 6
            else:
                new_interval = old_interval * old_ease
            
            new_ease = min(old_ease + 0.1, 3.0)  # Cap at 3.0
            
        else:  # Fail
            new_interval = 1
            new_ease = max(old_ease - 0.2, 1.3)  # Floor at 1.3
        
        # Calculate next review date
        next_review = today + pd.Timedelta(days=new_interval)
        
        # Update main dataframe
        self.df.loc[original_idx, 'times_reviewed'] = times_reviewed
        self.df.loc[original_idx, 'times_correct'] = times_correct
        self.df.loc[original_idx, 'review_interval_days'] = new_interval
        self.df.loc[original_idx, 'ease_factor'] = new_ease
        self.df.loc[original_idx, 'date_last_reviewed'] = today.strftime('%Y-%m-%d')
        self.df.loc[original_idx, 'next_review_date'] = next_review.strftime('%Y-%m-%d')
        
        # Mark as studied today if it was a new card
        if times_reviewed == 1:
            self.df.loc[original_idx, 'studied_today'] = 1
        
        # Also update legacy score tracking for compatibility
        score_col = f"score_{self.study_direction}"
        attempts_col = f"attempts_{self.study_direction}"
        
        current_scores = self.df.loc[original_idx, score_col]
        if pd.isna(current_scores) or not str(current_scores).strip():
            scores_list = []
        else:
            scores_list = [int(float(x)) for x in str(current_scores).split(',') if x.strip()]
        
        scores_list.append(score)
        self.df.loc[original_idx, score_col] = ','.join(map(str, scores_list))
        self.df.loc[original_idx, attempts_col] = times_reviewed
        
        # Save progress
        self.save_csv()
        
        # Update statistics
        self.update_statistics()
        self.update_srs_dashboard()
        
        # Provide feedback
        result_text = "Correct! ✓" if score == 1 else "Need more practice ✗"
        if score == 1:
            self.status_label.setText(f"{result_text} | Next review in {new_interval:.0f} days")
        else:
            self.status_label.setText(f"{result_text} | Will review again in 1 day")
        
        # Move to next card automatically
        QTimer.singleShot(500, self.next_card)
    
    def calculate_pass_rate(self, index):
        """Calculate pass rate (percentage) for a specific card"""
        times_reviewed = self.df.loc[index, 'times_reviewed']
        times_correct = self.df.loc[index, 'times_correct']
        
        if pd.isna(times_reviewed) or times_reviewed == 0:
            return None
        
        return (times_correct / times_reviewed) * 100
    
    def update_statistics(self):
        """Update the statistics display"""
        if self.df is None or len(self.df) == 0:
            return
        
        if self.study_df is None or len(self.study_df) == 0:
            return
        
        # Current card pass rate
        study_idx = self.current_index
        original_idx = self.study_df.iloc[study_idx].name
        
        current_pass_rate = self.calculate_pass_rate(original_idx)
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
        cards_studied = (self.df['times_reviewed'] > 0).sum()
        total_attempts = self.df['times_reviewed'].sum()
        total_correct = self.df['times_correct'].sum()
        
        if total_attempts > 0:
            overall_pass_rate = (total_correct / total_attempts) * 100
            self.overall_avg_label.setText(f"Overall: {overall_pass_rate:.0f}% pass rate")
        else:
            self.overall_avg_label.setText("Overall: No attempts yet")
        
        self.progress_label.setText(f"Cards Studied: {cards_studied}/{len(self.df)}")
    
    def previous_card(self):
        """Go to previous card"""
        if self.study_df is None or len(self.study_df) == 0:
            return
        self.current_index = (self.current_index - 1) % len(self.study_df)
        self.show_answer = False
        self.update_display()
        self.update_statistics()
    
    def next_card(self):
        """Go to next card"""
        if self.study_df is None or len(self.study_df) == 0:
            return
        self.current_index = (self.current_index + 1) % len(self.study_df)
        self.show_answer = False
        self.update_display()
        self.update_statistics()
    
    def generate_audio(self):
        """Generate audio for current Chinese text"""
        if not SPARKTTS_AVAILABLE or self.spark_tts_model is None:
            QMessageBox.warning(self, "Warning", "Audio generation not available")
            return
        
        if self.study_df is None or len(self.study_df) == 0:
            return
        
        # Always generate audio for the Chinese text
        study_idx = self.current_index
        #original_idx = self.study_df.iloc[study_idx].name
        #text = str(self.df.loc[original_idx, 'zh'])
        text = str(self.study_df.iloc[study_idx]['zh'])
        
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
    
    def reset_daily_flags(self):
        """Reset daily flags (call this at the start of a new day)"""
        if self.df is not None:
            self.df['studied_today'] = 0
            self.save_csv()
    
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