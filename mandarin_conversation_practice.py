import sys
import os
import pandas as pd
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from pypinyin import pinyin, Style
import tempfile
from difflib import SequenceMatcher
import torch
import unicodedata
import opencc
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QMessageBox, QScrollArea, QFrame, QCheckBox,
                            QSlider, QSpinBox, QGroupBox, QComboBox, QDialog,
                            QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QFont, QCursor
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
            
            sensevoice_model = None
            if SENSEVOICE_AVAILABLE:
                try:
                    self.progress.emit("Loading SenseVoice model...")
                    sensevoice_model = AutoModel(
                        model="iic/SenseVoiceSmall",
                        trust_remote_code=True,
                        remote_code="/home/richard/software/SenseVoice/model.py",
                        vad_model="fsmn-vad",
                        vad_kwargs={"max_single_segment_time": 30000},
                        device="cuda:0" if torch.cuda.is_available() else "cpu",
                        disable_update=True,
                    )
                except Exception as e:
                    print(f"Failed to load SenseVoice: {e}")
            
            self.finished.emit(True, "Models loaded successfully!")
            self.whisper_model = whisper_model
            self.spark_tts_model = spark_tts_model
            self.sensevoice_model = sensevoice_model
            
        except Exception as e:
            self.finished.emit(False, f"Error loading models: {str(e)}")

class AudioGenerator(QThread):
    finished = pyqtSignal(bool, str, str)
    
    def __init__(self, spark_tts_model, text, speed='moderate'):
        super().__init__()
        self.spark_tts_model = spark_tts_model
        self.text = text
        self.speed = speed
    
    def run(self):
        try:
            with torch.no_grad():
                wav = self.spark_tts_model.inference(
                    self.text,
                    "/home/richard/software/Spark-TTS/example/primsluer1.flac",
                    prompt_text=None,
                    gender='female',
                    pitch='moderate',
                    speed=self.speed
                )
            
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_audio_path, wav, 16000)
            self.finished.emit(True, "Audio generated", temp_audio_path)
            
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}", "")

class AudioTranscriber(QThread):
    finished = pyqtSignal(bool, str, str, str)
    
    def __init__(self, whisper_model, sensevoice_model, audio_data, sample_rate, selected_model):
        super().__init__()
        self.whisper_model = whisper_model
        self.sensevoice_model = sensevoice_model
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.selected_model = selected_model
    
    def preprocess_audio(self, audio_data):
        if len(audio_data) > 0:
            audio_data = audio_data - np.mean(audio_data)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data * (0.9 / max_val)
            noise_threshold = np.max(np.abs(audio_data)) * 0.01
            audio_data[np.abs(audio_data) < noise_threshold] *= 0.1
        return audio_data
    
    def run(self):
        try:
            processed_audio = self.preprocess_audio(self.audio_data)
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_audio_path, processed_audio, self.sample_rate)
            
            if self.selected_model == "sensevoice" and self.sensevoice_model is not None:
                res = self.sensevoice_model.generate(
                    input=temp_audio_path,
                    cache={},
                    language="zh",
                    use_itn=True,
                    batch_size_s=60,
                )
                transcribed_text = rich_transcription_postprocess(res[0]["text"]) if SENSEVOICE_AVAILABLE else res[0]["text"]
            else:
                result = self.whisper_model.transcribe(temp_audio_path, language="zh", task="transcribe", temperature=0.0)
                transcribed_text = result["text"].strip()
            
            pinyin_result = pinyin(transcribed_text, style=Style.TONE)
            pinyin_text = ' '.join([''.join(p) for p in pinyin_result])
            
            os.unlink(temp_audio_path)
            self.finished.emit(True, "Transcription complete", transcribed_text, pinyin_text)
            
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}", "", "")

class ChatBubble(QFrame):
    play_audio = pyqtSignal(str)
    
    def __init__(self, text, is_bot=True, audio_path=None, show_chinese=True, 
                 show_pinyin=True, show_english=True, pinyin_text="", english_text="", parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.is_bot = is_bot
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        if is_bot:
            layout.addStretch()
        
        bubble_frame = QFrame()
        bubble_frame.setObjectName("bubble")
        bubble_layout = QVBoxLayout(bubble_frame)
        bubble_layout.setSpacing(3)
        
        if audio_path:
            audio_btn = QPushButton("🔊")
            audio_btn.setMaximumSize(30, 30)
            audio_btn.clicked.connect(lambda: self.play_audio.emit(audio_path))
            audio_btn.setStyleSheet("background: transparent; border: none; font-size: 18px;")
            bubble_layout.addWidget(audio_btn)
        
        if show_chinese and text:
            chinese_label = QLabel(text)
            chinese_label.setFont(QFont("Noto Sans CJK SC", 14))
            chinese_label.setWordWrap(True)
            chinese_label.setStyleSheet("background: transparent;")
            bubble_layout.addWidget(chinese_label)
        
        if show_pinyin and pinyin_text:
            pinyin_label = QLabel(pinyin_text)
            pinyin_label.setFont(QFont("Consolas", 10))
            pinyin_label.setWordWrap(True)
            pinyin_label.setStyleSheet("color: #666; background: transparent;")
            bubble_layout.addWidget(pinyin_label)
        
        if show_english and english_text:
            eng_label = QLabel(english_text)
            eng_label.setFont(QFont("Arial", 10))
            eng_label.setWordWrap(True)
            eng_label.setStyleSheet("color: #888; font-style: italic; background: transparent;")
            bubble_layout.addWidget(eng_label)
        
        if is_bot:
            bubble_frame.setStyleSheet("""
                #bubble {
                    background-color: white;
                    border-radius: 10px;
                    padding: 10px;
                    border: 1px solid #e0e0e0;
                }
            """)
        else:
            bubble_frame.setStyleSheet("""
                #bubble {
                    background-color: #95ec69;
                    border-radius: 10px;
                    padding: 10px;
                }
            """)
        
        bubble_frame.setMaximumWidth(500)
        layout.addWidget(bubble_frame)
        
        if not is_bot:
            layout.addStretch()

class ConversationGuideDialog(QDialog):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.setWindowTitle("Conversation Guide")
        self.setModal(False)
        self.setMinimumSize(500, 600)
        self.df = df
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel("Right-click on any sentence to see pinyin hint")
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(info_label)
        
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
        """)
        
        self.populate_guide()
        layout.addWidget(self.list_widget)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        # Install event filter for right-click detection
        self.list_widget.viewport().installEventFilter(self)
        self.hint_label = None
    
    def populate_guide(self):
        self.list_widget.clear()
        if self.df is not None:
            for idx, row in self.df.iterrows():
                turn_id = row['turn_id']
                bot_eng = row.get('bot_eng', '')
                expected_eng = row.get('expected_eng', '')
                expected_zh = row.get('expected_response_zh', '')
                expected_pinyin = row.get('expected_pinyin', '')
                
                # Generate pinyin if not in CSV
                if not expected_pinyin or pd.isna(expected_pinyin):
                    pinyin_result = pinyin(expected_zh, style=Style.TONE)
                    expected_pinyin = ' '.join([''.join(p) for p in pinyin_result])
                
                item_text = f"Turn {turn_id}: {bot_eng}\n→ You say: {expected_eng}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, {
                    'chinese': expected_zh,
                    'pinyin': expected_pinyin,
                    'english': expected_eng
                })
                item.setFont(QFont("Arial", 11))
                self.list_widget.addItem(item)
    
    def eventFilter(self, obj, event):
        if obj == self.list_widget.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    item = self.list_widget.itemAt(event.pos())
                    if item:
                        self.show_pinyin_hint(item, event.globalPosition().toPoint())
                        return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.RightButton:
                    self.hide_pinyin_hint()
                    return True
        return super().eventFilter(obj, event)
    
    def show_pinyin_hint(self, item, pos):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            if self.hint_label is None:
                self.hint_label = QLabel(self)
                self.hint_label.setStyleSheet("""
                    QLabel {
                        background-color: #333;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-size: 12px;
                    }
                """)
                self.hint_label.setWindowFlags(Qt.WindowType.ToolTip)
            
            hint_text = f"{data['chinese']}\n{data['pinyin']}"
            self.hint_label.setText(hint_text)
            self.hint_label.adjustSize()
            self.hint_label.move(pos)
            self.hint_label.show()
    
    def hide_pinyin_hint(self):
        if self.hint_label:
            self.hint_label.hide()

class SettingsDialog(QDialog):
    def __init__(self, parent, show_chinese, show_pinyin, show_english, score_threshold, current_speed):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        hint_group = QGroupBox("Show Hints")
        hint_layout = QVBoxLayout(hint_group)
        
        self.chinese_check = QCheckBox("Chinese Characters")
        self.chinese_check.setChecked(show_chinese)
        hint_layout.addWidget(self.chinese_check)
        
        self.pinyin_check = QCheckBox("Pinyin")
        self.pinyin_check.setChecked(show_pinyin)
        hint_layout.addWidget(self.pinyin_check)
        
        self.english_check = QCheckBox("English Translation")
        self.english_check.setChecked(show_english)
        hint_layout.addWidget(self.english_check)
        
        layout.addWidget(hint_group)
        
        threshold_group = QGroupBox("Comprehension Threshold")
        threshold_layout = QHBoxLayout(threshold_group)
        threshold_layout.addWidget(QLabel("Minimum score to continue:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setValue(score_threshold)
        self.threshold_spin.setSuffix("%")
        threshold_layout.addWidget(self.threshold_spin)
        layout.addWidget(threshold_group)
        
        speed_group = QGroupBox("Speech Speed")
        speed_layout = QVBoxLayout(speed_group)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(4)
        self.speed_options = ['very_low', 'low', 'moderate', 'high', 'very_high']
        self.speed_slider.setValue(self.speed_options.index(current_speed))
        
        speed_labels = QHBoxLayout()
        speed_labels.addWidget(QLabel("Very Slow"))
        speed_labels.addStretch()
        speed_labels.addWidget(QLabel("Very Fast"))
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addLayout(speed_labels)
        layout.addWidget(speed_group)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
    
    def get_values(self):
        return (
            self.chinese_check.isChecked(),
            self.pinyin_check.isChecked(),
            self.english_check.isChecked(),
            self.threshold_spin.value(),
            self.speed_options[self.speed_slider.value()]
        )

class MeilingConversationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("美玲 (Měilíng) - Conversational Practice")
        self.setGeometry(100, 100, 900, 800)
        
        self.whisper_model = None
        self.spark_tts_model = None
        self.sensevoice_model = None
        self.df = None
        self.current_turn_id = None
        self.conversation_history = []
        self.is_recording = False
        self.recorded_audio = None
        self.sample_rate = 16000
        self.selected_device = None
        self.device_sample_rates = {}
        self.selected_asr_model = "whisper"
        self.current_speed = "moderate"
        self.score_threshold = 60
        
        self.show_chinese = True
        self.show_pinyin = True
        self.show_english = False
        
        try:
            self.converter = opencc.OpenCC('t2s')
        except:
            self.converter = None
        
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        self.confusion_responses = [
            "对不起，我没听懂。你能再说一遍吗？",
            "我不明白你说什么。",
            "什么？请再说一次。",
            "我听不清楚。",
        ]
        
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        top_bar = QHBoxLayout()
        
        load_btn = QPushButton("📁 Load Conversation CSV")
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #07c160;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #06ad56;
            }
        """)
        load_btn.clicked.connect(self.load_csv)
        top_bar.addWidget(load_btn)
        
        guide_btn = QPushButton("📖 Conversation Guide")
        guide_btn.setStyleSheet("""
            QPushButton {
                background-color: #1989fa;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1677d4;
            }
        """)
        guide_btn.clicked.connect(self.show_guide)
        top_bar.addWidget(guide_btn)
        
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333;
                font-size: 12px;
                padding: 8px 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        settings_btn.clicked.connect(self.show_settings)
        top_bar.addWidget(settings_btn)
        
        self.status_label = QLabel("Loading models...")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()
        
        main_layout.addLayout(top_bar)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none; background-color: #ededed;")
        
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()
        self.chat_layout.setSpacing(10)
        
        scroll_area.setWidget(self.chat_widget)
        main_layout.addWidget(scroll_area, stretch=1)
        
        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: white; border-top: 1px solid #ddd;")
        control_layout = QVBoxLayout(control_panel)
        
        record_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("🎤 Hold to Record")
        self.record_btn.setMinimumHeight(50)
        self.record_btn.pressed.connect(self.start_recording)
        self.record_btn.released.connect(self.stop_recording)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #07c160;
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:pressed {
                background-color: #06ad56;
            }
        """)
        record_layout.addWidget(self.record_btn)
        
        control_layout.addLayout(record_layout)
        
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Mic:"))
        self.mic_combo = QComboBox()
        self.populate_audio_devices()
        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)
        mic_layout.addWidget(self.mic_combo)
        
        mic_layout.addWidget(QLabel("ASR:"))
        self.asr_combo = QComboBox()
        self.asr_combo.addItems(["Whisper", "SenseVoice"])
        if not SENSEVOICE_AVAILABLE:
            self.asr_combo.model().item(1).setEnabled(False)
        self.asr_combo.currentTextChanged.connect(self.on_asr_changed)
        mic_layout.addWidget(self.asr_combo)
        
        control_layout.addLayout(mic_layout)
        
        main_layout.addWidget(control_panel)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ededed;
            }
            QPushButton {
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
        """)
    
    def show_settings(self):
        dialog = SettingsDialog(
            self,
            self.show_chinese,
            self.show_pinyin,
            self.show_english,
            self.score_threshold,
            self.current_speed
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            (self.show_chinese, self.show_pinyin, self.show_english,
             self.score_threshold, self.current_speed) = dialog.get_values()
            self.status_label.setText(f"Settings updated - Threshold: {self.score_threshold}%, Speed: {self.current_speed}")
    
    def show_guide(self):
        if self.df is None:
            QMessageBox.warning(self, "No Conversation", "Please load a conversation CSV first.")
            return
        
        guide_dialog = ConversationGuideDialog(self, self.df)
        guide_dialog.show()
    
    def populate_audio_devices(self):
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.mic_combo.addItem(device['name'], i)
            
            default_device = sd.query_devices(kind='input')
            self.selected_device = default_device.get('index', 0)
        except:
            self.mic_combo.addItem("No devices", None)
    
    def on_mic_changed(self):
        self.selected_device = self.mic_combo.currentData()
    
    def on_asr_changed(self, text):
        self.selected_asr_model = text.lower()
    
    def load_models(self):
        self.model_loader = ModelLoader()
        self.model_loader.progress.connect(lambda msg: self.status_label.setText(msg))
        self.model_loader.finished.connect(self.on_models_loaded)
        self.model_loader.start()
    
    def on_models_loaded(self, success, message):
        self.status_label.setText(message)
        if success:
            self.whisper_model = self.model_loader.whisper_model
            self.spark_tts_model = self.model_loader.spark_tts_model
            self.sensevoice_model = self.model_loader.sensevoice_model
    
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Conversation CSV", "", "CSV files (*.csv)")
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                required = ['turn_id', 'bot_zh', 'expected_response_zh']
                
                if not all(col in self.df.columns for col in required):
                    QMessageBox.critical(self, "Error", f"CSV must have: {required}")
                    return
                
                self.current_turn_id = self.df.iloc[0]['turn_id']
                self.conversation_history = []
                
                for i in reversed(range(self.chat_layout.count())):
                    widget = self.chat_layout.itemAt(i).widget()
                    if widget and not isinstance(widget, type(None)):
                        widget.deleteLater()
                
                self.chat_layout.addStretch()
                
                self.start_turn()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
    
    def start_turn(self):
        if self.df is None or self.current_turn_id is None:
            return
        
        turn = self.df[self.df['turn_id'] == self.current_turn_id].iloc[0]
        
        bot_text = turn['bot_zh']
        bot_pinyin = turn.get('bot_pinyin', '')
        bot_english = turn.get('bot_eng', '')
        
        if not bot_pinyin or pd.isna(bot_pinyin):
            pinyin_result = pinyin(bot_text, style=Style.TONE)
            bot_pinyin = ' '.join([''.join(p) for p in pinyin_result])
        
        self.generate_bot_audio(bot_text, bot_pinyin, bot_english)
    
    def generate_bot_audio(self, text, pinyin_text, english_text):
        self.audio_gen = AudioGenerator(self.spark_tts_model, text, self.current_speed)
        self.audio_gen.finished.connect(
            lambda success, msg, path: self.on_bot_audio_ready(success, path, text, pinyin_text, english_text)
        )
        self.audio_gen.start()
    
    def on_bot_audio_ready(self, success, audio_path, text, pinyin_text, english_text):
        if success:
            bubble = ChatBubble(
                text, is_bot=True, audio_path=audio_path,
                show_chinese=self.show_chinese,
                show_pinyin=self.show_pinyin,
                show_english=self.show_english,
                pinyin_text=pinyin_text,
                english_text=english_text
            )
            bubble.play_audio.connect(self.play_audio)
            
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
            
            # Auto-play all bot responses after a short delay
            QTimer.singleShot(500, lambda: self.play_audio(audio_path))
            
            self.conversation_history.append({
                'speaker': 'bot',
                'text': text,
                'audio': audio_path
            })
    
    def play_audio(self, audio_path):
        if os.path.exists(audio_path):
            url = QUrl.fromLocalFile(os.path.abspath(audio_path))
            self.media_player.setSource(url)
            self.media_player.play()
    
    def start_recording(self):
        if self.selected_device is None:
            return
        
        self.is_recording = True
        self.record_btn.setText("🔴 Recording...")
        self.recorded_audio = []
        
        sample_rate = self.get_supported_sample_rate(self.selected_device)
        
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.recorded_audio.append(indata.copy())
        
        try:
            self.stream = sd.InputStream(
                device=self.selected_device,
                samplerate=sample_rate,
                channels=1,
                callback=callback,
                dtype=np.float32
            )
            self.stream.start()
            self.current_recording_sample_rate = sample_rate
        except Exception as e:
            self.is_recording = False
            self.record_btn.setText("🎤 Hold to Record")
            QMessageBox.critical(self, "Error", f"Recording failed: {str(e)}")
    
    def stop_recording(self):
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.record_btn.setText("🎤 Hold to Record")
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.recorded_audio:
            self.recorded_audio = np.concatenate(self.recorded_audio, axis=0).flatten()
            self.transcribe_user_response()
    
    def transcribe_user_response(self):
        sample_rate = getattr(self, 'current_recording_sample_rate', self.sample_rate)
        
        self.transcriber = AudioTranscriber(
            self.whisper_model,
            self.sensevoice_model,
            self.recorded_audio,
            sample_rate,
            self.selected_asr_model
        )
        self.transcriber.finished.connect(self.on_user_transcribed)
        self.transcriber.start()
    
    def on_user_transcribed(self, success, message, transcription, pinyin_text):
        if not success:
            QMessageBox.warning(self, "Error", message)
            return
        
        bubble = ChatBubble(
            transcription, is_bot=False,
            show_chinese=True,
            show_pinyin=True,
            show_english=False,
            pinyin_text=pinyin_text
        )
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        
        self.conversation_history.append({
            'speaker': 'user',
            'text': transcription
        })
        
        self.evaluate_response(transcription)
    
    def evaluate_response(self, user_text):
        if self.df is None or self.current_turn_id is None:
            return
        
        turn = self.df[self.df['turn_id'] == self.current_turn_id].iloc[0]
        expected = turn['expected_response_zh']
        
        score = self.calculate_score(expected, user_text)
        
        print(f"Score: {score}% (threshold: {self.score_threshold}%)")
        self.status_label.setText(f"Score: {score}%")
        
        if score < self.score_threshold:
            confusion = np.random.choice(self.confusion_responses)
            pinyin_result = pinyin(confusion, style=Style.TONE)
            confusion_pinyin = ' '.join([''.join(p) for p in pinyin_result])
            self.generate_bot_audio(confusion, confusion_pinyin, "I don't understand. Can you repeat?")
        else:
            next_turn = turn.get('next_turn_id')
            if pd.notna(next_turn) and next_turn in self.df['turn_id'].values:
                self.current_turn_id = next_turn
                QTimer.singleShot(1000, self.start_turn)
            else:
                self.status_label.setText("Conversation complete! 很好！")
    
    def calculate_score(self, expected, response):
        # Handle placeholders like [名字], [name], etc.
        # Remove anything in square brackets from expected for matching
        import re
        expected_cleaned = re.sub(r'\[.*?\]', '', expected)
        
        norm_exp = self.normalize_text(expected_cleaned)
        norm_resp = self.normalize_text(response)
        
        # If expected is very short after removing placeholders, be very lenient
        if len(norm_exp) <= 3:
            # Just check if response contains the key phrase
            if norm_exp in norm_resp:
                return 100
            # Or if response shares most characters with expected
            exp_chars = set(norm_exp)
            resp_chars = set(norm_resp)
            if len(exp_chars) > 0:
                overlap = len(exp_chars & resp_chars) / len(exp_chars)
                if overlap >= 0.5:
                    return 100
        
        # Method 1: Direct character similarity
        char_sim = SequenceMatcher(None, norm_exp, norm_resp).ratio()
        
        # Method 2: Check if expected is contained in response (partial match)
        if norm_exp in norm_resp or norm_resp in norm_exp:
            containment_score = 1.0
        else:
            # Calculate how many expected characters are in the response
            exp_chars = set(norm_exp)
            resp_chars = set(norm_resp)
            overlap = len(exp_chars & resp_chars) / len(exp_chars) if len(exp_chars) > 0 else 0
            containment_score = overlap
        
        # Method 3: Pinyin similarity (pronunciation-based)
        exp_pinyin = ''.join([''.join(p) for p in pinyin(norm_exp, style=Style.NORMAL)])
        resp_pinyin = ''.join([''.join(p) for p in pinyin(norm_resp, style=Style.NORMAL)])
        pinyin_sim = SequenceMatcher(None, exp_pinyin, resp_pinyin).ratio()
        
        # Use the best score from different methods (more forgiving)
        best_score = max(
            char_sim * 0.7 + pinyin_sim * 0.3,  # Original method
            containment_score * 0.8 + pinyin_sim * 0.2  # Containment method
        )
        
        score = int(best_score * 100)
        return score
    
    def normalize_text(self, text):
        if not text:
            return text
        normalized = ''.join(char for char in text if char.strip() and not unicodedata.category(char).startswith('P'))
        if self.converter:
            try:
                normalized = self.converter.convert(normalized)
            except:
                pass
        return unicodedata.normalize('NFC', normalized)
    
    def get_supported_sample_rate(self, device_id):
        if device_id in self.device_sample_rates:
            return self.device_sample_rates[device_id]
        
        for rate in [44100, 48000, 16000, 22050]:
            try:
                test = sd.InputStream(device=device_id, samplerate=rate, channels=1, dtype=np.float32)
                test.start()
                sd.sleep(50)
                test.stop()
                test.close()
                self.device_sample_rates[device_id] = rate
                return rate
            except:
                continue
        
        self.device_sample_rates[device_id] = 44100
        return 44100

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MeilingConversationApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()