import sys
import os
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from pypinyin import pinyin, Style
import tempfile
import torch
import unicodedata
import opencc
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QMessageBox, QScrollArea, QFrame, QCheckBox,
                            QSlider, QGroupBox, QComboBox, QDialog, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl

# Anthropic API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")

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

CONFIG_DIR = Path.home() / ".meiling"
CONFIG_FILE = CONFIG_DIR / "config.json"

def load_config():
    """Load configuration from ~/.meiling/config.json"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    """Save configuration to ~/.meiling/config.json"""
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

class APIKeyDialog(QDialog):
    def __init__(self, parent, current_key=""):
        super().__init__(parent)
        self.setWindowTitle("Claude API Configuration")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        info = QLabel(
            "Enter your Claude API key for free conversation mode.\n\n"
            "Get your API key from: console.anthropic.com\n"
            "Cost: ~$0.12-$4/month for daily practice"
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info)
        
        layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-ant-api03-...")
        self.api_key_input.setText(current_key)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.api_key_input)
        
        show_key_check = QCheckBox("Show API key")
        show_key_check.toggled.connect(
            lambda checked: self.api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        layout.addWidget(show_key_check)
        
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "claude-3-5-haiku-20241022 (Cheapest - $0.12/month)",
            "claude-3-5-sonnet-20241022 (Balanced - $1-4/month)"
        ])
        model_layout.addWidget(QLabel("Choose model:"))
        model_layout.addWidget(self.model_combo)
        layout.addWidget(model_group)
        
        # HSK Level Selection
        hsk_group = QGroupBox("Language Level")
        hsk_layout = QVBoxLayout(hsk_group)
        
        self.hsk_combo = QComboBox()
        self.hsk_combo.addItems([
            "HSK 1 - Absolute Beginner (~150 words)",
            "HSK 2 - Beginner (~300 words)",
            "HSK 3 - Elementary (~600 words)",
            "HSK 4 - Intermediate (~1200 words)",
            "HSK 5 - Upper Intermediate (~2500 words)",
            "HSK 6 - Advanced (~5000 words)"
        ])
        self.hsk_combo.setCurrentIndex(1)  # Default to HSK 2
        hsk_layout.addWidget(QLabel("Choose difficulty:"))
        hsk_layout.addWidget(self.hsk_combo)
        layout.addWidget(hsk_group)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
    
    def get_config(self):
        model_text = self.model_combo.currentText()
        model = "claude-3-5-haiku-20241022" if "haiku" in model_text.lower() else "claude-3-5-sonnet-20241022"
        
        hsk_text = self.hsk_combo.currentText()
        hsk_level = int(hsk_text.split()[1])  # Extract number from "HSK X - ..."
        
        return {
            'api_key': self.api_key_input.text().strip(),
            'model': model,
            'hsk_level': hsk_level
        }

class TranslationThread(QThread):
    finished = pyqtSignal(bool, str)
    
    def __init__(self, api_key, model, chinese_text):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.chinese_text = chinese_text
    
    def run(self):
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": f"Translate this Chinese to natural English: {self.chinese_text}"
                }]
            )
            
            translation = response.content[0].text
            self.finished.emit(True, translation)
            
        except Exception as e:
            self.finished.emit(False, f"Translation error: {str(e)}")

class ClaudeConversationThread(QThread):
    finished = pyqtSignal(bool, str, str)
    
    def __init__(self, api_key, model, conversation_history, user_message, hsk_level=2):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.conversation_history = conversation_history
        self.user_message = user_message
        self.hsk_level = hsk_level
    
    def run(self):
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # HSK-specific vocabulary and complexity guidelines
            hsk_guidelines = {
                1: "HSK 1 vocabulary only (~150 words). Use very simple patterns like: 你好, 这是, 我喜欢, 多少钱. Keep responses to 3-5 words maximum.",
                2: "HSK 1-2 vocabulary (~300 words). Simple sentences with 你想, 我觉得, 在哪儿. Maximum 8 words per response.",
                3: "HSK 1-3 vocabulary (~600 words). Use simple connectors like 因为, 所以, 但是. Maximum 12 words per response.",
                4: "HSK 1-4 vocabulary (~1200 words). Can use 虽然, 不但, 而且. Keep responses under 15 words.",
                5: "HSK 1-5 vocabulary (~2500 words). Natural but clear speech. Maximum 20 words per response.",
                6: "HSK 1-6 vocabulary (~5000 words). Natural conversational Chinese. Maximum 25 words per response."
            }
            
            system_prompt = f"""你是美玲，一个友好的中文会话伙伴。你在帮助学生练习说中文。

CRITICAL RULES:
- {hsk_guidelines.get(self.hsk_level, hsk_guidelines[2])}
- 像朋友一样自然聊天，不要当老师讲课
- 如果学生问你吃饭，就说"好啊"或"不饿"，不要解释你是AI
- 不要列出清单或选项，只是正常聊天
- 保持话题轻松有趣
- 用简短的回应，像微信聊天一样
- 温柔地纠正大错误，但不要每次都纠正
- 如果学生说英语，用简单中文回应

TONE EXAMPLES (HSK {self.hsk_level}):
Student: "你想吃饭吗？" 
✅ Good: "好啊！你想吃什么？"
❌ Bad: "我很高兴你邀请我聊吃东西！作为AI助手，我不能真正吃东西，但我很乐意和你聊聊美食。中国有很多美味的食物，比如：北京烤鸭..."

Student: "你喜欢什么？"
✅ Good: "我喜欢看书。你呢？"
❌ Bad: "这是一个很好的问题！我喜欢很多东西。让我给你列举几个..."

REMEMBER: 短回答！自然！不要解释！"""
            
            messages = []
            for msg in self.conversation_history:
                if msg['speaker'] == 'user':
                    messages.append({"role": "user", "content": msg['text']})
                elif msg['speaker'] == 'bot':
                    messages.append({"role": "assistant", "content": msg['text']})
            
            messages.append({"role": "user", "content": self.user_message})
            
            response = client.messages.create(
                model=self.model,
                max_tokens=100,  # Reduced from 200 to encourage brevity
                system=system_prompt,
                messages=messages,
                temperature=0.8  # Higher temperature for more natural conversation
            )
            
            response_text = response.content[0].text
            self.finished.emit(True, "Response received", response_text)
            
        except Exception as e:
            self.finished.emit(False, f"API Error: {str(e)}", "")

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
    translate_requested = pyqtSignal(str, object)  # text, bubble_widget
    
    def __init__(self, text, is_bot=True, audio_path=None, show_chinese=True, 
                 show_pinyin=True, pinyin_text="", parent=None, allow_translation=False):
        super().__init__(parent)
        self.audio_path = audio_path
        self.is_bot = is_bot
        self.chinese_text = text
        self.allow_translation = allow_translation
        self.translation_label = None
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        if is_bot:
            layout.addStretch()
        
        bubble_frame = QFrame()
        bubble_frame.setObjectName("bubble")
        self.bubble_layout = QVBoxLayout(bubble_frame)
        self.bubble_layout.setSpacing(3)
        
        if audio_path:
            audio_btn = QPushButton("🔊")
            audio_btn.setMaximumSize(30, 30)
            audio_btn.clicked.connect(lambda: self.play_audio.emit(audio_path))
            audio_btn.setStyleSheet("background: transparent; border: none; font-size: 18px;")
            self.bubble_layout.addWidget(audio_btn)
        
        if show_chinese and text:
            chinese_label = QLabel(text)
            chinese_label.setFont(QFont("Noto Sans CJK SC", 14))
            chinese_label.setWordWrap(True)
            chinese_label.setStyleSheet("background: transparent;")
            self.bubble_layout.addWidget(chinese_label)
        
        if show_pinyin and pinyin_text:
            pinyin_label = QLabel(pinyin_text)
            pinyin_label.setFont(QFont("Consolas", 10))
            pinyin_label.setWordWrap(True)
            pinyin_label.setStyleSheet("color: #666; background: transparent;")
            self.bubble_layout.addWidget(pinyin_label)
        
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
        
        # Install event filter for right-click if translation allowed
        if allow_translation:
            bubble_frame.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        if self.allow_translation:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self.translate_requested.emit(self.chinese_text, self)
                    return True
        return super().eventFilter(obj, event)
    
    def show_translation(self, translation_text):
        if self.translation_label is None:
            self.translation_label = QLabel(translation_text)
            self.translation_label.setFont(QFont("Arial", 10))
            self.translation_label.setWordWrap(True)
            self.translation_label.setStyleSheet("color: #0066cc; font-style: italic; background: transparent; margin-top: 5px;")
            self.bubble_layout.addWidget(self.translation_label)
        else:
            self.translation_label.setText(translation_text)
            self.translation_label.show()

class SettingsDialog(QDialog):
    def __init__(self, parent, show_chinese, show_pinyin, current_speed):
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
        
        layout.addWidget(hint_group)
        
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
            self.speed_options[self.speed_slider.value()]
        )

class MeilingConversationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("美玲 (Měilíng) - Free Conversation")
        self.setGeometry(100, 100, 900, 800)
        
        self.whisper_model = None
        self.spark_tts_model = None
        self.sensevoice_model = None
        self.conversation_history = []
        self.is_recording = False
        self.recorded_audio = None
        self.sample_rate = 16000
        self.selected_device = None
        self.device_sample_rates = {}
        self.selected_asr_model = "whisper"
        self.current_speed = "moderate"
        
        self.show_chinese = True
        self.show_pinyin = True
        
        self.config = load_config()
        
        try:
            self.converter = opencc.OpenCC('t2s')
        except:
            self.converter = None
        
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top bar
        top_bar = QHBoxLayout()
        
        api_config_btn = QPushButton("🔑 API Configuration")
        api_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ee5a52;
            }
        """)
        api_config_btn.clicked.connect(self.show_api_config)
        top_bar.addWidget(api_config_btn)
        
        start_conv_btn = QPushButton("💬 Start New Conversation")
        start_conv_btn.setStyleSheet("""
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
        start_conv_btn.clicked.connect(self.start_conversation)
        top_bar.addWidget(start_conv_btn)
        
        top_bar.addStretch()
        
        settings_btn = QPushButton("⚙")
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333;
                font-size: 16px;
                padding: 8px 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        settings_btn.clicked.connect(self.show_settings)
        top_bar.addWidget(settings_btn)
        
        main_layout.addLayout(top_bar)
        
        # Status label
        self.status_label = QLabel("Loading models...")
        self.status_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        main_layout.addWidget(self.status_label)
        
        # Chat area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none; background-color: #ededed;")
        
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()
        self.chat_layout.setSpacing(10)
        
        scroll_area.setWidget(self.chat_widget)
        main_layout.addWidget(scroll_area, stretch=1)
        
        # Control panel
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
        
        # Microphone and ASR selection
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
        
        # Global styles
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
    
    def show_api_config(self):
        if not ANTHROPIC_AVAILABLE:
            QMessageBox.warning(
                self, 
                "Package Missing", 
                "The 'anthropic' package is not installed.\n\nInstall it with:\npip install anthropic"
            )
            return
        
        current_key = self.config.get('api_key', '')
        dialog = APIKeyDialog(self, current_key)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_config = dialog.get_config()
            self.config.update(new_config)
            save_config(self.config)
            self.status_label.setText(f"API configured - Model: {new_config['model'].split('-')[3]}")
    
    def start_conversation(self):
        if not ANTHROPIC_AVAILABLE:
            QMessageBox.warning(self, "Package Missing", "Install anthropic package first:\npip install anthropic")
            return
        
        if not self.config.get('api_key'):
            reply = QMessageBox.question(
                self,
                "API Key Required",
                "No API key configured. Would you like to set it up now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.show_api_config()
            return
        
        # Clear chat
        for i in reversed(range(self.chat_layout.count())):
            widget = self.chat_layout.itemAt(i).widget()
            if widget and not isinstance(widget, type(None)):
                widget.deleteLater()
        
        self.chat_layout.addStretch()
        self.conversation_history = []
        
        # Start with greeting
        greeting = "你好！我是美玲。我们今天聊什么？"
        greeting_pinyin = "nǐ hǎo! wǒ shì měi líng. wǒ men jīn tiān liáo shén me?"
        self.generate_bot_audio(greeting, greeting_pinyin)
    
    def show_settings(self):
        dialog = SettingsDialog(
            self,
            self.show_chinese,
            self.show_pinyin,
            self.current_speed
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            (self.show_chinese, self.show_pinyin, self.current_speed) = dialog.get_values()
            self.status_label.setText("Settings updated")
    
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
    
    def generate_bot_audio(self, text, pinyin_text):
        self.audio_gen = AudioGenerator(self.spark_tts_model, text, self.current_speed)
        self.audio_gen.finished.connect(
            lambda success, msg, path: self.on_bot_audio_ready(success, path, text, pinyin_text)
        )
        self.audio_gen.start()
    
    def on_bot_audio_ready(self, success, audio_path, text, pinyin_text):
        if success:
            bubble = ChatBubble(
                text, is_bot=True, audio_path=audio_path,
                show_chinese=self.show_chinese,
                show_pinyin=self.show_pinyin,
                pinyin_text=pinyin_text,
                allow_translation=True
            )
            bubble.play_audio.connect(self.play_audio)
            bubble.translate_requested.connect(self.translate_text)
            
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
            
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
    
    def translate_text(self, chinese_text, bubble_widget):
        if not self.config.get('api_key'):
            QMessageBox.warning(self, "API Key Missing", "Please configure your API key first.")
            return
        
        self.status_label.setText("Translating...")
        
        self.translation_thread = TranslationThread(
            self.config['api_key'],
            self.config.get('model', 'claude-3-5-haiku-20241022'),
            chinese_text
        )
        self.translation_thread.finished.connect(
            lambda success, translation: self.on_translation_received(success, translation, bubble_widget)
        )
        self.translation_thread.start()
    
    def on_translation_received(self, success, translation, bubble_widget):
        if success:
            bubble_widget.show_translation(f"📖 {translation}")
            self.status_label.setText("Translation complete")
        else:
            self.status_label.setText("Translation failed")
    
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
            pinyin_text=pinyin_text
        )
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        
        self.conversation_history.append({
            'speaker': 'user',
            'text': transcription
        })
        
        self.get_api_response(transcription)
    
    def get_api_response(self, user_message):
        if not self.config.get('api_key'):
            QMessageBox.warning(self, "API Key Missing", "Please configure your API key first.")
            return
        
        self.status_label.setText("美玲 is thinking...")
        
        hsk_level = self.config.get('hsk_level', 2)
        
        self.claude_thread = ClaudeConversationThread(
            self.config['api_key'],
            self.config.get('model', 'claude-3-5-haiku-20241022'),
            self.conversation_history[:-1],
            user_message,
            hsk_level
        )
        self.claude_thread.finished.connect(self.on_api_response_received)
        self.claude_thread.start()
    
    def on_api_response_received(self, success, message, response_text):
        if not success:
            self.status_label.setText(f"Error: {message}")
            QMessageBox.warning(self, "API Error", message)
            return
        
        pinyin_result = pinyin(response_text, style=Style.TONE)
        response_pinyin = ' '.join([''.join(p) for p in pinyin_result])
        
        self.generate_bot_audio(response_text, response_pinyin)
        self.status_label.setText("Response received")
    
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