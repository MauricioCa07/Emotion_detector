import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from datetime import datetime
import customtkinter as ctk

class EmotionDetectorGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Emotion Detection System")
        self.window.configure(bg="#1a1a1a")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.model = load_model('./modelo_entrenado.keras')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 300)
        self.cam.set(4, 300)
        
        self.main_container = ctk.CTkFrame(self.window)
        self.main_container.pack(pady=20, padx=20, fill="both", expand=True)
        

        self.title_label = ctk.CTkLabel(
            self.main_container,
            text="Real-time Emotion Detection",
            font=("Helvetica", 24, "bold")
        )
        self.title_label.pack(pady=10)
        
        self.create_video_frame()
        self.create_emotion_indicators()
        self.create_controls()
        

        self.is_running = False
        self.current_emotion = "neutral"
        

        self.update()
    
    def create_video_frame(self):
        self.video_frame = ctk.CTkFrame(self.main_container)
        self.video_frame.pack(pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack()
        
    
        self.emotion_label = ctk.CTkLabel(
            self.video_frame,
            text="Detected Emotion: --",
            font=("Helvetica", 16)
        )
        self.emotion_label.pack(pady=10)
    
    def create_emotion_indicators(self):
        self.indicators_frame = ctk.CTkFrame(self.main_container)
        self.indicators_frame.pack(pady=10, fill="x", padx=20)
        
        self.progress_bars = {}
        for emotion in self.emotions:
            frame = ctk.CTkFrame(self.indicators_frame)
            frame.pack(fill="x", pady=2)
            
            label = ctk.CTkLabel(frame, text=emotion.capitalize(), width=100)
            label.pack(side="left", padx=10)
            
            progress = ctk.CTkProgressBar(frame, width=400)
            progress.pack(side="left", padx=10, fill="x", expand=True)
            progress.set(0)
            
            self.progress_bars[emotion] = progress
    
    def create_controls(self):
        self.controls_frame = ctk.CTkFrame(self.main_container)
        self.controls_frame.pack(pady=10)
        
        self.toggle_button = ctk.CTkButton(
            self.controls_frame,
            text="Start Detection",
            command=self.toggle_detection,
            width=200
        )
        self.toggle_button.pack(side="left", padx=10)
        
        self.quit_button = ctk.CTkButton(
            self.controls_frame,
            text="Quit",
            command=self.quit_app,
            width=200,
            fg_color="#FF5555",
            hover_color="#FF0000"
        )
        self.quit_button.pack(side="left", padx=10)
    
    def toggle_detection(self):
        self.is_running = not self.is_running
        self.toggle_button.configure(
            text="Stop Detection" if self.is_running else "Start Detection"
        )
    
    def update_progress_bars(self, predictions):
        for emotion, progress_bar in zip(self.emotions, self.progress_bars.values()):
            value = float(predictions[self.emotions.index(emotion)])
            progress_bar.set(value)
    
    def update(self):
        if self.is_running:
            ret, frame = self.cam.read()
            if ret:
                # Process frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (300, 200))
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(gray_frame, (48, 48))
                img = img / 255.0
                img = np.expand_dims(img, axis=-1)
                img = np.expand_dims(img, axis=0)
                
                pred = self.model.predict(img, verbose=0)
                emotion_idx = np.argmax(pred)
                self.current_emotion = self.emotions[emotion_idx]
                
                self.emotion_label.configure(
                    text=f"Detected Emotion: {self.current_emotion.capitalize()}"
                )
                self.update_progress_bars(pred[0])
                
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
        
        self.window.after(10, self.update)
    
    def quit_app(self):
        self.cam.release()
        self.window.quit()

# Create and run application
if __name__ == "__main__":
    root = ctk.CTk()
    root.title("Emotion Detection System")
    
    window_width = 800
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    app = EmotionDetectorGUI(root)
    root.mainloop()