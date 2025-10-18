import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from collections import deque
import numpy as np
from chroma_model.live_chord_detector_chroma import LiveChordDetectorChroma

class ChordDetectorChromaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Chord Detector (Chroma Model)")
        self.root.geometry("600x400")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize detector
        self.detector = None
        self.is_detecting = False
        self.detection_thread = None
        
        # GUI variables
        self.current_chord_var = tk.StringVar(value="No chord detected")
        self.confidence_var = tk.StringVar(value="0.00")
        self.status_var = tk.StringVar(value="Ready to start")
        
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Live Chord Detector (Chroma Model)", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status
        status_label = ttk.Label(main_frame, text="Status:")
        status_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.status_display = ttk.Label(main_frame, textvariable=self.status_var,
                                       font=('Arial', 12), foreground='blue')
        self.status_display.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Current chord display
        chord_label = ttk.Label(main_frame, text="Current Chord:", font=('Arial', 14, 'bold'))
        chord_label.grid(row=2, column=0, sticky=tk.W, pady=(20, 5))
        
        self.chord_display = ttk.Label(main_frame, textvariable=self.current_chord_var,
                                      font=('Arial', 24, 'bold'), foreground='green')
        self.chord_display.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(20, 5))
        
        # Confidence display
        conf_label = ttk.Label(main_frame, text="Confidence:")
        conf_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        
        self.conf_display = ttk.Label(main_frame, textvariable=self.confidence_var,
                                     font=('Arial', 14))
        self.conf_display.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Progress bar for confidence
        self.progress = ttk.Progressbar(main_frame, length=300, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                      command=self.start_detection, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                     command=self.stop_detection, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Detection history
        history_label = ttk.Label(main_frame, text="Recent Detections:", font=('Arial', 12, 'bold'))
        history_label.grid(row=6, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        
        # Create treeview for history
        columns = ('Time', 'Chord', 'Confidence')
        self.history_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120)
        
        # Scrollbar for history
        history_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_tree.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        history_scrollbar.grid(row=7, column=2, sticky=(tk.N, tk.S))
        
        # Configure row weights for resizing
        main_frame.rowconfigure(7, weight=1)
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                                text="Instructions: Click 'Start Detection' and play chords on your instrument.\n"
                                     "The detector will show the detected chord and confidence level in real-time.",
                                font=('Arial', 10), foreground='gray')
        instructions.grid(row=8, column=0, columnspan=2, pady=10)
        
    def start_detection(self):
        """Start chord detection"""
        try:
            # Initialize detector
            self.detector = LiveChordDetectorChroma()
            
            # Start detection in a separate thread
            self.is_detecting = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_var.set("Detecting chords...")
            
            # Start UI update loop
            self._update_ui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
            self.is_detecting = False
    
    def stop_detection(self):
        """Stop chord detection"""
        self.is_detecting = False
        
        if self.detector:
            self.detector.stop_recording()
        
        # Update UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_var.set("Detection stopped")
        self.current_chord_var.set("No chord detected")
        self.confidence_var.set("0.00")
        self.progress['value'] = 0
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        if self.detector:
            self.detector.start_recording()
    
    def _update_ui(self):
        """Update UI elements with current detection results"""
        if self.is_detecting and self.detector:
            # Update chord and confidence
            current_chord = self.detector.current_chord
            confidence = self.detector.confidence
            
            # Only update display if confidence is 50% or higher
            if confidence >= 0.5:
                self.current_chord_var.set(current_chord)
                self.confidence_var.set(f"{confidence:.2f}")
                self.progress['value'] = confidence * 100
                
                # Add to history if chord changed and confidence is high enough
                if hasattr(self, '_last_displayed_chord'):
                    if current_chord != self._last_displayed_chord:
                        self._add_to_history(current_chord, confidence)
                else:
                    self._last_displayed_chord = current_chord
                
                self._last_displayed_chord = current_chord
            else:
                # Show low confidence state
                self.current_chord_var.set("Low confidence")
                self.confidence_var.set(f"{confidence:.2f}")
                self.progress['value'] = confidence * 100
            
            # Schedule next update
            self.root.after(100, self._update_ui)
    
    def _add_to_history(self, chord, confidence):
        """Add detection to history tree"""
        current_time = time.strftime("%H:%M:%S")
        
        # Insert at the top
        self.history_tree.insert('', 0, values=(current_time, chord, f"{confidence:.2f}"))
        
        # Keep only last 20 entries
        children = self.history_tree.get_children()
        if len(children) > 20:
            self.history_tree.delete(children[-1])

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configure colors
    style.configure('Accent.TButton', foreground='white', background='#0078d4')
    
    app = ChordDetectorChromaGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
