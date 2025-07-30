import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame
import os
import random

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for sound (Render-safe)
SOUND_ENABLED = True
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
except pygame.error:
    SOUND_ENABLED = False
    print("⚠️ Audio device not found. Sounds disabled.")

# Load sounds
SOUND_FOLDER = "sounds"
try:
    if SOUND_ENABLED:
        TOUCH_SOUND = pygame.mixer.Sound(os.path.join(SOUND_FOLDER, "touch_sound.wav"))
        CRACK_SOUND = pygame.mixer.Sound(os.path.join(SOUND_FOLDER, "crack_sound.wav"))
        POP_SOUND = pygame.mixer.Sound(os.path.join(SOUND_FOLDER, "pop_sound.wav"))

        # Set volumes
        TOUCH_SOUND.set_volume(0.3)
        CRACK_SOUND.set_volume(0.6)
        POP_SOUND.set_volume(0.8)
    else:
        TOUCH_SOUND = None
        CRACK_SOUND = None
        POP_SOUND = None
except:
    st.error("Please ensure sound files are in the 'sounds' folder!")
    TOUCH_SOUND = None
    CRACK_SOUND = None
    POP_SOUND = None

# Initialize session state
if 'bubbles' not in st.session_state:
    st.session_state.bubbles = None
if 'total_pops' not in st.session_state:
    st.session_state.total_pops = 0
if 'session_pops' not in st.session_state:
    st.session_state.session_pops = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'last_touch_time' not in st.session_state:
    st.session_state.last_touch_time = {}
if 'touched_bubbles' not in st.session_state:
    st.session_state.touched_bubbles = set()

# Bubble configuration
GRID_ROWS = 6
GRID_COLS = 8
BUBBLE_RADIUS = 30
BUBBLE_SPACING = 65
GRID_OFFSET_X = 80
GRID_OFFSET_Y = 80

# Colors - More realistic bubble wrap colors with better opacity
BACKGROUND_COLOR = (245, 245, 240)  # Off-white like real bubble wrap
COLOR_BUBBLE_BASE = (250, 250, 250)  # Almost white
COLOR_BUBBLE_SHADOW = (80, 80, 75)  # Darker shadow for visibility
COLOR_BUBBLE_HIGHLIGHT = (255, 255, 255)  # Pure white highlight
COLOR_CRACK_LINE = (60, 60, 55)  # Darker crack lines
COLOR_POPPED = (200, 200, 195)  # More visible when popped
COLOR_BUBBLE_OUTLINE = (120, 120, 115)  # Outline for better visibility

class BubbleState:
    UNPOPPED = 0
    TOUCHED = 1
    CRACKED = 2
    POPPED = 3

class Bubble:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.state = BubbleState.UNPOPPED
        self.touch_scale = 1.0
        self.crack_animation = 0
        self.crack_angles = [random.uniform(0, 360) for _ in range(4)]  # Random crack directions
        self.pop_animation = 0
        self.last_interaction = 0
        self.highlight = False
        self.crack_segments = []  # Store the 4 segments after cracking
        
    def touch(self):
        """Handle touch interaction"""
        if self.state == BubbleState.UNPOPPED:
            self.touch_scale = 0.95  # Slight depression
            self.state = BubbleState.TOUCHED
            current_time = time.time()
            bubble_id = f"{self.x}_{self.y}"
            
            # Play touch sound only if not recently touched
            if bubble_id not in st.session_state.last_touch_time or \
               current_time - st.session_state.last_touch_time[bubble_id] > 0.5:
                if TOUCH_SOUND:
                    TOUCH_SOUND.play()
                st.session_state.last_touch_time[bubble_id] = current_time
            return True
        return False
    
    def crack(self):
        """First stage - crack the bubble"""
        if self.state == BubbleState.TOUCHED:
            self.state = BubbleState.CRACKED
            self.crack_animation = 1
            if CRACK_SOUND:
                CRACK_SOUND.play()
            
            # Initialize crack segments
            self._create_crack_segments()
            return True
        return False
    
    def pop(self):
        """Second stage - fully pop the bubble"""
        if self.state == BubbleState.CRACKED:
            self.state = BubbleState.POPPED
            self.pop_animation = 1
            if POP_SOUND:
                POP_SOUND.play()
            return True
        elif self.state == BubbleState.TOUCHED:
            # Allow direct pop from touched state
            self.crack()
            self.state = BubbleState.POPPED
            self.pop_animation = 1
            if POP_SOUND:
                POP_SOUND.play()
            return True
        return False
    
    def _create_crack_segments(self):
        """Create 4 segments from crack lines"""
        # Calculate crack endpoints
        self.crack_segments = []
        for i in range(4):
            angle = self.crack_angles[i] * math.pi / 180
            end_x = self.x + self.radius * 0.8 * math.cos(angle)
            end_y = self.y + self.radius * 0.8 * math.sin(angle)
            self.crack_segments.append({
                'start': (self.x, self.y),
                'end': (int(end_x), int(end_y)),
                'offset': 0
            })
    
    def release(self):
        """Release touch"""
        if self.state == BubbleState.TOUCHED:
            self.touch_scale = 1.0
            self.state = BubbleState.UNPOPPED
    
    def draw(self, frame):
        """Draw bubble with realistic effects"""
        if self.state == BubbleState.UNPOPPED or self.state == BubbleState.TOUCHED:
            # Draw shadow for depth
            shadow_offset = 4
            cv2.circle(frame, 
                      (self.x + shadow_offset, self.y + shadow_offset), 
                      int(self.radius * self.touch_scale), 
                      COLOR_BUBBLE_SHADOW, -1)
            
            # Draw main bubble with semi-transparency effect
            overlay = frame.copy()
            cv2.circle(overlay, 
                      (self.x, self.y), 
                      int(self.radius * self.touch_scale), 
                      COLOR_BUBBLE_BASE, -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Draw edge gradient for 3D effect
            for i in range(3):
                color_val = 250 - i * 20
                radius = int(self.radius * self.touch_scale * (1 - i * 0.1))
                cv2.circle(frame, (self.x, self.y), radius, 
                          (color_val, color_val, color_val), 2)
            
            # Draw highlight
            highlight_x = self.x - int(self.radius * 0.3)
            highlight_y = self.y - int(self.radius * 0.3)
            highlight_radius = int(self.radius * 0.3)
            
            # Create highlight with gradient
            overlay = frame.copy()
            cv2.circle(overlay, (highlight_x, highlight_y), 
                      highlight_radius, COLOR_BUBBLE_HIGHLIGHT, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw outline for visibility
            cv2.circle(frame, (self.x, self.y), 
                      int(self.radius * self.touch_scale), 
                      COLOR_BUBBLE_OUTLINE, 2)
            
            # Highlight on hover
            if self.highlight:
                cv2.circle(frame, (self.x, self.y), 
                          int(self.radius * self.touch_scale + 5), 
                          (100, 150, 255), 3)
            
        elif self.state == BubbleState.CRACKED:
            # Draw cracked bubble
            self.crack_animation = min(self.crack_animation + 1, 10)
            
            # Draw base bubble (slightly deflated) with transparency
            overlay = frame.copy()
            cv2.circle(overlay, (self.x, self.y), 
                      int(self.radius * 0.9), 
                      COLOR_BUBBLE_BASE, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw crack lines with thicker strokes
            for i, segment in enumerate(self.crack_segments):
                # Animate crack growth
                progress = min(self.crack_animation / 10, 1.0)
                end_x = int(self.x + (segment['end'][0] - self.x) * progress)
                end_y = int(self.y + (segment['end'][1] - self.y) * progress)
                
                # Draw crack line
                cv2.line(frame, (self.x, self.y), (end_x, end_y), 
                        COLOR_CRACK_LINE, 3)
                
                # Add slight separation between segments
                if progress > 0.5:
                    offset = int((progress - 0.5) * 4)
                    segment['offset'] = offset
            
            # Draw outline
            cv2.circle(frame, (self.x, self.y), 
                      int(self.radius * 0.9), 
                      COLOR_BUBBLE_OUTLINE, 2)
            
        elif self.state == BubbleState.POPPED:
            # Draw fully popped bubble
            self.pop_animation = min(self.pop_animation + 1, 15)
            
            if self.pop_animation < 10:
                # Draw separating segments with better visibility
                for i, segment in enumerate(self.crack_segments):
                    # Calculate segment center and offset
                    angle = self.crack_angles[i] * math.pi / 180
                    offset = self.pop_animation * 3
                    offset_x = int(offset * math.cos(angle))
                    offset_y = int(offset * math.sin(angle))
                    
                    # Draw segment as small arc
                    segment_radius = int(self.radius * 0.3 * (1 - self.pop_animation / 20))
                    if segment_radius > 0:
                        overlay = frame.copy()
                        cv2.circle(overlay, 
                                  (self.x + offset_x, self.y + offset_y), 
                                  segment_radius, 
                                  COLOR_BUBBLE_BASE, -1)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw flat popped bubble with better visibility
            overlay = frame.copy()
            cv2.circle(overlay, (self.x, self.y), self.radius, COLOR_POPPED, -1)
            cv2.circle(overlay, (self.x, self.y), self.radius, COLOR_BUBBLE_OUTLINE, 2)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

def initialize_bubbles():
    """Create a new bubble wrap sheet"""
    bubbles = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Slight offset for more realistic look
            x = GRID_OFFSET_X + col * BUBBLE_SPACING + random.randint(-2, 2)
            y = GRID_OFFSET_Y + row * BUBBLE_SPACING + random.randint(-2, 2)
            bubbles.append(Bubble(x, y, BUBBLE_RADIUS))
    return bubbles

def detect_pinch(hand_landmarks):
    """Detect pinch gesture with better accuracy"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate distance
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x)**2 + 
        (thumb_tip.y - index_tip.y)**2
    )
    
    # Check if pinching
    if distance < 0.05:
        pinch_x = (thumb_tip.x + index_tip.x) / 2
        pinch_y = (thumb_tip.y + index_tip.y) / 2
        return pinch_x, pinch_y, distance
    
    return None

def detect_touch(hand_landmarks):
    """Detect when finger is near/touching bubble"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return index_tip.x, index_tip.y

def main():
    st.set_page_config(page_title="Realistic Bubble Wrap", layout="wide")
    
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f0;
    }
    .stats-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🫧 Realistic Virtual Bubble Wrap")
    st.markdown("Touch to depress • Pinch to crack • Pinch again to pop!")
    
    # Initialize bubbles
    if st.session_state.bubbles is None:
        st.session_state.bubbles = initialize_bubbles()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 📊 Statistics")
        st.markdown(f"""
        <div class="stats-box">
        <b>Session Pops:</b> {st.session_state.session_pops}<br>
        <b>Total Pops:</b> {st.session_state.total_pops}<br>
        <b>Bubbles Left:</b> {sum(1 for b in st.session_state.bubbles if b.state != BubbleState.POPPED)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎮 How to Play")
        st.markdown("""
        1. **Touch**: Move finger over bubble
        2. **Crack**: Pinch once (thumb + index)
        3. **Pop**: Pinch cracked bubble again
        4. **New Sheet**: Click button below
        """)
        
        if st.button("🔄 New Bubble Sheet", key="new_sheet"):
            st.session_state.bubbles = initialize_bubbles()
            st.session_state.session_pops = 0
            st.session_state.touched_bubbles.clear()
    
    with col1:
        run = st.checkbox('Start Camera', value=True)
        FRAME_WINDOW = st.image([])
        
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Camera not found. Please check your camera settings.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            
            with mp_hands.Hands(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                max_num_hands=1
            ) as hands:
                
                previous_pinch = False
                
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Flip and resize camera frame
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (800, 600))
                    
                    # Create semi-transparent overlay for bubble wrap
                    white_overlay = np.ones((600, 800, 3), dtype=np.uint8) * 245
                    frame = cv2.addWeighted(frame, 0.3, white_overlay, 0.7, 0)
                    
                    # Process hand detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    # Reset touch states
                    for bubble in st.session_state.bubbles:
                        bubble.highlight = False
                        if bubble.state == BubbleState.TOUCHED:
                            bubble.release()
                    
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        # Detect touch position
                        touch_x, touch_y = detect_touch(hand_landmarks)
                        touch_x = int(touch_x * 800)
                        touch_y = int(touch_y * 600)
                        
                        # Check for touch interactions
                        for bubble in st.session_state.bubbles:
                            if bubble.state != BubbleState.POPPED:
                                distance = math.sqrt(
                                    (bubble.x - touch_x)**2 + 
                                    (bubble.y - touch_y)**2
                                )
                                
                                if distance < bubble.radius * 1.5:
                                    bubble.highlight = True
                                    if distance < bubble.radius:
                                        bubble.touch()
                        
                        # Check for pinch
                        pinch_result = detect_pinch(hand_landmarks)
                        
                        if pinch_result and not previous_pinch:
                            pinch_x, pinch_y, pinch_strength = pinch_result
                            pinch_x = int(pinch_x * 800)
                            pinch_y = int(pinch_y * 600)
                            
                            # Find bubble at pinch location
                            for bubble in st.session_state.bubbles:
                                distance = math.sqrt(
                                    (bubble.x - pinch_x)**2 + 
                                    (bubble.y - pinch_y)**2
                                )
                                
                                if distance < bubble.radius:
                                    if bubble.state == BubbleState.TOUCHED:
                                        if bubble.crack():
                                            break
                                    elif bubble.state == BubbleState.CRACKED:
                                        if bubble.pop():
                                            st.session_state.session_pops += 1
                                            st.session_state.total_pops += 1
                                            break
                                    elif bubble.state == BubbleState.UNPOPPED:
                                        # Direct pop with strong pinch
                                        if pinch_strength < 0.03:
                                            bubble.touch()
                                            bubble.crack()
                                            if bubble.pop():
                                                st.session_state.session_pops += 1
                                                st.session_state.total_pops += 1
                                                break
                            
                            # Visual feedback for pinch
                            cv2.circle(frame, (pinch_x, pinch_y), 20, (100, 255, 100), 2)
                        
                        previous_pinch = pinch_result is not None
                        
                        # Draw hand indicator
                        cv2.circle(frame, (touch_x, touch_y), 5, (255, 100, 100), -1)
                    
                    # Draw all bubbles
                    for bubble in st.session_state.bubbles:
                        bubble.draw(frame)
                    
                    # Draw stats on frame
                    cv2.putText(frame, f"Pops: {st.session_state.session_pops}", 
                              (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                    
                    # Show frame
                    FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
            
            cap.release()

if __name__ == "__main__":
    main()
