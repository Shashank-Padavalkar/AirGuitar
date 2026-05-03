import cv2
import numpy as np

def draw_text_with_background(img: np.ndarray, text: str, pos: tuple, 
                              font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2):
    x, y = pos
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y + 5), bg_color, -1)
    cv2.putText(img, text, pos, font, font_scale, text_color, thickness)
