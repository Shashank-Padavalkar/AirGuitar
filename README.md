# AirGuitarCV 🎸

AirGuitarCV is a vision-only augmented reality guitar system. When a valid guitar-playing body pose is detected, a physics-based interactive virtual guitar overlay appears. 

You can interact with it using hand gestures (grabbing, holding, swinging naturally).

> ⚠️ **Note:** This project is currently a work in progress (WIP) and under active development.

## Setup & Run

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

2. **Install**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

3. **Run Demo**
   ```bash
   python scripts/run_demo.py
   ```
