# AirGuitarCV

AirGuitarCV is a vision-only augmented reality guitar system. 
When a valid guitar-playing body pose is detected, a holographic virtual guitar overlay appears aligned to the body.

## Phase 1
Detects "guitar holding pose" in real time and visualizes system state.

## Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   make install
   # Or manually:
   pip install -e .
   pip install -r requirements.txt
   ```

## Run

```bash
make run
# Or manually:
python scripts/run_demo.py
```
