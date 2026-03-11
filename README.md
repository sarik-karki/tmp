# Parking Monitor System

A comprehensive parking management and monitoring system using computer vision to track vehicle occupancy and manage permits/violations.

## Features

- **Vehicle Tracking**: Real-time tracking of vehicles in a parking lot using YOLOv8 and BoTSORT/ByteTrack.
- **Space Management**: Monitors specific parking spots defined by polygons and detects occupancy.
- **Automatic License Plate Recognition (ALPR)**:
  - Detects license plates on vehicles entering the lot.
  - Reads plate text using Tesseract OCR.
- **Plate Matching**: Correlates plates read at the entrance with vehicles tracked in the overhead view.
- **Database Integration**:
  - Maintains a registry of authorized vehicles and permits.
  - Records entry/exit logs and durations.
  - Automatically flags violations for unauthorized parking.
- **Dual Camera Support**: Handles both an overhead lot view and a dedicated entrance/plate camera.

## Project Structure

- `main.py`: The primary entry point for live operation.
- `src/`: Core logic modules (tracker, detector, database, etc.).
- `config/`: Configuration files for YOLO models, tracking, and parking space definitions.
- `models/`: Pre-trained YOLOv8 models for vehicles and plates.
- `tools/`: Utility scripts for mapping spaces, testing cameras, and real-time detection tests.
- `tests/`: Comprehensive test suite for all system components.

## Demos

- `demo_full.py`: A high-fidelity simulation that mocks detection results to demonstrate the full logic (tracking -> matching -> occupancy -> database) without requiring real cameras or a GPU.
- `main.py`: The live system designed for real hardware (Raspberry Pi / Desktop) with connected cameras.

## Setup

1. Run `./setup.sh` to install system dependencies and set up the Python virtual environment.
2. Activate the environment: `source venv/bin/activate`.
3. Configure your cameras in `config/settings.yaml`.
4. Define your parking spaces using `python tools/map_spaces.py`.
5. Run the monitor: `python main.py`.
