# AI Proctoring & Behaviour Analysis System

Created by Mayank. The application displays a prominent MAYANK startup banner and a persistent MAYANK watermark on the video and saved alert snapshots.

This project is a real-time webcam proctoring and behaviour analysis tool built for hackathon problem statement ID 102. It monitors face presence, head pose, gaze direction, suspicious movement, and other behavioural indicators using a local webcam. When a violation is detected, it logs it in the terminal and can send an instant Telegram alert with a snapshot.

It runs fully locally with Python and does not require any cloud-based processing.

## Features

- Face presence detection
- Multiple face detection
- Head pose and gaze tracking
- Suspicious movement detection
- Behaviour scoring and valence analysis
- Live terminal logging
- Local snapshot saving
- Telegram alerts with image attachment
- Cooldown-based spam prevention

## Project Structure

```text
proctoring_ai_102/
|-- main.py              # main application logic
|-- config.py            # app configuration and environment settings
|-- .env.example         # sample environment variables
|-- requirements.txt     # Python dependencies
|-- .gitignore           # ignore local secrets and generated files
|-- package_project.py    # script to package the project
|-- README.md            # project documentation
|-- snapshots/           # generated violation images (created at runtime)
`-- models/              # optional MediaPipe model files (downloaded at runtime if needed)
```

## Requirements

- Python 3.9 to 3.12
- Webcam
- Internet access (only required once for MediaPipe model download if needed)
- Telegram bot token and chat ID for alerts (optional but recommended)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/MayanKiz/proctoring_ai_102.git
cd proctoring_ai_102
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create your environment file

```bash
cp .env.example .env
```

Windows:

```powershell
copy .env.example .env
```

Then open `.env` and fill in your Telegram values:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

If Telegram credentials are missing, the application still works in local mode and logs violations to the terminal, but it will skip Telegram alerts.

### 5. Test Telegram integration

```bash
python main.py --test-telegram
```

### 6. Run the application

```bash
python main.py
```

On first run, the system calibrates for about 1 second. Look directly at the screen during calibration. Use:

- `c` to recalibrate
- `q` or `Esc` to quit
- `Ctrl + C` to stop the program

At startup, the program asks whether to enable Telegram. Choose **Skip** to run locally with no data sent over the internet. Choose **Yes** to enter your bot token and chat ID; the app sends a one-time test message before enabling violation alerts/snapshot delivery. Credentials are stored in the git-ignored `.env` file. Use `--non-interactive` to skip the prompt and use configured `.env` values only.

## Telegram Setup

### Create a bot and get the chat ID

1. Open Telegram and message `@BotFather`.
2. Send `/newbot` and follow the prompts.
3. Copy the bot token provided by BotFather.
4. Start a chat with the new bot and send any message like `hi`.
5. Open this URL in a browser:

```text
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

6. Find the `chat` object and copy the `id` value. If `result` is empty, send a message to your bot first and refresh the page.
7. Add both values to your `.env` file:

```env
TELEGRAM_BOT_TOKEN=123456789:AAExampleToken
TELEGRAM_CHAT_ID=987654321
```

## Command Line Options

```bash
python main.py [--camera N] [--no-window] [--backend auto|legacy|tasks|haar] [--test-telegram] [--non-interactive]
```

Examples:

```bash
python main.py --camera 1
python main.py --backend tasks
python main.py --no-window --backend tasks
```

## Use with Codex, ChatGPT, Claude, Gemini, or another coding assistant

1. Open this repository in the coding tool (clone the GitHub repo or upload the project folder).
2. Give the assistant a specific task, for example: “Review this project, explain the webcam pipeline, and add a configurable alert cooldown. Do not expose `.env` or the bot token.”
3. Ask it to inspect `README.md`, `config.py`, and relevant sections of `main.py` before editing. Review the proposed diff and run `pip install -r requirements.txt` plus `python main.py --test-telegram` (after configuring `.env`) or your normal app checks.
4. Keep credentials in local `.env` or the coding tool's secret manager. Never paste a live token into prompts, source code, issues, or commits. If a token is exposed, revoke it with `@BotFather` and create a replacement.
5. For Codex/Claude Code or other terminal agents, clone the repo, install dependencies in a virtual environment, then start the tool from the project directory. For browser-based assistants, use their GitHub integration or upload a ZIP; do not grant broader GitHub permissions than needed.

Assistants may make mistakes; inspect changes before accepting or pushing them. The app sends violation details and a webcam snapshot to Telegram only after you opt in and pass the Telegram connection test. Choose **Skip** for local-only operation.

## Detection and Analysis

The system analyzes facial landmarks and head movement in real time. It can detect:

- no face detected
- multiple faces detected
- looking away from the screen
- suspicious head movement
- distracted or abnormal behaviour
- negative or unusual emotional valence

These signals are compared against a per-user baseline established during calibration.

## Configuration

The application can be configured using `.env`. Available options include:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
ALERT_COOLDOWN_SECONDS=5
LOOK_AWAY_SECONDS=2
CAMERA_INDEX=0
SHOW_WINDOW=true
FACE_BACKEND=auto
FLIP_FRAME=false
MIRROR_DIRECTIONS=false
YAW_THRESHOLD_DEG=25
PITCH_THRESHOLD_DEG=18
ALERT_ON_SUSPICIOUS_MOVEMENT=false
```

## Troubleshooting

### Camera not opening

- Close apps using the camera, such as Zoom or Teams.
- Try a different camera index:

```bash
python main.py --camera 1
```

- Check your system privacy settings to allow camera access for Terminal or VS Code.

### Mediapipe issues

Use Python 3.9 to 3.12.

```bash
python main.py --backend tasks
```

If the model file is missing, download it once and place it in the `models/` folder, or allow the app to download it automatically if network access is available.

### Telegram errors

- `HTTP 401`: invalid token
- `HTTP 400/403`: invalid chat ID or bot not interacted with yet

### Head pose seems reversed

Set this in `.env`:

```env
FLIP_FRAME=true
```

or

```env
MIRROR_DIRECTIONS=true
```

## Responsible Use

This tool is designed for monitoring and assistance, not as a fully automatic decision-maker. It is useful for flagging possible violations for human review, but false positives can occur due to lighting, glasses, camera angle, or unusual posture. Treat alerts as signals requiring human interpretation.

## Verified Runtime

This repository has been verified with Python 3.12 using the current MediaPipe Tasks backend. The project is designed to work best with the default `auto` backend mode.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Built For

Hackathon Problem Statement ID 102 - AI Proctoring & Behaviour Analysis System
