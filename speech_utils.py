import os
import subprocess


class PowerShellSpeechEngine:
    def __init__(self, rate=150):
        self.rate = rate

    def setProperty(self, name, value):
        if name == "rate":
            self.rate = value

    def say(self, text):
        self.text = str(text)

    def runAndWait(self):
        env = os.environ.copy()
        env["GESTURETALK_TTS_TEXT"] = getattr(self, "text", "")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Rate = 0; "
            "$speaker.Speak($env:GESTURETALK_TTS_TEXT)"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        pass


def create_tts_engine(rate=150):
    try:
        import pyttsx3
    except ImportError:
        if os.name == "nt":
            return PowerShellSpeechEngine(rate=rate), None
        return None, "pyttsx3 is not installed. Install it with: pip install pyttsx3"

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        return engine, None
    except Exception as exc:
        if os.name == "nt":
            return PowerShellSpeechEngine(rate=rate), None
        return None, f"Could not initialize text-to-speech engine: {exc}"


def speak_text(text, engine=None, rate=150):
    if not text:
        return False

    close_engine = engine is None
    if engine is None:
        engine, error = create_tts_engine(rate=rate)
        if engine is None:
            raise RuntimeError(error)

    engine.say(str(text))
    engine.runAndWait()

    if close_engine:
        try:
            engine.stop()
        except Exception:
            pass

    return True
