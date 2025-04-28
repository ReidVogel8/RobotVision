# tts.py
import os
import shlex

def speak(text: str):
    """
    Send `text` to the espeak CLI.
    """
    # shlex.quote to avoid shell injection issues
    os.system(f'espeak {shlex.quote(text)}')
