# src/utils.py

import pandas as pd

def convert_minutes_to_seconds(minutes_str):
    """Convert MM:SS format to total seconds."""
    minutes, seconds = map(int, minutes_str.split(':'))
    return minutes * 60 + seconds
