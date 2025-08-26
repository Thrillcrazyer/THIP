import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
EVENTLOG_DIR = os.path.join(ROOT_DIR,'eventlogs')
SAVE_DIR = os.path.join(ROOT_DIR, 'saved')
RESULT_DIR = os.path.join(ROOT_DIR, 'results')