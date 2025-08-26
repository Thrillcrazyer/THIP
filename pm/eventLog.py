import os
from pathlib import Path
import pandas as pd

from .configs import EVENTLOG_DIR

class EventLog(object):

    def __init__(self, dataframe:pd.DataFrame):
        #self.LogName = Name
        #FileName = Path(str(Name)+'.csv')
        #self.FilePath = os.path.join(EVENTLOG_DIR, FileName)
        self.log = dataframe
        self.preprocess()

    def preprocess(self):
        self.log['Case ID'] = self.log['Case ID'].astype(str)
        self.log['Step'] = self.log['Step'].astype(int)
        self.log['Activity'] = self.log['Activity'].astype(str)
        self.log['Step'] = pd.to_datetime(self.log['Step'], unit='s')
        return self
    
    @property
    def num_cases(self):
        return len(self.log['Case ID'].unique())
    
    

