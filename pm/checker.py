import os
import pandas as pd

from pm4py import fitness_token_based_replay, precision_token_based_replay, fitness_alignments, precision_alignments
import numpy as np
from .eventLog import EventLog
import pm4py

def check_conformance(Net:pm4py.PetriNet, Log:EventLog) -> pd.DataFrame:
    result_log = pd.DataFrame(columns=['Case ID', 'Correctness', 'Fitness', 'Precision', 'F1 Score'])

    caseid=Log['Case ID'].unique()[0]
    if caseid not in Net:
        raise ValueError(f"Case ID {caseid} not found in Net.")

    reason_net = Net[caseid]['net']
    reason_im = Net[caseid]['initial_marking']
    reason_fm = Net[caseid]['final_marking']
    correctness = Net[caseid].get('correctness', None)

    fitness = fitness_alignments(Log, reason_net, reason_im, reason_fm,
                                     activity_key='Activity', timestamp_key='Step', case_id_key='Case ID')['log_fitness']

    precision = precision_alignments(Log, reason_net, reason_im, reason_fm,
                                    activity_key='Activity', timestamp_key='Step', case_id_key='Case ID')

    f1_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0

    new_row = {
        'Case ID': caseid, 'Correctness': correctness,
        'Fitness': fitness, 'Precision': precision,
        'F1 Score': f1_score}
    result_log = pd.concat([result_log, pd.DataFrame([new_row])], ignore_index=True)

    return result_log


class Checker:
    def __init__(self, TrueLog: object, ReasonNet: dict):

        self.truelog = TrueLog.log
        self.check_log()
        self.net = ReasonNet

    def check_log(self):
        assert self.truelog['Step'].dtype == 'datetime64[ns]', f'datetime expected, but got {self.truelog["Step"].dtype}'
        assert 'Case ID' in self.truelog.columns, f'Case ID column not found in log'
        assert 'Activity' in self.truelog.columns, f'Activity column not found in log'
        assert 'Step' in self.truelog.columns, f'Step column not found in log'

    def check(self):
        check=check_conformance(self.net, self.truelog)
        return check