from pm.miner import Miner
from pm.checker import Checker
import reward
import pm
import pandas as pd

truelogs=pd.read_csv('eventlogs/DeepMath_eventlog.csv')

def process_reward_func(think:str, index:int)->float:
    think_log = reward.Answer2EventAgent().make_event_log(think)
    think_log.log['Case ID'] = str(index)
    reason_net = Miner(think_log).discover()
    
    true_log_df = truelogs[truelogs['Case ID'] == index].copy()
    true_log_df['Case ID'] = str(index)
    true_eventlog = pm.EventLog(true_log_df)

    conf_df = Checker(true_eventlog, reason_net).check()

    return conf_df['F1 Score'].values[0]