from pm4py import discover_petri_net_inductive



class Miner:
    def __init__(self, EventLog: object):
        self.log = EventLog.log
        self.check_log()

    def check_log(self):
        assert self.log['Step'].dtype == 'datetime64[ns]', f'datetime expected, but got {self.log["Step"].dtype}'
        assert 'Case ID' in self.log.columns, f'Case ID column not found in log'
        assert 'Activity' in self.log.columns, f'Activity column not found in log'
        assert 'Step' in self.log.columns, f'Step column not found in log'

    def discover(self) -> dict:
        """Discover a Petri net for exactly one Case ID.

        Raises ValueError if the log contains zero or more than one unique Case ID.
        Returns a dict: { case_id: { 'net': net, 'initial_marking': im, 'final_marking': fm, ... } }
        """
        unique_cases = list(self.log['Case ID'].unique())
        if len(unique_cases) != 1:
            raise ValueError(f"Expected exactly 1 unique Case ID, but found {len(unique_cases)}: {unique_cases}")

        cid = unique_cases[0]
        sublog = self.log.copy()
        sublog.reset_index(drop=True, inplace=True)

        net, initial_marking, final_marking = discover_petri_net_inductive(
            log=sublog,
            activity_key='Activity',
            timestamp_key='Step',
            case_id_key='Case ID'
        )

        result = {
            cid: {
                'net': net,
                'initial_marking': initial_marking,
                'final_marking': final_marking
            }
        }

        if 'IsTrue' in sublog.columns:
            result[cid]['correctness'] = sublog['IsTrue'].values[0]

        return result
