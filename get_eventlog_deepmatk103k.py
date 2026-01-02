import pandas as pd
import ray
import reward


@ray.remote
def process_row(row_data: dict) -> pd.DataFrame:
    """Ray remote function to process a single row"""
    agent = reward.Answer2EventAgent()
    log = row_data['generated_reason']
    event_df = agent.make_event_df(log)
    if event_df is not False and event_df is not None:
        event_df['CaseID'] = row_data['CaseID']
        event_df['dataset'] = row_data['dataset'].split('/')[1]
        return event_df
    return pd.DataFrame()


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    df = pd.read_csv('generated.csv')
    
    # Convert DataFrame rows to list of dicts for Ray
    rows = df.to_dict('records')
    
    # Submit all tasks to Ray
    futures = [process_row.remote(row) for row in rows]
    
    # Process results with progress tracking
    event_logs = pd.DataFrame()
    total = len(futures)
    
    for i, future in enumerate(futures):
        result = ray.get(future)
        if not result.empty:
            event_logs = pd.concat([event_logs, result], ignore_index=True)
        print(f"Processed {i+1}/{total}")
        print(f"EVENT LOG: {rows[i]['dataset'].split('/')[1]}, CaseID: {rows[i]['CaseID']}")
    
    event_logs.to_csv('DeepSeek_testset_eventlog.csv', index=False)
    
    # Shutdown Ray
    ray.shutdown()
    return

if __name__ == "__main__":
    main()