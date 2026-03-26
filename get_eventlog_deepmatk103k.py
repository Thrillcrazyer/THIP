import pandas as pd
import ray
import reward


@ray.remote
def process_row(row_data: dict) -> pd.DataFrame:
    """Ray remote function to process a single row"""
    agent = reward.Answer2EventAgent()
    results = []
    for sol_col in ['r1_solution_1']:
        log = row_data.get(sol_col)
        if not log or (isinstance(log, float) and pd.isna(log)):
            continue
        event_df = agent.make_event_df(log)
        if event_df is not False and event_df is not None:
            event_df['Case ID'] = f"{row_data['id']}"
            results.append(event_df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    df = pd.read_csv('DeepMath-103k_id.csv')
    
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
        print(f"Processed {i+1}/{total}, Case ID: {rows[i]['id']}")
    
    event_logs.to_csv('DeepMath-103k_eventlog.csv', index=False)
    
    # Shutdown Ray
    ray.shutdown()
    return

if __name__ == "__main__":
    main()