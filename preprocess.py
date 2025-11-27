import os
import reward
import pandas as pd
from pm.configs import EVENTLOG_DIR

# Optional Ray support
try:
    import ray  # type: ignore
except Exception:  # keep running without Ray
    ray = None


def main():
    """
    Trace Extraction to Proplem Solving
    """
    eventagent=reward.Answer2EventAgent()
    deepmath=pd.read_csv('DeepMath-103k_id.csv')
    
    # ensure output directory exists
    os.makedirs(EVENTLOG_DIR, exist_ok=True)
    out_file = os.path.join(EVENTLOG_DIR, 'DeepMath_eventlog.csv')

    # load already processed Case IDs to avoid reprocessing
    processed_cases = set()
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        try:
            existing = pd.read_csv(out_file, usecols=['Case ID'])
            processed_cases = set(existing['Case ID'].astype(str).unique())
            print(f"[init] loaded {len(processed_cases)} already processed cases from {out_file}")
        except Exception as e:
            print(f"[warn] couldn't read existing event log ({e}); proceeding without skip cache")

    # Ray-parallel path (fallback to sequential if Ray unavailable)
    if ray is not None:
        # Define remote processing function
        @ray.remote
        def process_case(case_id: str, solution_text: str):
            try:
                import reward  # ensure availability in worker
                agent = reward.Answer2EventAgent()
                df = agent.make_event_df(solution_text)
            except Exception as e:  # generation error
                return case_id, None, f"[skip] case {case_id}: generation error -> {e}"

            if df is None or df.empty:
                return case_id, None, f"[skip] case {case_id}: empty/invalid event df"

            # standardize column name expected by pm.EventLog
            df['Case ID'] = case_id
            return case_id, df, None

        # Submit tasks
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        futures = []
        for i, logdata in deepmath.iterrows():
            if i == 60001:
                break
            case_id = str(i)
            if case_id in processed_cases:
                print(f"[skip] case {case_id}: already processed")
                continue

            solution = logdata.get('r1_solution_1', None)
            if pd.isna(solution):
                print(f"[skip] case {case_id}: solution text is NaN")
                continue

            futures.append(process_case.remote(case_id, solution))

        # Append results as they complete to avoid large memory usage
        wrote_header = os.path.exists(out_file) and os.path.getsize(out_file) > 0
        pending = set(futures)
        while pending:
            done, pending = ray.wait(list(pending), num_returns=1)
            for ref in done:
                case_id, df, err = ray.get(ref)
                if err:
                    print(err)
                    continue
                if df is None or df.empty:
                    print(f"[skip] case {case_id}: empty/invalid event df")
                    continue

                write_header = not wrote_header
                try:
                    df.to_csv(out_file, mode='a', index=False, header=write_header)
                    if not wrote_header:
                        wrote_header = True
                    print(f"[append] case {case_id}: +{len(df)} rows -> {out_file}")
                except Exception as e:
                    print(f"[error] case {case_id}: failed to write -> {e}")

        try:
            ray.shutdown()
        except Exception:
            pass
    else:
        # iterate and append logs at the bottom (sequential fallback)
        for i, logdata in deepmath.iterrows():
            case_id = str(i)

            # skip if case already processed
            if case_id in processed_cases:
                print(f"[skip] case {i}: already processed")
                continue

            solution = logdata.get('r1_solution_1', None)
            if pd.isna(solution):
                print(f"[skip] case {case_id}: solution text is NaN")
                continue

            try:
                df = eventagent.make_event_df(solution)
            except Exception as e:
                print(f"[skip] case {i}: generation error -> {e}")
                continue

            if df is None or df.empty:
                print(f"[skip] case {i}: empty/invalid event df")
                continue

            # standardize column name expected by pm.EventLog
            df['Case ID'] = case_id

            # write header only if file doesn't exist or is empty
            write_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
            try:
                df.to_csv(out_file, mode='a', index=False, header=write_header)
                print(f"[append] case {i}: +{len(df)} rows -> {out_file}")
            except Exception as e:
                print(f"[error] case {i}: failed to write -> {e}")

    # Check for missing cases from 0 to 60000
    print("\n[check] Verifying cases 0 to 60000...")
    if os.path.exists(out_file):
        try:
            final_df = pd.read_csv(out_file, usecols=['Case ID'])
            existing_cases = set(pd.to_numeric(final_df['Case ID'], errors='coerce').dropna().astype(int).unique())
            
            missing_cases = [c for c in range(60001) if c not in existing_cases]
            
            if missing_cases:
                print(f"[warn] Found {len(missing_cases)} missing cases between 0 and 60000:")
                if len(missing_cases) > 20:
                    print(f"Examples: {missing_cases[:20]} ...")
                else:
                    print(f"Missing: {missing_cases}")
            else:
                print("[success] All cases from 0 to 60000 are present.")
        except Exception as e:
            print(f"[error] Failed to verify cases: {e}")
    else:
        print(f"[error] Output file {out_file} not found.")

if __name__ == "__main__":
    main()