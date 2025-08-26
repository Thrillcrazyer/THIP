import reward
import pandas as pd
import transformers


def main():
    eventagent=reward.Answer2EventAgent()
    deepmath=pd.read_csv('DeepMath-103k.csv')
    Event=eventagent.make_event_log(deepmath.loc[0,'r1_solution_1'])
    

if __name__ == "__main__":
    main()