from dotenv import load_dotenv
from openai import OpenAI
import os
import yaml
from io import StringIO
import pandas as pd
import csv

import pm

class Answer2EventAgent():
    def __init__(self):
        load_dotenv()
        api_key=os.getenv("DEEPSEEK_KEY")
        self.client = OpenAI(api_key=api_key,base_url="https://api.deepseek.com")
        self.model_name="deepseek-chat"
        self.template=self.load_template()
        
    def load_template(self, yaml_file='./reward/prompt.yaml'):
      with open(yaml_file, 'r', encoding='utf-8') as file:
        template_data = yaml.safe_load(file)
      return template_data['prompt']

    def get_csv_data(self,log):
        query=self.template.format(problem=log)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        #breakpoint()
        return response.choices[0].message.content

    def parsing_answer(self, answer):
        lines = answer.strip().splitlines()
        expected_header = ["Case ID", "Step", "Activity", "Description"]
        header_idx = None
        for idx, line in enumerate(lines):
            tokens = next(csv.reader([line], quotechar='"'))
            if tokens == expected_header:
                header_idx = idx
                break
        if header_idx is None:
            print(f"⚠️ CSV 헤더가 올바르지 않아 저장하지 않습니다. (헤더를 찾을 수 없음)")
            return False
        # header_idx만큼 건너뛰고 해당 줄을 헤더로 사용
        df = pd.read_csv(StringIO(answer), skiprows=header_idx, quotechar='"')
        return df

    def make_event_df(self,log:str)->pd.DataFrame:
        answer=self.get_csv_data(log)
        df=self.parsing_answer(answer)
        return df

    def make_event_log(self,log:str)->pm.EventLog:
        answer=self.get_csv_data(log)
        df=self.parsing_answer(answer)
        #breakpoint()
        if df is not None:
            event_log = pm.EventLog(df)
            return event_log
        return None


