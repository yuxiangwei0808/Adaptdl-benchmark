import subprocess
import pandas
import os

workload = pandas.read_csv('/home/lcwyx/demo_adaptdl/job_submit/adaptdl/benchmark/workloads/workload-6.csv')

for row in workload.sort_values(by='time').itertuples():
    job_name = row.name
    subprocess.run(['kubectl', 'delete', 'adaptdljob', f'{job_name}'])

# def find_keyword():
#     files = os.listdir('./logs_1')
    
#     for file in files:
#         with open(os.path.join('./logs_1', file), 'r') as f:
#             log = f.read()
#             if 'Stream' in log:
#                 print(file)

# find_keyword()