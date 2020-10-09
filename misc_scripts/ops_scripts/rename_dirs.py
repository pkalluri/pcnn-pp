import os
import re
import shutil

if __name__ == '__main__':

    ls = os.listdir()

    latest_sample = 0
    for filename in ls:
        # find largest and move that one only
        'shapes_sample{}.png'

        # if re.fullmatch('shapes_standard_.*', filename):
        #     params = filename.split('_')
        #     dataset, loss, modelsize, batchsize, jobid = params
        #     new_filename = f'{dataset}_{modelsize}_{loss}_{batchsize}_{jobid}'
        #     print(filename, new_filename)
        #     shutil.move(filename, new_filename)

        # if re.fullmatch('[0-9].*_log_sum_.*', filename):
        #     print(filename)
        #     params = filename.split('_')
        #     jobid, dataset, loss1, loss2, batchsize, modelsize  = params
        #     shutil.move(filename, f'{dataset}_{modelsize}_{loss1}{loss2}_{batchsize}_{jobid}')
        # elif re.fullmatch('[0-9].*_.*_.*', filename): # anything other than log_sum
        #     print(filename)
        #     params = filename.split('_')
        #     jobid, dataset, loss, batchsize, modelsize  = params
        #     shutil.move(filename, f'{dataset}_{modelsize}_{loss}_{batchsize}_{jobid}')