import json
import argparse

def out_to_args(out_path:str) -> dict:
    with open(out_path, 'r') as f:
        json_string = ''
        reading_json = False
        for line in f:
            if reading_json:
                json_string += line.strip()
                if line.startswith('}'):
                    reading_json = False

            if line.startswith('input args:'):
                reading_json = True
    
    data = json.loads(json_string)
    return data

def out_to_updated_args(out_path: str, args: argparse.Namespace) -> argparse.Namespace:
    data: dict = out_to_args(out_path)
    args_dict: dict = vars(args)
    args_dict.update(data)
    return args

def args_to_summary_string(args) -> str:
    '''From args object, constructs a concise string summarizing the key parameters'''
    return f"{args['data_set']}_{args['nr_filters']}L_{args['accumulator']}_B{args['batch_size']}"
