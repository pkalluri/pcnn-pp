import json

def read_args_from_out_file(filename):
    with open(filename, 'r') as f:
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

def overwrite_args_from_out_file(filename, args_to_overwrite):
    data = read_args_from_out_file(filename)
    d = vars(args_to_overwrite)
    d.update(data)
    return args_to_overwrite