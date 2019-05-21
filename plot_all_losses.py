import argparse
import matplotlib.pyplot as plt
import re

def out2losses(fp, all_losses):
    losses = []
    f = open(fp, "r")
    losses.append(([float(line[:6]) for line in f.read().split('train bits_per_dim = ')[1:]], 'train_loss'))
    losses.append(([float(line[:6]) for line in f.read().split('test bits_per_dim = ')[1:]], 'test_loss'))
    if all_losses:
        losses.append(([float(line[:6]) for line in f.read().split('standard bits_per_dim = ')[1:]], 'standard_loss'))
        losses.append(([float(line[:6]) for line in f.read().split('sum bits_per_dim = ')[1:]], 'sum_loss'))
        losses.append(([float(line[:6]) for line in f.read().split('max bits_per_dim = ')[1:]], 'max_loss'))
    
    return losses

def out_path2prefix(fp):
    return re.search(r'.*\d+', fp).group()

def get_between(s, before, after):
    return s.split(before)[1].split(after)[0]

def out2details(fp):
    txt = open(fp, "r").read()
    details = []
    details.append('dataset='+re.search('"data_set":"(.*)",', txt).group(1))
    details.append('loss='+re.search('"accumulator":"(.*)",', txt).group(1))
    # print(details)
    # details.append('_batch_size=' + re.search('"batch_size":"(.*)",', txt).group(1))
    return details

def losses2plot(losses, details, out_path):
    for l_data, l_label in losses:
        plt.plot(range(len(l_data)), l_data)
    plt.title("Train Losses ({})".format(details))
    plt.savefig(out_path)
    plt.show()

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path', type=str, help='Location of the .out file')
    parser.add_argument('-a', '--all_losses', type=str, action='store_true', help='Should we plot all losses?')
    # parser.add_argument('-o', '--save_dir', type=str, default='save', help='Out path')
    args = parser.parse_args()

    prefix = out_path2prefix(args.path)
    details = out2details(args.path)
    out_path = os.path.join('{}_{}'.format(prefix,'_'.join(details))+'.png')
    print(out_path)

    losses = out2losses(args.path, args.all_losses)
    losses2plot(losses, ' '.join(details), out_path)