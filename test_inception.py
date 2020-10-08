from scoring.inception import get_inception_score as is_tf
from inception_score_torch import inception_score as is_torch
import numpy as np

if __name__ == "__main__":
    from data import cifar10_data
    train_data = cifar10_data.DataLoader('data', 'train', 1, rng=None, shuffle=True, return_labels=False)

    # from data import imagenet_data
    # train_data = imagenet_data.DataLoader('data', 'train', 1, rng=None, shuffle=True, return_labels=False)

    images = train_data.data

    # images1 = np.transpose(images, (0,3,1,2))
    # images1 = list(images1)[:100]
    #
    # # process = lambda img: ((img.astype('float') / 255.0))
    # # images1 = [process(s) for s in images1]
    #
    # print('num images:', len(images1))
    # print('images[0].shape:', images1[0].shape)
    #
    # mean, var = is_torch(images1, cuda=False, batch_size=10, resize=True, splits=1)
    # print('Inception Score Torch: mean={}, variance={}'.format(mean, var))
    #
    # # predictions_path = 'test_preds.npz'
    # # print('saving predictions to {} ...'.format(predictions_path))
    # # np.savez(predictions_path, preds=preds)

    images2 = list(images)[:100]
    assert(type(images2) == list)
    assert(type(images2[0]) == np.ndarray)
    assert(len(images2[0].shape) == 3)
    assert(np.max(images2[0]) > 10)
    assert(np.min(images2[0]) >= 0.0)

    print(np.min(images2[0]))
    print(np.max(images2[0]))

    print('num images:', len(images2))
    print('images[0].shape:', images2[0].shape)

    # process = lambda img: ((img + 1) * 255 / 2).astype('uint8')
    # images2 = [process(s) for s in images2]

    mean, var, preds = is_tf(images2, splits=1)
    print('Inception Score TF: mean={}, variance={}'.format(mean, var))

    # predictions_path = 'test_preds.npz'
    # print('saving predictions to {} ...'.format(predictions_path))
    # np.savez(predictions_path, preds=preds)

