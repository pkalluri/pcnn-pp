from scoring.inception import get_inception_score
import numpy as np

if __name__ == "__main__":
    from data import cifar10_data
    # batch_size = 16
    # nr_gpu = 8
    train_data = cifar10_data.DataLoader('data', 'train', 1, rng=None, shuffle=True, return_labels=False)
    images = train_data.next(5)
    images = list(images)
    print('num images:', len(images))
    print('images[0].shape:', images[0].shape)
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    mean, var, preds = get_inception_score(images, splits=1)
    print('Inception Score: mean={}, variance={}'.format(mean, var))

    predictions_path = 'test_preds.npz'
    print('saving predictions to {} ...'.format(predictions_path))
    np.savez(predictions_path, preds=preds)