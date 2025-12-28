from src.fashion.data import get_fashionmnist_loaders

def test_fashion_dev_split():
    # Use return_indices=True to get the actual indices used in the split
    train_loader, dev_loader, test_loader, n_train, n_dev, n_test, train_indices, dev_indices = get_fashionmnist_loaders(
        data_dir='data', batch_size=8, dev_ratio=0.2, quick=True, seed=123, return_indices=True)
    # 200 train subset, 20% dev => 160 train, 40 dev
    assert n_train == 160
    assert n_dev == 40
    # No overlap in indices
    assert set(train_indices).isdisjoint(set(dev_indices))
