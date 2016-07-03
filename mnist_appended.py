import numpy as np
import os
import sys


# path {datasets_dir}/mnist should contain the following files
# t10k-images.idx3-ubyte  train-images.idx3-ubyte
# t10k-labels.idx1-ubyte  train-labels.idx1-ubyte
datasets_dir = os.path.join(os.path.abspath(".."), "Theano-Tutorials")


def append_digits(images, labels, start_idx, end_idx):
    """
    Parameters
        images: array-like (n_examples, 28*28)
        labels: array-like (n_examples,) integer labels
        start_idx: index to start appending digits
        end_idx: index to stop appending digits
    Returns
        appended_image: array-like (28*28*n_appended_digits,)
        appended_label: string "98352" for example
    """
    appended_image = np.column_stack(
        [images[idx].reshape((28, 28)) for idx in range(start_idx, end_idx)])
    appended_label = "".join([str(num) for num in labels[start_idx:end_idx]])
    return (appended_image, appended_label + "\n")


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(ntrain=60000, ntest=10000, onehot=False):
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX/255.
    teX = teX/255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


def example():
    import matplotlib.pyplot as plt

    # append a couple of digits and display the image and the label
    trX, teX, trY, teY = mnist(ntrain=100, ntest=100, onehot=False)
    img, cap = append_digits(trX, trY, 0, 5)
    plt.title(cap)
    plt.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.savefig('mnist_appended_sample.png', bbox_inches='tight')


def generate_dataset(X, Y):
    assert X.shape[0] == Y.shape[0]
    n_examples = X.shape[0]

    images = []
    captions = []
    idx = 0
    while (idx < n_examples - 1):
        # how long will the appended example be?
        n_to_append = np.random.randint(2, 10)

        # define the index in the dataset to include up to
        # when at the last examples, select examples up until the end
        end_idx = (idx + n_to_append)
        if end_idx > n_examples:
            end_idx = n_examples

        # make new appended example
        img, cap = append_digits(X, Y, idx, end_idx)
        images.append(img)
        captions.append(cap)

        # start off from last used example
        idx = end_idx

    return images, captions


def main():
    if len(sys.argv) < 2:
        print("Not enough arguments supplied")
        return

    datapath = sys.argv[1]
    train_image_path = os.path.join(datapath, "train_images.npz")
    test_image_path = os.path.join(datapath, "test_images.npz")
    train_cap_path = os.path.join(datapath, "train_labels.txt")
    test_cap_path = os.path.join(datapath, "test_labels.txt")

    trX, teX, trY, teY = mnist(onehot=False)

    print("Generating Appended MNIST Training...")
    train_imgs, train_caps = generate_dataset(trX, trY)
    print("Generating Appended MNIST Testing...")
    test_imgs, test_caps = generate_dataset(teX, teY)

    print("Save Training/Testing Images...")
    np.savez_compressed(train_image_path, *train_imgs)
    np.savez_compressed(test_image_path, *test_imgs)

    print("Save Training/Testing Captions...")
    with open(train_cap_path, 'w') as train_cap_file:
        train_cap_file.writelines(train_caps)
    with open(test_cap_path, 'w') as test_cap_file:
        test_cap_file.writelines(test_caps)

    print("DONE. SUMMARY")
    print("# Train Examples: " + str(len(train_imgs)))
    print("# Test Examples: " + str(len(test_imgs)))

if __name__ == '__main__':
    main()