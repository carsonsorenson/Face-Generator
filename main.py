from flags import FLAGS
from training import train


def train_all_images(flags):
    train(flags, 'all', load=False)


def main():
    flags = FLAGS()
    train_all_images(flags)


if __name__ == '__main__':
    main()
