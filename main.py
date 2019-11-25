from flags import FLAGS
from training import train


def train_all_images(flags):
    attributes = {'male': True}
    train(flags, 'male', load=False, attributes=attributes)


def main():
    flags = FLAGS()
    train_all_images(flags)


if __name__ == '__main__':
    main()
