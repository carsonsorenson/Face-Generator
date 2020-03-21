from flags import FLAGS
from training import train


def train_all_images(flags):
    train(flags, 'all', load=False)


def train_male(flags):
    attributes = {'male': True}
    train(flags, 'male', load=True, epoch=30, attributes=attributes)


def train_female(flags):
    attributes = {'male': False}
    train(flags, 'female', load=False, attributes=attributes)


def train_eyeglasses(flags):
    attributes = {'eyeglasses': True}
    train(flags, 'eyeglasses', load=False, attributes=attributes)


def train_blonde_hair_female(flags):
    attributes = {'male': False, 'blonde_hair': True}
    train(flags, 'blonde_hair_female', load=False, attributes=attributes)


def train_black_hair_male(flags):
    attributes = {'male': True, 'black_hair': True}
    train(flags, 'black_hair_male', load=False, attributes=attributes)


def main():
    flags = FLAGS()
    train_all_images(flags)
    train_male(flags)
    #train_female(flags)
    #train_eyeglasses(flags)
    #train_blonde_hair_female(flags)
    #train_black_hair_male(flags)


if __name__ == '__main__':
    main()
