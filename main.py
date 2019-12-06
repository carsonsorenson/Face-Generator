from flags import FLAGS
from training import train
from generate import generate


# First we generate a completely random image, just pull in the entire dataset
def train_all_images(flags):
    train(flags, 'all', load=True, epoch=7, iteration=11074)


# Add one attribute, in this case gender
def train_one_attribute(flags):
    attributes = {'male': True}
    train(flags, 'male', load=False, attributes=attributes)

    attributes = {'male': False}
    train(flags, 'female', load=False, attributes=attributes)


# Add two attributes, in this case use gender and hair color
def train_two_attributes(flags):
    attributes = {'male': True, 'black_hair': True}
    train(flags, 'black_hair_male', load=False, attributes=attributes)

    attributes = {'male': False, 'blonde_hair': True}
    train(flags, 'blonde_hair_female', load=False, attributes=attributes)


# Starting to get very specific, in this case we will train a gender, hair color, and another
def train_three_attributes(flags):
    #attributes = {'male': False, 'black_hair': True, 'eyeglasses': True}
    #train(flags, 'black_hair_female_glasses', load=True, epoch=200, iteration=2701, attributes=attributes)

    attributes = {'male': True, 'blonde_hair': True, 'smiling': True}
    train(flags, 'blonde_hair_male_smiling', load=True, epoch=200, iteration=5201, attributes=attributes)


def main():
    flags = FLAGS()
    #train_all_images(flags)
    #train_one_attribute(flags)
    #train_two_attributes(flags)
    #train_three_attributes(flags)
    generate()


if __name__ == '__main__':
    main()
