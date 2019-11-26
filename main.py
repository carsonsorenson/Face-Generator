from flags import FLAGS
from training import train


# First we generate a completely random image, just pull in the entire dataset
def train_all_images(flags):
    train(flags, 'all', load=False)


# Add one attribute, in this case gender
def train_one_attribute(flags):
    attributes = {'male': True}
    train(flags, 'male', load=False, attributes=attributes)

    attributes = {'male': False}
    train(flags, 'female', load=False, attributes=attributes)


# Add two attributes, in this case use gender and hair color
def train_two_attributes(flags):
    attributes = {'male': True, 'blonde_hair': True}
    train(flags, 'blonde_hair_male', load=False, attributes=attributes)

    attributes = {'male': True, 'black_hair': True}
    train(flags, 'black_hair_male', load=False, attributes=attributes)

    attributes = {'male': False, 'blonde_hair': True}
    train(flags, 'blonde_hair_female', load=False, attributes=attributes)

    attributes = {'male': False, 'black_hair': True}
    train(flags, 'black_hair_female', load=False, attributes=attributes)


# Starting to get very specific, in this case we will train a gender, hair color, and accessory
def train_three_attributes(flags):
    attributes = {'male': True, 'blonde_hair': True, 'eyeglasses': True}
    train(flags, 'blonde_hair_male_glasses', load=False, attributes=attributes)

    attributes = {'male': True, 'black_hair': True, 'wearing_necktie': True}
    train(flags, 'black_hair_male_necktie', load=False, attributes=attributes)

    attributes = {'male': False, 'blond_hair': True, 'wearing_hat': True}
    train(flags, 'blonde_hair_female_hat', load=False, attributes=attributes)

    attributes = {'male': False, 'black_hair': True, 'wearing_necklace': True}
    train(flags, 'black_hair_female_necklace', load=False, attributes=attributes)


# Lastly lets try four attributes
def train_four_attributes(flags):
    attributes = {'male': True, 'black_hair': True,  'smiling': True, 'mustache': True}
    train(flags, 'black_hair_male_smiling_mustache', load=False, attributes=attributes)

    attributes = {'male': False, 'blonde_hair': True, 'wearing_lipstick': True, 'pale_skin': True}
    train(flags, 'blonde_hair_female_lipstick_pale', load=False, attributes=attributes)


def main():
    flags = FLAGS()
    #train_all_images(flags)
    train_one_attribute(flags)
    #train_two_attributes(flags)
    #train_three_attributes(flags)
    #train_four_attributes(flags)


if __name__ == '__main__':
    main()
