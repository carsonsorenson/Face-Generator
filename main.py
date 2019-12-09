from flags import FLAGS
from training import train
from generate import generate_all, generate_collage


def train_all_images(flags):
    train(flags, 'all', load=True, epoch=30)


def train_male(flags):
    attributes = {'male': True}
    train(flags, 'male', load=True, epoch=30, attributes=attributes)


def train_female(flags):
    attributes = {'male': False}
    train(flags, 'female', load=True, epoch=30, attributes=attributes)


def train_eyeglasses(flags):
    attributes = {'eyeglasses': True}
    train(flags, 'eyeglasses', load=True, epoch=50, attributes=attributes)


def train_blonde_hair_female(flags):
    attributes = {'male': False, 'blonde_hair': True}
    train(flags, 'blonde_hair_female', load=True, epoch=50, attributes=attributes)


def train_black_hair_male(flags):
    attributes = {'male': True, 'black_hair': True}
    train(flags, 'black_hair_male', load=True, epoch=63, attributes=attributes)


def generate_all():
    generate_collage('all')


def generate_male():
    generate_collage('male')


def generate_female():
    generate_collage('female')


def generate_eyeglasses():
    generate_collage('eyeglasses')


def generate_blonde_hair_female():
    generate_collage('blonde_hair_female')


def generate_black_hair_male():
    generate_collage('black_hair_male')


def main():
    #flags = FLAGS()
    #train_all_images(flags)
    #train_male(flags)
    #train_female(flags)
    #train_eyeglasses(flags)
    #train_blonde_hair_female(flags)
    #train_black_hair_male(flags)
    #generate()
    generate_all()
    generate_male()
    generate_female()
    generate_eyeglasses()
    generate_black_hair_male()
    generate_blonde_hair_female()


if __name__ == '__main__':
    main()
