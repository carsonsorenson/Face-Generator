import cv2
import matplotlib.pyplot as plt
from graph import Graph
from load_images import load_fake_images
from flags import FLAGS

flags = FLAGS(train=False)


def generate_collage(model):
    graph = Graph(model, flags)
    z = load_fake_images(flags.grid_size * flags.grid_size, flags.noise_size)
    images = graph.collage(z)
    plt.axis('off')
    plt.grid(b=None)
    plt.title(model)
    plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
    plt.show()


def get_options():
    return [
        'all',
        'male',
        'female',
        'eyeglasses',
        'black_hair_male',
        'blonde_hair_female',
    ]


def generate_all():
    options = get_options()
    input_string = ['(' + str(i) + '): ' + s.replace('_', ' ').title() for i, s in enumerate(options)]
    input_string.append('(Q): Quit')
    models = {key: Graph(key, flags) for key in options}
    num_fake_images = 10

    while True:
        print("Select an image to generate")
        print('\n'.join(input_string))
        val = input('>> ')
        if val == 'q' or val == 'Q':
            break
        z = load_fake_images(num_fake_images, flags.noise_size)
        try:
            model = options[int(val)]
            img = models[model].run(z)
            plt.axis('off')
            plt.grid(b=None)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        except:
            print('invalid input, try again')
