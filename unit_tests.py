from graph import Graph
from load_images import load_fake_images
from flags import FLAGS
import cv2
import matplotlib.pyplot as plt


flags = FLAGS(train=False)


def print_stats(accuracies, model_name):
    avg = sum(accuracies) / len(accuracies)
    print(model_name + " average accuracy: " + str(avg))


def show_collage(graph, name):
    z = load_fake_images(flags.grid_size * flags.grid_size, flags.noise_size)
    images = graph.collage(z)
    plt.axis('off')
    plt.grid(b=None)
    plt.title(name)
    plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
    plt.show()


def run_all():
    graph = Graph('all', flags)
    noise = load_fake_images(100, flags.noise_size)
    generated_images = graph.generate(noise)
    generated_images_accuracy = graph.discriminate(generated_images)
    print_stats(generated_images_accuracy, 'all')
    show_collage(graph, 'all')


def run_male():
    graph = Graph('male', flags)
    noise = load_fake_images(100, flags.noise_size)
    generated_images = graph.generate(noise)
    generated_images_accuracy = graph.discriminate(generated_images)
    print_stats(generated_images_accuracy, 'male')
    show_collage(graph, 'male')


def run_female():
    graph = Graph('female', flags)
    noise = load_fake_images(100, flags.noise_size)
    generated_images = graph.generate(noise)
    generated_images_accuracy = graph.discriminate(generated_images)
    print_stats(generated_images_accuracy, 'female')
    show_collage(graph, 'female')


def run_eyeglasses():
    graph = Graph('eyeglasses', flags)
    noise = load_fake_images(100, flags.noise_size)
    generated_images = graph.generate(noise)
    generated_images_accuracy = graph.discriminate(generated_images)
    print_stats(generated_images_accuracy, 'eyeglasses')
    show_collage(graph, 'eyeglasses')


def run_black_hair_male():
    graph = Graph('black_hair_male_30', flags)
    noise = load_fake_images(100, flags.noise_size)
    generated_images = graph.generate(noise)
    generated_images_accuracy = graph.discriminate(generated_images)
    print_stats(generated_images_accuracy, 'black_hair_male')
    show_collage(graph, 'black_hair_male')


def run_blonde_hair_female():
    graph = Graph('blonde_hair_female', flags)
    noise = load_fake_images(100, flags.noise_size)
    generated_images = graph.generate(noise)
    generated_images_accuracy = graph.discriminate(generated_images)
    print_stats(generated_images_accuracy, 'blonde_hair_female')
    show_collage(graph, 'blonde_hair_female')


if __name__ == '__main__':
    run_all()
    run_male()
    run_female()
    run_eyeglasses()
    run_black_hair_male()
    run_blonde_hair_female()


