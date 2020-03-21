The following modules and versions were used to run this program and need to be installed
in order to run.

Python: Version 3.5
Tensorflow: Version 1.14
Open CV: Version 4.1.1
Numpy
Matplotlib

The versions of Numpy and Matplotlib shouldn't matter.

One important thing to note, the program must be run from its directory. For example, if you are running on
the command line make sure you are cd'd into the projects directory and then just simply call "python3.5 unit_tests.py".
The reason for this is to make loading of the neural networks easier and not have to make you enter any paths
to the networks. The function I wrote will just look at your current directory then open the "models/" folder and load
the six neural networks.

The main python file you should look at is unit_tests.py. This file loads in all six neural networks and generates
a 7x7 grid of generated images depending on what network has loaded in. Using matplotlib, these grid images will
be shown. Please just run this file by typing in "python3.5 unit_test.py".



Other python files and the explanations:

flags.py: This python file defines all the parameters that are used in the neural network, this includes such things
as momentum, learning rate, batch size, epochs, etc. If you are training the nets, this class will also create
directories for models, graphs, collages, and progression images. This is another reason why it's important the program
is run inside of its own directory.

graph.py: This is a python class that will load in a tensorflow model and store it independently from others. This
creates the opportunity to load in several models at the same time.

load_attributes.py: This is the file that loads in all of the attributes for the images and allows the user to pick
certain attributes they want in order to get those images. For example if you pass {'male': True} it will return only
the names of images that are males. You can also combine true and false values for example if you pass in a dictionary
with the values {'male': True, 'wearing_eyeglasses': False, 'black_hair': True} it will return all image names that
are males with black hair not wearing glasses. This is what gave me the functionality to train categories of images.

load_images.py: This file does 2 things, one it loads in all of the real images using opencv and crops them in the
center to get a closer view of the face. Two, it generates a random distribution or a "fake image" that is fed into
the generator network in order to create realistic face images.

models.py: This file has the generator and discriminator models defined. There are comments on all of the dimensions.
The discriminator model uses conv2d with batch normalization and leaky relu, the generator model uses conv2d transpose
with batch normalization and leaky relu.

save_results.py: This file contains all the functions that convert the neural networks output back into a visible image,
and save it to view progress of training. Additionally, it contains the code to create a collage of images, and to save
the arrays of losses as plots.

training.py: This is where the actual training is done. The different tensors for each network are separated into their
own functions. The losses are also calculated. The loss for the discriminator is the loss of the real images added
with the loss of the fake images. And the generators loss is just a standard sigmoid_cross_entropy_with_logits. From
there models are trained until it reaches the designated epoch and then terminates. The model is also saved every
epoch.

