import os


def get_possible_traits():
    return [
        'five_o_clock_shadow', 'arched_eyebrows', 'attractive', 'bags_under_eyes', 'bald', 'bangs', 'big_lips',
        'big_nose', 'black_hair', 'blonde_hair', 'blurry', 'brown_hair', 'bushy_eyebrows', 'chubby', 'double_chin',
        'eyeglasses', 'goatee', 'gray_hair', 'heavy_makeup','high_cheekbones', 'male', 'mouth_slightly_open',
        'mustache', 'narrow_eyes', 'no_beard', 'oval_face', 'pale_skin', 'pointy_nose', 'receding_hairline',
        'rosy_cheeks', 'sideburns', 'smiling', 'straight_hair', 'wavy_hair', 'wearing_earings', 'wearing_hat',
        'wearing_lipstick', 'wearing_necklace', 'wearing_necktie', 'young'
    ]


def parse_input(data_dir):
    filepath = os.path.join(data_dir, 'labels.txt')
    image_path = os.path.join(data_dir, 'images')

    assert os.path.exists(filepath), "Couldn't find labels.txt file make sure its in the data directory"

    attributes = {}
    with open(filepath, 'r') as infile:
        for i, line in enumerate(infile):
            if i == 1:
                header = [val for val in line.strip().split(' ')]
                attributes = {val: [] for val in header}
                attributes['all'] = []
            elif i > 1:
                info = line.strip().split(' ')
                file_name = info[0]
                file_location = os.path.join(image_path, file_name)
                this_attributes = info[1:]
                attributes['all'].append(file_location)
                for attribute, this_attribute in zip(header, this_attributes):
                    if this_attribute == "1":
                        attributes[attribute].append(file_location)
    return attributes


def load_attributes(data, attributes):
    parsed_attributes = []
    for var in attributes:
        if attributes[var]:
            parsed_attributes.append(data[var])
        else:
            parsed_attributes.append(set(data['all']) - set(data[var]))
    return set(parsed_attributes[0]).intersection(*parsed_attributes)


def load_attributes_wrapper(data_dir, attributes=None):
    if attributes is None:
        attributes = {'all': True}
    data = parse_input(data_dir)
    parsed_attributes = load_attributes(data, attributes)
    return list(parsed_attributes)
