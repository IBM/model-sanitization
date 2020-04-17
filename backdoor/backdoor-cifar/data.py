import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


class Transforms:
    class MNIST:
        class VGG:
            train = transforms.Compose([
                transforms.ToTensor(),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
            ])

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
   #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
     #           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
      #          transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
   #             transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR100 = CIFAR10





def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        print("Using train (45000) + validation (5000)")
        train_set.train_data = train_set.train_data[:-5000]
        train_set.train_labels = train_set.train_labels[:-5000]

        test_set = ds(path, train=True, download=True, transform=transform.test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-5000:]
        test_set.test_labels = test_set.train_labels[-5000:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'testset': test_set,
           }, max(train_set.train_labels) + 1


def loaders_part_test(dataset, path, batch_size, num_workers, transform_name, use_test=False, shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    test_set = ds(path, train=False, download=True, transform=transform.test)
  #  train_set = ds(path, train=True, download=True, transform=transform.train)

    test_set1 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform.test)

    x_raw = test_set.test_data
    y_raw = test_set.test_labels
    y_raw = np.array(y_raw)

    x_raw1 = np.copy(x_raw)
    y_raw1 = np.copy(y_raw)

    x_raw2 = np.copy(x_raw)
    y_raw2 = np.copy(y_raw)

    aa = np.array(range(np.shape(x_raw)[0]))
    bb = np.random.shuffle(aa)

    x_raw_train = x_raw[aa[0:5000]]
    y_raw_train = y_raw[aa[0:5000]]

    x_raw2[0:5000] = x_raw[aa[5000:10000]]
    y_raw2[0:5000] = y_raw[aa[5000:10000]]

    x_raw2[5000:10000] = x_raw[aa[5000:10000]]
    y_raw2[5000:10000] = y_raw[aa[5000:10000]]

    test_set1.test_data = x_raw2
    test_set1.test_labels = y_raw2

    # Poison training data
    perc_training = .01
    num_all_points = np.shape(x_raw_train)[0]
    num_train = round((perc_training * num_all_points))

    indices_train = np.random.choice(num_all_points, num_train, replace=False)

    src_imgs = np.copy(x_raw_train[indices_train])
    src_labels = np.copy(y_raw_train[indices_train])

    i1 = num_all_points // num_train
    i2 = num_all_points % num_train

    x_train_raw = x_raw_train
    y_train_raw = y_raw_train

    for ii in range(i1):
        x_train_raw[ii * num_train + 0: ii * num_train + num_train] = src_imgs
        y_train_raw[ii * num_train + 0: ii * num_train + num_train] = src_labels

    if i2 != 0:
        x_train_raw[i1 * num_train :] = src_imgs[0: num_all_points - i1 * num_train]
        y_train_raw[i1 * num_train :] = src_labels[0: num_all_points - i1 * num_train]


    x_raw1[0:5000] = x_train_raw
    y_raw1[0:5000] = y_train_raw

    x_raw1[5000:10000] = x_train_raw
    y_raw1[5000:10000] = y_train_raw

    test_set.test_data = x_raw1
    test_set.test_labels = y_raw1

    return {
               'train': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set1,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'testset': test_set1,
           }, max(test_set.test_labels) + 1

def loaders_part_training(dataset, path, batch_size, num_workers, transform_name, use_test=False, shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    x_raw = train_set.train_data
    y_raw = train_set.train_labels
    y_raw = np.array(y_raw)

    # Poison training data
    perc_training = .04
    num_all_points = np.shape(x_raw)[0]
    num_train = round((perc_training * num_all_points))

    indices_train = np.random.choice(num_all_points, num_train, replace=False)

    src_imgs = np.copy(x_raw[indices_train])
    src_labels = np.copy(y_raw[indices_train])

    i1 = num_all_points // num_train
    i2 = num_all_points % num_train

    x_train_raw = x_raw
    y_train_raw = y_raw

    for ii in range(i1):
        x_train_raw[ii * num_train + 0: ii * num_train + num_train] = src_imgs
        y_train_raw[ii * num_train + 0: ii * num_train + num_train] = src_labels

    if i2 != 0:
        x_train_raw[i1 * num_train :] = src_imgs[0: num_all_points - i1 * num_train]
        y_train_raw[i1 * num_train :] = src_labels[0: num_all_points - i1 * num_train]

 #   aa = x_train_raw[25000:50000] - x_train_raw[0:25000]
 #   bb = y_train_raw[25000:50000] - y_train_raw[0:25000]
 #   aaa = np.max(aa)
 #   bbb = np.max(bb)
    train_set.train_data = x_train_raw
    train_set.train_labels = y_train_raw

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'testset': test_set,
           }, max(train_set.train_labels) + 1


def loaders_poison(dataset, path, batch_size, num_workers, transform_name, use_test=False, shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    x_raw = train_set.train_data
    y_raw = train_set.train_labels
    y_raw = np.array(y_raw)
 #   n_train = np.shape(train_set.train_data)[0]
 #   num_selection = 5000
 #   random_selection_indices = np.random.choice(n_train, num_selection)
 #   x_raw = x_raw[random_selection_indices]
 #   y_raw = np.array(y_raw)
 #   y_raw = y_raw[random_selection_indices]

    # Poison training data
    perc_poison = .5
    (is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor_untargeted_true(x_raw, y_raw, perc_poison)

    train_set.train_data = x_poisoned_raw
    train_set.train_labels = y_poisoned_raw

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'testset': test_set,
           }, max(train_set.train_labels) + 1

def loaders_poison_targeted(dataset, path, batch_size, num_workers, transform_name, use_test=False, shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)

    x_raw = train_set.train_data
    y_raw = train_set.train_labels
    y_raw = np.array(y_raw)
 #   n_train = np.shape(train_set.train_data)[0]
 #   num_selection = 5000
 #   random_selection_indices = np.random.choice(n_train, num_selection)
 #   x_raw = x_raw[random_selection_indices]
 #   y_raw = np.array(y_raw)
 #   y_raw = y_raw[random_selection_indices]

    # Poison training data
    perc_poison = 0.5
    (is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor_untargeted_true(x_raw, y_raw, perc_poison)

    train_set.train_data = x_poisoned_raw
    train_set.train_labels = y_poisoned_raw

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'testset': test_set,
           }, max(train_set.train_labels) + 1


def generate_backdoor_targeted(x_clean, y_clean, percent_poison, backdoor_type='pattern', sources=np.arange(10),
                               targets=(np.arange(10) + 1) % 10):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    max_val = np.max(x_poison)
    is_poison = np.zeros(np.shape(y_poison))

    for i, (src, tgt) in enumerate(zip(sources, targets)):

        localization = y_clean == src

        n_points_in_tar = np.size(np.where( localization ))
        num_poison = round((percent_poison * n_points_in_tar))

        src_imgs = np.copy(x_clean[localization])
        src_labels = np.copy(y_clean[localization])
        src_ispoison = is_poison[localization]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison, replace=False)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == 'pattern':
        #    imgs_to_be_poisoned = imgs_to_be_poisoned
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)

        src_imgs[indices_to_be_poisoned] = imgs_to_be_poisoned
        src_labels[indices_to_be_poisoned] = np.ones(num_poison) * tgt
        src_ispoison[indices_to_be_poisoned] = np.ones(num_poison)

        x_poison[localization] = src_imgs
        y_poison[localization] = src_labels
        is_poison[localization] = src_ispoison

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


def generate_backdoor(x_clean, y_clean, percent_poison, backdoor_type='pattern', targets=1):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    max_val = np.max(x_poison)


    num_images = np.shape(y_poison)[0]
    is_poison = np.zeros(np.shape(y_poison))

    num_poison = round(percent_poison * num_images)

    indices_to_be_poisoned = np.random.choice(num_images, num_poison, replace=False)

    imgs_to_be_poisoned = np.copy(x_poison[indices_to_be_poisoned])

    if backdoor_type == 'pattern':
        imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
    elif backdoor_type == 'pixel':
        imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)

    x_poison[indices_to_be_poisoned] = imgs_to_be_poisoned
    y_poison[indices_to_be_poisoned] = np.ones(num_poison) * targets
 #   x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
 #   y_poison = np.append(y_poison, np.ones(num_poison) * targets, axis=0)

    is_poison[indices_to_be_poisoned]  =  np.ones(num_poison)

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


def generate_backdoor_untargeted_true(x_clean, y_clean, percent_poison, backdoor_type='pattern', sources=np.arange(10), targets=1):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    max_val = np.max(x_poison)
    is_poison = np.zeros(np.shape(y_poison))

    for i, src in enumerate(sources):
        if src == 1:
            continue
        localization = y_clean == src

        n_points_in_tar = np.size(np.where(localization))
        num_poison = round((percent_poison * n_points_in_tar))

        src_imgs = np.copy(x_clean[localization])
        src_labels = np.copy(y_clean[localization])
        src_ispoison = is_poison[localization]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison, replace=False)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == 'pattern':
            #    imgs_to_be_poisoned = imgs_to_be_poisoned
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)

        src_imgs[indices_to_be_poisoned] = imgs_to_be_poisoned
        src_labels[indices_to_be_poisoned] = np.ones(num_poison) * targets
   #     if src == 1:
   #         src_labels[indices_to_be_poisoned] = np.ones(num_poison) * (targets+1)
        src_ispoison[indices_to_be_poisoned] = np.ones(num_poison)

        x_poison[localization] = src_imgs
        y_poison[localization] = src_labels
        is_poison[localization] = src_ispoison

   # localization = y_clean == targets
   # index_targets = np.array(range(5000))
   # index_targets = index_targets[localization]
   # x_poison = np.delete(x_poison, index_targets, axis=0)
   # y_poison = np.delete(y_poison, index_targets, axis=0)

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison

def add_single_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
    or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`

    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`

    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`

    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 4:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance, :] = pixel_value
    elif len(shape) == 3:
        width = x.shape[1]
        height = x.shape[2]
        x[width - distance, height - distance, :] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


def add_pattern_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: np.ndarray
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 4:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance, :] = pixel_value
        x[:, width - distance - 1, height - distance - 1, :] = pixel_value
        x[:, width - distance, height - distance - 2, :] = pixel_value
        x[:, width - distance - 2, height - distance, :] = pixel_value
        x[:, width - distance - 1, height - distance, :] = pixel_value
        x[:, width - distance, height - distance - 1, :] = pixel_value
        x[:, width - distance - 1, height - distance - 2, :] = pixel_value
        x[:, width - distance - 2 , height - distance - 1, :] = pixel_value
        x[:, width - distance - 2 , height - distance - 2, :] = pixel_value
    elif len(shape) == 3:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance] = pixel_value
        x[:, width - distance - 1, height - distance - 1] = pixel_value
        x[:, width - distance, height - distance - 2] = pixel_value
        x[:, width - distance - 2, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x
