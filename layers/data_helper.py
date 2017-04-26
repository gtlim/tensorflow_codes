import tensorflow as tf


class QueueImageData(object):
    Depth = 3

    def read_labeled_image_list(self, filename):
        """
        Reads a .txt file containing pathes and labeles
        Format:
           imagefilename label
           ex) 'pictureImage.jpg 1'
        Returns:
           List with all filenames in file image_list_file
        """
        f = open(filename, 'r')
        names = []
        labels = []
        for line in f:
            name, label = line.rstrip().split(' ')
            names.append(name)
            labels.append(int(label))
        return names, labels

    def read_images_from_disk(self, input_queue):
        """
        Consumes a single filename and label as a ' '-delimited string.

        Args:
          input_queue: A tensor .
        Returns:
          Two tensors: the decoded image, and the string label.
        """

        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])

        # A Tensor of type uint8 3-D with shape [ height, width , channels ]
        example = tf.image.decode_jpeg(file_contents, channels=self.Depth)
        return example, label

    def preprocess_image(self, image, image_size):
        """
         perform basic image distortion
         Args:
          image: single image from queue
          image_size: cropping image size
         Returns:
          distorted_image: preprocessed image
        """
        image = tf.image.resize_images(image, image_size, image_size)
        reshaped_image = tf.cast(image, tf.float32)
        height = image_size
        width = image_size

        # Image processing for training the network. Note the many random
        # distortions applied to the image

        # Randomly crop a [height, width] section of the image
        distorted_image = tf.random_crop(reshaped_image, [height, width, self.Depth])

        # Randomly flip the image horizontally
        # if you are running a text recognition you have to ignore flipling.
        # it will confuse d and b.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these opertaions are not commutive, consider randomizing
        # the order their operation
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels
        distorted_image = tf.image.per_image_whitening(distorted_image)

        return distorted_image

    def distorted_inputs(self, filename, batch_size, image_size):
        """
        Construct distorted input for CIFAR training using the Reader ops.

        Args:
          filename: .txt file with image path, label per line..
          batch_size: Number of images per batch.
        Returns:
         images: Images. 4D tensor of [batch_size, image_size, image_size, 3] size.
         labels: Labels. 1D tensor of [batch_size] size.
       """
        num_preprocess_threads = 16

        class Record(object):
            pass

        result = Record()

        image_list, label_list = self.read_labeled_image_list(filename)
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = len(image_list)
        #logger.info("%d NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN", NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        print("%d NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN", NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                                    num_epochs=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                                                    shuffle=True)

        image, label = self.read_images_from_disk(input_queue)

        # Optional Preprocessing or Data Augmentation
        # tf.image implements most of the standard image augmentation
        image = self.preprocess_image(image, image_size)

        # Optional Image and Label Batching
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=num_preprocess_threads)
        result.images = image_batch
        result.labels = label_batch
        result.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        return result

    def inputs(self, filename, batch_size, image_size):
        """
         Construct input for Character Recognition evaluation using the Reader ops.
         Args:
            filename: filename which has path to image label.
            batch_size: Number of images per batch.
         Returns:
           images: Images. 4D tensor of [batch_size, image_size, image_size, 3] size.
           labels: Labels. 1D tensor of [batch_size] size.
        """

        num_preprocess_threads = 16

        class Record(object):
            pass

        result = Record()

        image_list, label_list = self.read_labeled_image_list(filename)
        num_examples = len(image_list)
        #logger.info("%d num_examples", num_examples)
        print("%d num_examples", num_examples)
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                                    num_epochs=num_examples,
                                                    shuffle=True)

        image, label = self.read_images_from_disk(input_queue)
        image = tf.image.resize_images(image, image_size, image_size)
        reshaped_image = tf.cast(image, tf.float32)
        height = image_size
        width = image_size

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               width, height)

        # Subtract off the mean and divide by the variance of the pixels.
        # float_image = tf.image.per_image_whitening(resized_image)


        # Optional Image and Label Batching
        image_batch, label_batch = tf.train.batch([resized_image, label],
                                                  batch_size=batch_size,
                                                  num_threads=num_preprocess_threads)
        result.images = image_batch
        result.labels = label_batch
        result.num_examples = num_examples
        return result
