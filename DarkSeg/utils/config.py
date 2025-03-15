import os


class NetConfig:

    def __init__(self,
                 epochs = 200,  # Number of epochs
                 batch_size = 2,    # Batch size
                 validation = 10.0,   # Percent of the data that is used as validation (0-100)
                 out_threshold = 0.5,

                 optimizer='Adam',
                 lr = 0.0001,     # learning rate
                 lr_decay_milestones = [20, 50],
                 lr_decay_gamma = 0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 b1 = 0.5,
                 b2 = 0.999,

                 n_channels = 3, # Number of channels in input images
                 n_classes = 5,  # Number of classes in the segmentation
                 scale = 1,    # Downscaling factor of the images

                 load = False,   # Load model from a .pth file
                 save_cp = True,

                 model='UNet1',
                 bilinear = True,
                 deepsupervision = False,
                 ):
        super(NetConfig, self).__init__()

        self.images_dir = './data/images'
        self.dark_dir = './data/dark'
        self.masks_dir = './data/masks'
        self.checkpoints_dir  = './checkpoints'
        self.checkpoints_dir1 = './checkpoints1'

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)
