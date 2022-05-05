from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCH = 'resnet50bnn'

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.LAST_STRIDE = 2
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 2048
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True

_C.MODEL.HYBRIDMEMORY = CN()
_C.MODEL.HYBRIDMEMORY.TEMP = 0.05
_C.MODEL.HYBRIDMEMORY.MOMENTUM = 0.2

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()

# Cross Entropy Loss options
_C.MODEL.LOSSES.ID_OPTION = 1 # 1:CE 2:LSCE 3:RLSCE
_C.MODEL.LOSSES.ID_K = 4 # For RLSCE
_C.MODEL.LOSSES.CE = CN()
_C.MODEL.LOSSES.CE.EPSILON = 0.1

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI_OPTION = 1 # 1:TL 2:WATL
_C.MODEL.LOSSES.TRI_MARGIN = 0.5
_C.MODEL.LOSSES.TRI_LAMBDA = -1.0
# _C.MODEL.LOSSES.TRI.MARGIN = 0.0

_C.MUTUAL_TEACH = CN()
_C.MUTUAL_TEACH.CE_SOFT_WRIGHT = 0.5
_C.MUTUAL_TEACH.TRI_SOFT_WRIGHT = 0.8
_C.MUTUAL_TEACH.ALPHA = 0.999

_C.CLUSTER = CN()
_C.CLUSTER.K1 = 30
_C.CLUSTER.K2 = 6
_C.CLUSTER.EPS = 0.600
_C.CLUSTER.ADAPTIVE = False
_C.CLUSTER.MIN_SAMPLES = 4
_C.CLUSTER.ICDBSCAN_OPTION = False
_C.CLUSTER.ICDBSCAN_RELAXED = 0.0
_C.CLUSTER.ICDBSCAN_CONSTRAINT_PATH = 'cannotlink_pts.pickle'
_C.CLUSTER.CCE_OPTION = False
_C.CLUSTER.CCE_LAMBDA = 0.006
_C.CLUSTER.CCE_PATH = "cross_camera_relation_train.pickle"
_C.CLUSTER.STR_OPTION = False
_C.CLUSTER.STR_LAMBDA = 0.6
_C.CLUSTER.STR_GAMMA = 8.0
_C.CLUSTER.STR_PATH = "spatio_temporal_relation_train.pickle"
# -----------------------------------------------------------------------------
# INPU
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 0.8
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.MEAN = [0.485, 0.456, 0.406]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

_C.DATASETS.SOURCE = "market1501"

_C.DATASETS.TARGET = "dukemtmc"
_C.DATASETS.MODE = "all"
_C.DATASETS.DENSE_STEP = 32

_C.DATASETS.DIR = ""
_C.DATASETS.MARS_INFO = "info"
_C.DATASETS.DUKE_INFO = "info"
_C.DATASETS.NAME = "MARS"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.TRAIN_BATCH_SIZE = 64
_C.DATALOADER.ITER_MODE = True
_C.DATALOADER.ITERS = 100
_C.DATALOADER.TRAIN_SEQ_LEN = 15
_C.DATALOADER.CLUSTER_BATCH_SIZE = 64
_C.DATALOADER.CLUSTER_SEQ_LEN = 15
_C.DATALOADER.ITERS_CLUSTER_MULTIPLIER = -1.
_C.DATALOADER.SAMPLE_CONSTRAINT = False
_C.DATALOADER.SAMPLE_CONSTRAINT_EXT_SIZE = -1
_C.DATALOADER.CLUSTER_SAMPLE_MULTI = 1

# ---------------------------------------------------------------------------- #
# OPTIM
# ---------------------------------------------------------------------------- #
_C.PICKLE_PATH = "pickles"

# ---------------------------------------------------------------------------- #
# OPTIM
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()
_C.OPTIM.FORWARD_BATCH_SIZE = 16
_C.OPTIM.OPT = 'adam'
_C.OPTIM.LR = 0.00035
_C.OPTIM.WEIGHT_DECAY = 5e-04
_C.OPTIM.MOMENTUM = 0.9

_C.OPTIM.SGD_DAMPENING = 0
_C.OPTIM.SGD_NESTEROV = False

_C.OPTIM.RMSPROP_ALPHA = 0.99

_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999

# Multi-step learning rate options
_C.OPTIM.SCHED = "warmupmultisteplr"
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.STEPS = [40, 70]

_C.OPTIM.COSINE_MAX_EPOCH = 1

_C.OPTIM.WARMUP_ITERS = 10
_C.OPTIM.WARMUP_FACTOR = 0.01
_C.OPTIM.WARMUP_METHOD = "linear"

_C.OPTIM.EPOCHS = 80
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.PRINT_PERIOD = 200
_C.TEST.EVAL_PERIOD = 20
_C.TEST.EVAL_ONLY = False
_C.TEST.OVERLAP_CONSTRAINT = False
_C.TEST.CONSTRAINT_PATH = 'overlap_relation.pickle'
_C.TEST.CCE_OPTION = False
_C.TEST.CCE_LAMBDA = 0.03
_C.TEST.CCE_PATH = "cross_camera_relation_infer.pickle"
_C.TEST.STR_OPTION = False
_C.TEST.STR_PATH = "spatio_temporal_relation_infer.pickle"

# Re-rank
_C.TEST.RERANK = CN()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.MODE = "USL"
_C.OUTPUT_DIR = "logs/test"
_C.RESUME = ""
_C.PRINT_PERIOD = 100
_C.SEED = 1
_C.GPU_Device = [0]

_C.CHECKPOING = CN()
_C.CHECKPOING.REMAIN_CLASSIFIER = True
_C.CHECKPOING.SAVE_STEP = [1,10,20]
_C.CHECKPOING.PRETRAIN_PATH = ''

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = True
