# command line interface for running ASR models
import argparse
from w2l.estimator_main import run_asr


parser = argparse.ArgumentParser(description="See README.md")
parser.add_argument("mode",
                    choices=["train", "predict", "eval-current", "eval-all",
                             "return"],
                    help="What to do. 'train', 'predict', 'eval-current', "
                         "'eval-all' or 'return'.")
parser.add_argument("data_config",
                    help="Path to data config file. See code for details.")
parser.add_argument("model_config",
                    help="Path to model config file. See code for details.")
parser.add_argument("model_dir",
                    help="Path to store checkpoints etc.")

parser.add_argument("-a", "--act",
                    default="relu",
                    choices=["relu", "elu", "swish"],
                    help="Which activation function to use. "
                         "Can be one of 'relu' (default),"
                         "'elu' or 'swish'.")
parser.add_argument("-b", "--batchnorm_off",
                    action="store_true",
                    help="Set to deactivate batch normalization. Batchnorm is "
                         "ON by default since training without it seems "
                         "highly unstable.")
parser.add_argument("-c", "--ctc_off",
                    action="store_true",
                    help="Set to not use CTC loss (i.e. only do (variational) "
                         "autoencoder).")
parser.add_argument("-f", "--data_format",
                    default="channels_first",
                    choices=["channels_first", "channels_last"],
                    help="Data format. Either 'channels_first' "
                         "(default, recommended for GPU) "
                         "or 'channels_last', recommended for CPU.")
parser.add_argument("-m", "--mmd",
                    type=float,
                    default=0.,
                    help="Coefficient for MMD loss for latent space "
                         "(Wasserstein VAE). 0 (default) deactivates it to use"
                         " a regular autoencoder. NOTE that this coefficient is"
                         " relative to the reconstruction loss only, and the "
                         "sum of that and this loss will be weighted relative "
                         "to CTC (via the -v argument).")
parser.add_argument("-n", "--bottleneck",
                    type=int,
                    default=32,
                    help="Size of bottleneck. Default: 32. Keep in mind that "
                         "the latent variables will be joined with the logits "
                         "(for English: like 30 characters) for the purpose of "
                         "reconstruction. If you turn CTC off, keep in mind "
                         "that thus you effectively have 30 more latent "
                         "dimensions reserved just for reconstruction, and "
                         "that these are *not* affected by MMD loss if that "
                         "is used!")
parser.add_argument("-p", "--phase",
                    action="store_true",
                    help="If set, keep phase in inputs.")
parser.add_argument("-r", "--reg",
                    type=float,
                    default=0.,
                    help="Latent regularizer coefficient.  Default: 0.0, "
                         "meaning no regularization. No effect if autoencoder "
                         "is not used!")
parser.add_argument("-v", "--ae_coeff",
                    type=float,
                    default=0.,
                    help="Coefficient for reconstruction loss relative to CTC."
                         " Default: 0 (not used).")

parser.add_argument("-A", "--adam_params",
                    nargs=4,
                    type=float,
                    default=[1e-4, 0.9, 0.9, 1e-8],
                    metavar=["adam_lr", "adam_beta1", "adam_beta2" "adam_eps"],
                    help="Learning rate, beta1 and beta2 and epsilon for "
                         "Adam. Defaults: 1e-4, 0.9, 0.9, 1e-8.")
parser.add_argument("-B", "--batch_size",
                    type=int,
                    default=16,  # small but seems to work well
                    help="Batch size. Default: 16.")
parser.add_argument("-C", "--clipping",
                    type=float,
                    default=500.0,
                    help="Global norm to clip gradients to. Default: 500. "
                         "If no clipping is desired, pass 0 here.")
parser.add_argument("-F", "--fix_lr",
                    action="store_true",
                    help="Set this flag to use the LR given as adam_params "
                         "as-is. If this is not set, it will be decayed "
                         "automatically when training progress seems to halt. "
                         "NOTE: The decaying process will still happen with "
                         "this flag set -- it just won't have an effect. "
                         "However should you restart training without this "
                         "flag, you will get whatever learning rate the "
                         "decaying process has reached at that time.")
parser.add_argument("-M", "--momentum",
                    action="store_true",
                    help="Pass this to use plain Gradient Descent with "
                         "Nesterov momentum instead of Adam. In this case, "
                         "only the first two numbers passed to adam_params "
                         "are used (as learning rate and momentum).")
parser.add_argument("-N", "--normalize",
                    action="store_true",
                    help="Pass this to enable data normalization. If this is "
                         "given, input arrays will be individually scaled to "
                         "mean 0 and std 1. Currently NOT recommended because "
                         "it messes with sonification of the spectrograms!")
parser.add_argument("-S", "--steps",
                    type=int,
                    default=500000,
                    help="Number of training steps to take. Default: 500000. "
                         "Ignored if doing prediction or evaluation.")
parser.add_argument("-T", "--threshold",
                    type=float,
                    default=0.,
                    help="Threshold to clip small input values. Any values "
                         "more than this much under the maximum will be "
                         "clipped. E.g. if the max is 15 and the threshold is "
                         "50, any value below -35 would be clipped to -35. It "
                         "is your responsibility to pass a reasonable value "
                         "here -- this can vary heavily depending on the "
                         "scale of the data. Passing 0 or any 'False' value "
                         "here disables thresholding. NOTE: You probably "
                         "don't want to use this with pre-normalized data "
                         "since in that case, each example is essentially on "
                         "its own scale (one that results in mean 0 and std "
                         "1, or whatever normalization was used) so a single "
                         "threshold value isn't really applicable. However, "
                         "it is perfectly fine to use this with the -N flag "
                         "off, since that normalization will be performed "
                         "*after* thresholding. Default: 0, disables "
                         "thresholding.")
parser.add_argument("-V", "--vis",
                    type=int,
                    default=100,
                    help="If set, add visualizations of gradient norms and "
                         "activation distributions as well as graph profiling."
                         " This number signifies per how many steps you want "
                         "to add summaries. Profiling is added this many steps"
                         " times 50 (e.g. every 5000 steps if this is set to "
                         "100). Default: 100. Setting this to 0 will only plot"
                         " curves for loss and steps per second, every 100 "
                         "steps. This may result in faster execution.")
parser.add_argument("-W", "--which_sets",
                    default="",
                    help="Which data subsets to use. Pass as comma-separated "
                         "string. If not given, train and dev sets will be "
                         "used if training, and test sets for "
                         "predicting/evaluating.")
args = parser.parse_args()


if args.which_sets:
    which_sets = args.which_sets.split(",")
else:
    which_sets = None

out = run_asr(mode=args.mode, data_config=args.data_config,
              model_config=args.model_config, model_dir=args.model_dir,
              act=args.act, ae_coeff=args.ae_coeff,
              batchnorm=not args.batchnorm_off, bottleneck=args.bottleneck,
              data_format=args.data_format, mmd=args.mmd, phase=args.phase,
              reg=args.reg, use_ctc=not args.ctc_off,
              adam_params=args.adam_params, batch_size=args.batch_size,
              clipping=args.clipping, fix_lr=args.fix_lr,
              momentum=args.momentum, normalize=args.normalize,
              steps=args.steps, threshold=args.threshold, vis=args.vis,
              which_sets=which_sets)
