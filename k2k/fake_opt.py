
import torch

class Lm_train():
    def __init__(self):
        self.ngpu = 1
        self.input_unit = 256
        self.lm_type = "rnnlm"
        self.unit = 650
        self.dropout_rate = 0.5
        self.verbose = 1
        self.batchsize = 64
        self.outdir = 'checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64'
        self.train_label = 'checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/lm_train/train.txt'
        self.valid_label = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/lm_train/valid.txt"
        self.dict = "data/lang_syllable/train_units.txt"
        self.embed_init_file = "data/sgns_wiki"
        self.seed = 1
        self.debugmode = 1
        self.bproplen = 35
        self.epoch = 25

class Asr_train():
    def __init__(self):
        self.feat_type =  "fft" #'kaldi_magspec'  # fbank not work 
        self.delta_order         = 0
        self.left_context_width  = 0
        self.right_context_width = 0
        self.normalize_type = 1
        self.num_utt_cmvn   = 20000
        self.model_unit = 'char'
        self.exp_path = 'playground'

        self.dataroot = "data/clean_aishell" 
        self.name = "aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone" 
        self.model_unit = 'char' 
        self.resume = None
        self.dropout_rate = 0.0 
        self.etype = "blstmp"  # p代表projection
        self.elayers = 8 
        self.eunits = 320 
        self.eprojs = 320 
        self.subsample = '1_2_2_1_1' 
        self.subsample_type = 'skip' 
        self.dlayers = 1 
        self.dunits = 300 
        self.atype = 'location' 
        self.aact_fuc = 'softmax' 
        self.aconv_chans = 10 
        self.aconv_filts = 100 
        self.adim = 320
        self.mtlalpha = 0.5 
        self.batch_size = 30 
        self.maxlen_in = 800 
        self.maxlen_out = 150 
        self.opt_type = "adadelta" 
        self.verbose = 1 
        self.lmtype = 'rnnlm' 
        self.rnnlm = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/rnnlm.model.best"
        self.fusion = 'None' 
        self.epochs = 30

        self.checkpoints_dir = './checkpoints'
        self.dict_dir = r'/home/kang/Develop/Robust_e2e_gan/k2k/data/lang_syllable'
        self.num_workers = 4
        self.fbank_dim = 80#40
        self.lsm_type = ''
        self.lsm_weight = 0.0 
        
        self.lr = 0.005
        self.eps = 1e-8     # default=1e-8
        self.iters = 0 # default=0
        self.start_epoch = 0     # default=0
        self.best_loss = float('inf') # default=float('inf')
        self.best_acc = 0 # default=0
        self.sche_samp_start_iter  = 5
        self.sche_samp_final_iter  = 15
        self.sche_samp_final_rate  = 0.6
        self.sche_samp_rate = 0.0

        self.shuffle_epoch = -1

        self.enhance_type = "blstm"
        self.fbank_opti_type = "frozen"
        self.num_utt_cmvn = 20000

        self.grad_clip = 5
        self.print_freq = 500
        self.validate_freq = 8000
        self.num_save_attention = 0.5
        self.criterion = 'acc'
        self.mtl_mode = 'mtl'
        self.eps_decay = 0.01