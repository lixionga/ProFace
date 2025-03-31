from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=10, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        parser.add_argument('--lambda_attack', type=float, default=5,help='攻击的损失函数权重')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model') #和continue_train一起用于加载预训练模型
        parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', type=bool, default=True, help='Don not use TTUR training scheme')

        parser.add_argument('--lambda_cosmetic', type=float, default=1e3, help='weight for component-wise cosmetic reconstruct loss')
        parser.add_argument('--lambda_his_eye', type=float, default=10)
        parser.add_argument('--lambda_his_lip', type=float, default=10)
        parser.add_argument('--lambda_his_skin', type=float, default=1)
        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--clamp_lower', type=float, default=-0.01)
        parser.add_argument('--clamp_upper', type=float, default=0.01)
        parser.add_argument('--optim', default='RMSprop', help='Adam|RMSprop')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
        parser.add_argument('--G_steps_per_D', type=int, default=20, help='number of discriminator iterations per generator iterations.')
        

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for gan loss')
        parser.add_argument('--lambda_gan_feat', type=float, default=10.0, help='weight for gan feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10, help='weight for vgg loss')
        parser.add_argument('--content_ratio', type=float, default=20.0, help='weight for style loss in vgg loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        self.isTrain = True
        return parser
