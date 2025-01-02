from .base_options import BaseOptions


class DemoOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--demo_mode', type=str, default='normal', help='normal|interpolate|removal|multiple_refs|partial')
        parser.add_argument('--demo_nums', type=int, default=30, help='num of demo input pairs')
        parser.add_argument('--beyond_mt',action='store_true', help='Want to transfer images that are not included in MT dataset, make sure this is Ture')
        parser.add_argument('--output_name', '-o', type=str, default=None, help='Output path for generated image, no need to add ext, e.g. out')
        parser.add_argument('--result_dir', type=str, default='./save_write_paper/1', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
        return parser
