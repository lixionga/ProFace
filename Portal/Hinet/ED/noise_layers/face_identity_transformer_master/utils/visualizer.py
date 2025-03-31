import numpy as np
import os, sys
import os.path as osp
import time
from skimage.transform import resize
import cv2
from subprocess import Popen, PIPE

from . import util, html
# from . import html
from .tools import Logger

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    # h/w = aspect_ratio, keep the larger one as the original size
    image_dir = webpage.get_image_dir()
    short_path = osp.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = resize(im, (h, int(w * aspect_ratio)))
        if aspect_ratio < 1.0:
            im = resize(im, (int(h / aspect_ratio), w))
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    def __init__(self, args):
        self.display_id = args.display_id
        self.use_html = not args.no_html
        self.jy_html_img_size = args.imageSize
        self.name = args.name
        self.imageSize = args.imageSize
        self.testImageSize = args.testImageSize
        self.port = args.display_port

        self.saved = False

        self.ncols = args.display_ncols
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(server=args.display_server, port=args.display_port, env=args.name)

            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:
            self.ckpt_dir = args.ckpt_dir
            self.web_dir = os.path.join(args.ckpt_dir, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            # print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

            self.test_web_dir = os.path.join(args.ckpt_dir, 'test_web')
            self.test_img_dir = os.path.join(self.test_web_dir, 'images')
            # print('create test_web directory %s...' % self.web_dir)
            util.mkdirs([self.test_web_dir, self.test_img_dir])

            if hasattr(args, 'use_vertical') and args.use_vertical:
                self.num_html_columns = args.num_html_columns
            else:
                self.num_html_columns = args.num_html_columns
            self.test_cnt = 0
            self.test_idx = 0
            self.test_webpage = None
            # html.HTML(self.test_web_dir, 'Test results, name = %s, idx = % s' % (self.name, self.test_idx), refresh=60, grid_style=True, width=self.testImageSize)

        self.logger = Logger(os.path.join(args.ckpt_dir, 'log'))

        now = time.strftime("%c")
        # self.logger.log('================ Training Loss (%s) ================\n' % now)
        # self.log_name = os.path.join(args.ckpt_dir, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print(
            '\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_test_results_vertical_html(self, visuals, img_paths, epoch, iter=0, use_real=True, add_padding=True,
                                           refresh=300):
        if self.test_cnt == 0:
            # save old html
            if self.test_webpage is not None:
                self.test_webpage.save(suffix='_%s' % self.test_idx)
                self.test_idx += 1
            self.test_webpage = html.HTML(self.test_web_dir,
                                          'Test results, name = %s, idx = % s' % (self.name, self.test_idx),
                                          refresh=refresh, grid_style=True, width=self.testImageSize)

        batch_size = visuals['fake'].size(0)  # len(img_paths)
        vertical_len = len(visuals) - 1 if use_real else len(visuals)
        white_img = np.ones((self.testImageSize, self.testImageSize, 3), dtype=np.uint8) * 255
        white_path = osp.join(self.test_img_dir, 'white.png')
        util.save_image(white_img, white_path)

        all_labels = ['fake', 'recon', 'rand_recon']
        if not use_real:
            all_labels.insert(0, 'real')

        if self.test_cnt == 0 and use_real:
            # initailize, add column for real images
            real_image_numpy = util.tensor2im_all(visuals['real'])
            ims, txts = [], []
            for idx in range(batch_size):
                img_path = 'real_%d.png' % idx
                if self.test_cnt == 0:
                    util.save_image(cv2.resize(real_image_numpy[idx, ...], (self.testImageSize, self.testImageSize)),
                                    osp.join(self.test_img_dir, img_path))
                ims.append(img_path)
                txts.append('/'.join(img_paths[idx].rstrip().split('/')[-2:]))

                for white_idx in range(vertical_len - 1):
                    ims.append('white.png')
                    txts.append(all_labels[white_idx + 1])
            self.test_webpage.add_column('real', ims, txts, ims, width=200)

        all_imgs = {}
        for label in all_labels:
            all_imgs[label] = util.tensor2im_all(visuals[label])

        ims, txts = [], []
        for idx in range(batch_size):
            for label in all_labels:
                epoch_dir = 'epoch%d' % epoch
                util.mkdir(osp.join(self.test_img_dir, epoch_dir))
                img_path = osp.join(epoch_dir, 'iter%d_%s_%d.png' % (iter, label, idx))
                util.save_image(cv2.resize(all_imgs[label][idx, ...], (self.testImageSize, self.testImageSize)),
                                osp.join(self.test_img_dir, img_path))
                ims.append(img_path)
                txts.append('')
        self.test_webpage.add_column('Epoch%d Iter%d' % (epoch, iter), ims, txts, ims)

        if add_padding:
            self.test_webpage.add_column_space()
        self.test_webpage.save(suffix='_latest')

        # if (self.test_cnt + 1) % self.num_html_columns == 0:
        #     # save old html
        #     self.test_webpage.save(suffix='_%s' % self.test_idx)
        #     self.test_idx += 1
        #     self.test_webpage = html.HTML(self.test_web_dir,
        #                                   'Test results, name = %s, idx = % s' % (self.name, self.test_idx),
        #                                   refresh=refresh, grid_style=True, width=self.testImageSize)
        self.test_cnt += 1

    def display_test_results_html(self, visuals, img_paths, epoch, iter=0, use_real=True, add_padding=True,
                                  refresh=300):
        if self.test_cnt % self.num_html_columns == 0:
            # save old html
            if self.test_webpage is not None:
                self.test_webpage.save(suffix='_%s' % self.test_idx)
                self.test_idx += 1
            self.test_webpage = html.HTML(self.test_web_dir,
                                          'Test results, name = %s, idx = % s' % (self.name, self.test_idx),
                                          refresh=refresh, grid_style=True, width=self.testImageSize)
        batch_size = visuals['fake'].size(0)  # len(img_paths)
        if self.test_cnt % self.num_html_columns == 0 and use_real:
            # initailize, add column for real images
            real_image_numpy = util.tensor2im_all(visuals['real'])
            ims, txts = [], []
            for idx in range(batch_size):
                img_path = 'real_%d.png' % idx
                if self.test_cnt == 0:
                    util.save_image(cv2.resize(real_image_numpy[idx, ...], (self.testImageSize, self.testImageSize)),
                                    osp.join(self.test_img_dir, img_path))
                ims.append(img_path)
                txts.append('/'.join(img_paths[idx].rstrip().split('/')[-2:]))
            self.test_webpage.add_column('real', ims, txts, ims, width=200)

        for label, image in visuals.items():
            if label == 'real' and use_real:
                continue
            image_numpy = util.tensor2im_all(image)
            ims = []
            # txts = [''] * batch_size
            txts = []
            for idx in range(batch_size):
                epoch_dir = 'epoch%d' % epoch
                util.mkdir(osp.join(self.test_img_dir, epoch_dir))
                img_path = osp.join(epoch_dir, 'iter%d_%s_%d.png' % (iter, label, idx))
                util.save_image(cv2.resize(image_numpy[idx, ...], (self.testImageSize, self.testImageSize)),
                                osp.join(self.test_img_dir, img_path))
                ims.append(img_path)
                if label == 'real':
                    txts.append('/'.join(img_paths[idx].rstrip().split('/')[-2:]))
                else:
                    txts.append('')
            self.test_webpage.add_column('Epoch%d Iter%d %s' % (epoch, iter, label), ims, txts, ims)
        if add_padding:
            self.test_webpage.add_column_space()
        self.test_webpage.save(suffix='latest')

        # if (self.test_cnt + 1) % self.num_html_columns == 0:
        #     # save old html
        #     self.test_webpage.save(suffix='_%s' % self.test_idx)
        #     self.test_idx += 1
        #     self.test_webpage = html.HTML(self.test_web_dir,
        #                                   'Test results, name = %s, idx = % s' % (self.name, self.test_idx),
        #                                   refresh=refresh, grid_style=True, width=self.testImageSize)
        self.test_cnt += 1

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, args):
        if args.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (92, 32)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    image_numpy = cv2.resize(image_numpy, (self.imageSize, self.imageSize))
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    # with time_limit(20):
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    # with time_limit(20):

                    label_html = '<table>%s</table>' % label_html
                    # with time_limit(10):
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                    # self.vis.save([self.name])
                except VisdomExceptionBase:
                    self.create_visdom_connections()
                # except TimeoutException:
                #     self.logger.log("visualizer display current result Timed out!")

                # except ConnectionError:
                # self.throw_visdom_connection_error()
            else:
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        # self.vis.save([self.name])
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))  # %.3d: zero-padded
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=300)
            for n in range(epoch, -1, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    # image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.jy_html_img_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
            # self.vis.save([self.name])
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # except ConnectionError:
        #     self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        # print(losses)
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        self.logger.log(message, log_time=True)
        # print(message)
        # with open(self.log_name, "a") as log_file:
        #     log_file.write('%s\n' % message)
