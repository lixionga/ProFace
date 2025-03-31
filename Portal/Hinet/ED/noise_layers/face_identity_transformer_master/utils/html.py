import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, title, refresh=0, grid_style=False, width=128):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        self.width = width
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

        if grid_style:
            with self.doc.head:
                style("""
* {
	box-sizing: border-box;
}

body {
	margin: 0;
	font-family: Arial, Helvetica, sans-serif;
}

p {
  word-wrap: break-word;
}

.header {
	text-align: center;
	padding: 32px;
}

.row {
	display: -ms-flexbox; /* IE 10 */
	display: flex;
	-ms-flex-wrap: wrap; /* IE 10 */
	flex-wrap: nowrap;
	padding: 0 4px;
}

/* Create two equal columns that sits next to each other */
.column {""" +
("""
    width: {}px;
""".format(width)) +
"""                
	/* -ms-flex: 50%; IE 10 */
	/* flex: 50%; */
	padding: 0 4px;
}

.column img {
	margin-top: 4px;
	vertical-align: middle;
}
""")
        with self.doc:
            self.row = div(cls="row")

    def add_column_space(self, width=20):
        with self.row:
            div(cls="column", style="width:%dpx" % width)

    def add_column(self, head, ims, txts, links, width=None):
        with self.row:
            if width is not None:
                column_style = "width:%spx" % width
            else:
                column_style = ""
            with div(cls="column", style=column_style):
                h3(head, style="height:60px")
                for im, txt, link in zip(ims, txts, links):
                    with a(href=os.path.join('images', link)):
                        img(style="width:%dpx" % self.width, src=os.path.join('images', im))
                    p(txt, style="height:16px")


    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=0):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=128):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self, suffix=''):
        html_file = '%s/index%s.html' % (self.web_dir, suffix)
        f = open(html_file, 'wt')  # 't': text mode (default)
        f.write(self.doc.render())
        f.close()

    def save_at_specified_dir(self, ckpt_dir):
        html_file = '%s/index.html' % (ckpt_dir)
        f = open(html_file, 'wt')  # 't': text mode (default)
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
