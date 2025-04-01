import datetime


class Logger(object):
    def __init__(self, out_fname):
        self.log_name = out_fname

    def log(self, out_str, end='\n', log_time=False):
        """
        out_str: single object now
        """
        # for classes, if __str__() is not defined, then it's the same as __repr__()
        if not isinstance(out_str, str):
            out_str = out_str.__repr__()
        if log_time:
            out_str = '{0:%Y-%m-%d %H:%M:%S} '.format(datetime.datetime.now()) + out_str
        with open(self.log_name, "a") as log_file:
            log_file.write(out_str + end)
        print(out_str, end=end, flush=True)


