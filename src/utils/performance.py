from pytictoc import TicToc

class TicToc(TicToc):
    """Extend the original TicToc class"""
    def toc(self, msg='Elapsed time:', restart=False):
        from timeit import default_timer
        from datetime import timedelta
        self.end     = default_timer()
        self.elapsed = self.end - self.start
        print('%s %s' % (msg, timedelta(seconds=round(self.elapsed))))
        if restart:
            self.start = default_timer()