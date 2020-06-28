import os


class SimpleLogger:
    def __init__(self, logdir):
        self.logdir = logdir

    def add(self, name, step, value):
        with open(os.path.join(self.logdir, name + '.csv'), 'a') as f:
            print('%d,%f' % (step, value), file=f)

        print('step=%d %s=%f' % (step, name, value))
