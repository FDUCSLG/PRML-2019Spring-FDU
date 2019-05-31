from fastNLP import Callback

import time

start_time = time.time()


class TimingCallback(Callback):
    def on_epoch_end(self):
        print('Sum Time: {:d}s\n\n'.format(round(time.time() - start_time)))
