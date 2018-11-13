import tensorflow as tf


class Logger(object):
    def __init__(self, tb_dir, log_path):
        self.writer = tf.summary.FileWriter(tb_dir)
        self.file = open(log_path, 'w+')

    def __del__(self):
        self.file.close()

    def log_kv(self, key_value_list, prefix, step, write_to_tb=False, write_to_file=False):
        message = '{prefix}, step {step}: '.format(prefix=prefix, step=step)
        for key, value in key_value_list:
            if write_to_tb:
                self.scalar_summary('{}_{}'.format(prefix, key), value, step)
            message += '{} {:.3}\t'.format(key, value)
        print(message)
        if write_to_file:
            self.file.write(message + '\n')
            self.file.flush()

    def log_str(self, msg):
        print(msg)
        self.file.write(msg + '\n')
        self.file.flush()

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
