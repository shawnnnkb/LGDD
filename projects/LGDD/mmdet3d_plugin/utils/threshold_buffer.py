from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class EnableTrainThresholdHookIter(Hook):

    def __init__(self,
                 enable_after_iter=5000,
                 threshold_buffer=0,
                 buffer_iter=2000,
                 ):
        self.enable_after_iter = enable_after_iter
        self.buffer_iter = buffer_iter
        self.delta = threshold_buffer / buffer_iter
        self.threshold_buffer = threshold_buffer

    def before_train_iter(self, runner):
        cur_iter = runner.iter # begin from 0
        if cur_iter == self.enable_after_iter:
            runner.logger.info(f'Enable Detection from now.')
        if cur_iter >= self.enable_after_iter: # keep the sanity when resuming model
            runner.model.module.runtime_info['enable_detection'] = True
        if self.threshold_buffer > 0 and cur_iter > self.enable_after_iter and cur_iter < self.enable_after_iter + self.buffer_iter:
            runner.model.module.runtime_info['threshold_buffer'] = (self.enable_after_iter + self.buffer_iter - cur_iter) * self.delta
        else:
            runner.model.module.runtime_info['threshold_buffer'] = 0