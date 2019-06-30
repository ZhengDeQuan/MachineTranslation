"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from copy import deepcopy
import itertools
import torch
import traceback

import onmt.utils
from onmt.utils.logging import logger


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        if self.n_gpu > 1:
            train_iter = itertools.islice(
                train_iter, self.gpu_rank, None, self.n_gpu) #islice(iterable, start, stop[, step]) --> islice object

        best_accuracy = None
        best_step = None
        print("in train ")
        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            print("batches = ",batches)
            print("len(batches) = ",len(batches))
            exit(78)
            step = self.optim.training_step
            print("current step = ",step, "gpu_rank = ",self.gpu_rank)
            print("valis step = ",valid_steps)
            print("in for i,(batches, normalization) in enumerate() i = ",i)
            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:

                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                print("valid_stats.accuracy=", valid_stats.accuracy(),", valid_stats.ppl=",valid_stats.ppl(),", valid_stats.n_correct=",valid_stats.n_correct, " , valid_stats.n_words=",valid_stats.n_words," ,valid_stats.n_words=",valid_stats.loss)
                logger.info("valid_stats.accuracy=%f, valid_stats.ppl=%f, valid_stats.n_correct=%f, valid_stats.n_words=%f, valid_stats.loss=%f" % (valid_stats.accuracy(), valid_stats.ppl(),valid_stats.n_correct,valid_stats.n_words,valid_stats.loss))
                print("valid_stats.bleu = ",valid_stats.bleu)
                logger.info("valid_stats.bleu=%f"%valid_stats.bleu)
                current_accuracy = valid_stats.accuracy()
                if best_accuracy is None or current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_step = step

                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break
            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                suffix = str(current_accuracy) if current_accuracy >= best_accuracy else ""
                self.model_saver.save(step, moving_average=self.moving_average,suffix = suffix)
            if train_steps > 0 and step >= train_steps:
                break
            print("finish one iter")

        print("finish training processdure , \n the best accuracy is %f , happens_in %d step"%(best_accuracy ,step))
        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def zq_validate(self, valid_iter, moving_avergae=None):
        '''
        仿照validate()的骨架和translation的肉组成的，
        要在这里面实现translate()[decoder的时候完全没有tgt的加入]并且要有计算bleu的功能。
        最令我担心的是，多进程并行的事情。
        '''
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data.half() if self.model_dtype == "fp16" \
                    else avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)

                tgt = batch.tgt #如果这里有真是的tgt的句子，那么在valid_model()中预测第n个tgt的单词的时候，就有前n-1个单词的参与。
                                #如果这里没有真正tgt的句子，那么在valid_loss()中怎么根据标准答案算loss和准确率。

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths) #直接调用model中的forward，这样预测tgt的第n个单词的时候，会将前n-1个gtruth的tgt的单词考虑进去。
                #而在_translate_batch中，却是先调用self._run_encoder()-->self.model.encoder ,
                #在调用self._decode_and_generate() -->self.model.decoder(decoder_in,...,)其中的decoder_in是利用BeamSearch类的一个实例完成的beam.current_predictions
                #每次_decode_and_generate执行完成后，要配合使用beam.advance(log_probs, attn)



                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)
                #我需要再valid_loss中加入算bleu-4的函数，

                # Update statistics.
                stats.update(batch_stats)
                stats.zq_update_bleu(batch_stats,src.size(1)) #self.bleu += stat.bleu

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data.half() if self.model_dtype == "fp16" \
                    else avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)

                tgt = batch.tgt #如果这里有真是的tgt的句子，那么在valid_model()中预测第n个tgt的单词的时候，就有前n-1个单词的参与。
                                #如果这里没有真正tgt的句子，那么在valid_loss()中怎么根据标准答案算loss和准确率。

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths) #直接调用model中的forward，这样预测tgt的第n个单词的时候，会将前n-1个gtruth的tgt的单词考虑进去。
                #而在_translate_batch中，却是先调用self._run_encoder()-->self.model.encoder ,
                #在调用self._decode_and_generate() -->self.model.decoder(decoder_in,...,)其中的decoder_in是利用BeamSearch类的一个实例完成的beam.current_predictions
                #每次_decode_and_generate执行完成后，要配合使用beam.advance(log_probs, attn)



                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)
                #我需要再valid_loss中加入算bleu-4的函数，

                # Update statistics.
                stats.update(batch_stats)
                stats.zq_update_bleu(batch_stats,src.size(1)) #self.bleu += stat.bleu

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):

        '''
        在这里真正的看见model的输出，然后model的输出的答案和标准答案之间计算loss
        外部的调用是：
        self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)
        额外的信息
        len(batches) == 3
        '''
        if self.accum_count > 1:
            self.optim.zero_grad()
        print("in _gradient_accumulation")
        logger.info("in _gradient_accumulation")
        for k, batch in enumerate(true_batches):
            # print("batch = ",batch)
            '''
            当我设置batch_size = 3的时候
            batch=  
            [torchtext.data.batch.Batch of size 3]
            [.src]:('[torch.LongTensor of size 38x3x1]', '[torch.LongTensor of size 3]')
            [.tgt]:[torch.LongTensor of size 37x3x1]
            [.indices]:[torch.LongTensor of size 3]
            '''
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size
                #走地下这个分支的化，相当于是没有truncat，
                #truncat bptt，是为了不让过久之前的梯度影响现在。

            #[.src]:('[torch.LongTensor of size 38x3x1]', '[torch.LongTensor of size 3]')
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt#[.tgt]:[torch.LongTensor of size 37x3x1] tgt_len * batch_size * 1

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # print("tgt = ",tgt.size()) #[.tgt]:[torch.LongTensor of size 37x3x1]
                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()
                outputs, attns = self.model(src, tgt, src_lengths, bptt=bptt) #[.src]:('[torch.LongTensor of size 38x3x1]', '[torch.LongTensor of size 3]')
                # print("outputs = ",outputs.size())#[16, 3, 100] [src_len-1, batch_size, rnn_size]
                #attns.keys() == dict_keys(['std'])
                #attns本身是一个字典，字典中只有一个键‘std’，其对应的值是一个[16, 3, 20]的tensor 20是什么呢？需要进入模型内部仔细观察
                # print("atns = ",attns)
                bptt = True

                # print("before compute loss k = ", k, " j= ", j)
                # logger.info("before compute loss k = %d, j= %d",k, j)
                # 3. Compute loss.
                # import pdb
                # pdb.set_trace()
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)

                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)
                # print("k = ",k," j= ",j)
                # logger.info("k = %d , j=%d",k,j)
                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]

                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # added by zhengquan to calculate the amonut of transmission
                # grads = [p.grad.data for p in self.model.parameters()
                #          if p.requires_grad
                #          and p.grad is not None]
                # total_sz = 0
                # for t in grads:
                #     sz = t.numel() * t.element_size()
                #     total_sz += sz
                # print("total_sz = ",total_sz)
                # import pdb
                # pdb.set_trace()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

            # print("end of for")
            # exit(78)
        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads,float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
