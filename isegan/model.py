from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from scipy.io import wavfile
from generator import *
from discriminator import *
import numpy as np
from data_loader import read_and_decode, de_emph
from bnorm import VBN
from ops import *
import timeit
import os
import shutil

class Model(object):
    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess, os.path.join(save_path, model_name), global_step=step)

    def load(self, save_path, model_file=None):
        if not os.path.exists(save_path):
            print('[!] Checkpoints path does not exist...')
            return False
        print('[*] Reading checkpoints...')
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True

class SEGAN(Model):
    """ Speech Enhancement Generative Adversarial Network """

    def __init__(self, sess, args, devices, infer=False, name='SEGAN'):
        super(SEGAN, self).__init__(name)
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        if infer:
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        else:
            self.keep_prob = 0.5
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.d_label_smooth = args.d_label_smooth
        self.devices = devices
        self.z_dim = args.z_dim
        self.z_depth = args.z_depth
        # type of deconv
        self.deconv_type = args.deconv_type
        # specify if use biases or not
        self.bias_downconv = args.bias_downconv
        self.bias_deconv = args.bias_deconv
        self.bias_D_conv = args.bias_D_conv
        # clip D values
        self.d_clip_weights = False
        # apply VBN or regular BN?
        self.disable_vbn = False
        self.save_path = args.save_path
        # num of updates to be applied to D before G
        # this is k in original GAN paper (https://arxiv.org/abs/1406.2661)
        self.disc_updates = 1
        # set preemph factor
        self.preemph = args.preemph
        if self.preemph > 0:
            print('*** Applying pre-emphasis of {} ***'.format(self.preemph))
        else:
            print('--- No pre-emphasis applied ---')
        # canvas size
        self.canvas_size = args.canvas_size
        self.deactivated_noise = False
        # dilation factors per layer (only in atrous conv G config)
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # num fmaps for AutoEncoder SEGAN (v1)
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        # Define D fmaps
        self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.init_noise_std = args.init_noise_std
        self.disc_noise_std = tf.Variable(self.init_noise_std, trainable=False)
        '''
        # skip summary
        self.disc_noise_std_summ = scalar_summary('disc_noise_std', self.disc_noise_std)
        '''
        self.e2e_dataset = args.e2e_dataset
        # G's supervised loss weight
        self.l1_weight = args.init_l1_weight
        self.l1_lambda = tf.Variable(self.l1_weight, trainable=False)
        self.deactivated_l1 = False
        # define the functions
        self.discriminator = discriminator
        # register G non linearity
        self.g_nl = args.g_nl
        if args.g_type == 'ae':
            self.generator = AEGenerator(self)
        elif args.g_type == 'dwave':
            self.generator = Generator(self)
        else:
            raise ValueError('Unrecognized G type {}'.format(args.g_type))

        # number of iterations
        self.iteration = args.iteration
        self.weights = np.ones(self.iteration)
        for nr in range(self.iteration-2, -1, -1):
            # the weight of generator nr is half of that of generator nr+1
            self.weights[nr] = self.weights[nr+1] / 2
        print('Weights:')
        print(self.weights)

        self.build_model(args)




    def build_model(self, config):
        all_d_grads = []
        all_g_grads = []
        d_opt = tf.train.RMSPropOptimizer(config.d_learning_rate)
        g_opt = tf.train.RMSPropOptimizer(config.g_learning_rate)
        # d_opt = tf.train.AdamOptimizer(
        #     config.d_learning_rate, beta1=config.beta_1)
        # g_opt = tf.train.AdamOptimizer(
        #     config.g_learning_rate, beta1=config.beta_1)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for idx, device in enumerate(self.devices):
                with tf.device("/%s" % device):
                    with tf.name_scope("device_%s" % idx):
                        with variables_on_gpu0():
                            self.build_model_single_gpu(idx)
                            d_grads = d_opt.compute_gradients(self.d_losses[-1], var_list=list(self.d_vars))
                            g_grads = g_opt.compute_gradients(self.g_losses[-1], var_list=list(self.g_vars))
                            all_d_grads.append(d_grads)
                            all_g_grads.append(g_grads)
                            # tf.get_variable_scope().reuse_variables()
        avg_d_grads = average_gradients(all_d_grads)
        avg_g_grads = average_gradients(all_g_grads)
        self.d_opt = d_opt.apply_gradients(avg_d_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)

    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # create the nodes to load for input pipeline
            filename_queue = tf.train.string_input_producer([self.e2e_dataset])
            self.get_wav, self.get_noisy = read_and_decode(
                filename_queue, self.canvas_size, self.preemph)
        # load the data to input pipeline
        wavbatch, \
        noisybatch = tf.train.shuffle_batch([self.get_wav,
                                             self.get_noisy],
                                             batch_size=self.batch_size,
                                             num_threads=2,
                                             capacity=1000 + 3 * self.batch_size,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy')
        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []
            for nr in range(self.iteration):
                self.Gs.append([])
                self.zs.append([])
        self.gtruth_wavs.append(wavbatch)
        self.gtruth_noisy.append(noisybatch)

        # add channels dimension to manipulate in D and G
        wavbatch = tf.expand_dims(wavbatch, -1)
        noisybatch = tf.expand_dims(noisybatch, -1)
        # by default leaky relu is used
        do_prelu = False
        if self.g_nl == 'prelu':
            do_prelu = True
        if gpu_idx == 0:
            ref_Gs = self.generator(noisybatch, is_ref=True, spk=None, do_prelu=do_prelu)
            print('num of G returned: ', len(ref_Gs))
            self.reference_G = ref_Gs[0] # returned wave by the generator
            self.ref_z = ref_Gs[1]       # returned z by the generator
            if do_prelu:
                self.ref_alpha = ref_Gs[2:]
                '''
                # skip summary
                self.alpha_summ = []
                for m, ref_alpha in enumerate(self.ref_alpha):
                    # add a summary per alpha
                    self.alpha_summ.append(histogram_summary('alpha_{}'.format(m), ref_alpha))
                '''
            # make a dummy copy of discriminator to have variables and then
            # be able to set up the variable reuse for all other devices
            # merge along channels and this would be a real batch
            dummy_joint = tf.concat([wavbatch, noisybatch], 2)
            dummy = discriminator(self, dummy_joint, reuse=False)

        input_ = noisybatch
        Gs = []
        zs = []
        for nr in range(self.iteration):
            G, z = self.generator(input_, is_ref=False, spk=None, do_prelu=do_prelu)
            Gs.append(G)
            zs.append(z)
            self.Gs[nr].append(G)
            self.zs[nr].append(z)
            input_ = G


        # add new dimension to merge with other pairs
        D_rl_joint = tf.concat([wavbatch, noisybatch], 2) # real
        D_fk_joint = []
        for nr in range(self.iteration):
            D_fk_joint.append(tf.concat([Gs[nr], noisybatch], 2)) # fake1
        # build rl discriminator
        d_rl_logits = discriminator(self, D_rl_joint, reuse=True)
        # build fk G discriminator
        d_fk_logits = []
        for nr in range(self.iteration):
            d_fk_logits.append(discriminator(self, D_fk_joint[nr], reuse=True))


        if gpu_idx == 0:
            self.g_losses = []
            self.g_l1_losses = []
            for nr in range(self.iteration):
                self.g_l1_losses.append([])
            self.g_adv_losses = []
            self.d_rl_losses = []
            self.d_fk_losses = []
            for nr in range(self.iteration):
                self.d_fk_losses.append([])
            self.d_losses = []

        d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
        d_fk_loss = []
        for nr in range(self.iteration):
            d_fk_loss.append(tf.reduce_mean(tf.squared_difference(d_fk_logits[nr], 0.)))
        g_adv_loss = 0.
        for nr in range(self.iteration):
            g_adv_loss += tf.reduce_mean(tf.squared_difference(d_fk_logits[nr], 1.))
        ## corrected division of self.iteration here
        g_adv_loss /= self.iteration

        d_loss = d_rl_loss
        for nr in range(self.iteration):
            ## corrected division of self.iteration here
            d_loss += d_fk_loss[nr]/self.iteration

        # Add the L1 loss to G
        g_l1_loss = []
        for nr in range(self.iteration):
            g_l1_loss.append(self.l1_lambda * self.weights[nr] *
                             tf.reduce_mean(tf.abs(tf.subtract(Gs[nr], wavbatch))))

        #g_loss
        g_loss = g_adv_loss
        for nr in range(self.iteration):
            g_loss += g_l1_loss[nr]

        for nr in range(self.iteration):
            self.g_l1_losses[nr].append(g_l1_loss[nr])
        self.g_adv_losses.append(g_adv_loss)
        self.g_losses.append(g_loss)
        self.d_rl_losses.append(d_rl_loss)
        for nr in range(self.iteration):
            self.d_fk_losses[nr].append(d_fk_loss[nr])
        self.d_losses.append(d_loss)

        if gpu_idx == 0:
            self.get_vars()


    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars_dict = {}
        self.g_vars_dict = {}
        for var in t_vars:
            if var.name.startswith('d_'):
                self.d_vars_dict[var.name] = var
            if var.name.startswith('g_'):
                self.g_vars_dict[var.name] = var
        self.d_vars = self.d_vars_dict.values()
        self.g_vars = self.g_vars_dict.values()
        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in t_vars:
            assert x in self.g_vars or x in self.d_vars, x.name
        self.all_vars = t_vars
        if self.d_clip_weights:
            print('Clipping D weights')
            self.d_clip = [
                v.assign(tf.clip_by_value(v, -0.05, 0.05)) for v in self.d_vars
            ]
        else:
            print('Not clipping D weights')

    def vbn(self, tensor, name):
        if self.disable_vbn:

            class Dummy(object):
                # Do nothing here, no bnorm
                def __init__(self, tensor, ignored):
                    self.reference_output = tensor

                def __call__(self, x):
                    return x

            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def train(self, config, devices):
        """ Train the SEGAN """

        print('Initializing optimizers...')
        # init optimizers
        d_opt = self.d_opt
        g_opt = self.g_opt
        num_devices = len(devices)

        try:
            init = tf.global_variables_initializer()
        except AttributeError:
            # fall back to old implementation
            init = tf.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        # Hang onto a copy of wavs so we can feed the same one every time
        # we store samples to disk for hearing
        # pick a single batch

        # intentionally skipped saving intermediate audio files

        save_path = config.save_path
        synthesis_path = config.synthesis_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset, num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        d_fk_losses = []
        for nr in range(self.iteration):
            d_fk_losses.append([])
        d_rl_losses = []
        g_adv_losses = []
        g_l1_losses = []
        for nr in range(self.iteration):
            g_l1_losses.append([])
        try:
            while not coord.should_stop():
                start = timeit.default_timer()

                for d_iter in range(self.disc_updates):
                    _d_opt, \
                    d_fk_loss_, \
                    d_rl_loss = self.sess.run([d_opt,
                                               self.d_fk_losses,
                                               self.d_rl_losses[0]])
                    d_fk_loss = [d_fk_loss_i[0] for d_fk_loss_i in d_fk_loss_] # extract only the first element
                    if self.d_clip_weights:
                        self.sess.run(self.d_clip)

                _g_opt, \
                g_adv_loss, \
                g_l1_loss_ = self.sess.run([g_opt, self.g_adv_losses[0],
                                            #self.g_l1_losses[:][0]])
                                            self.g_l1_losses])
                g_l1_loss = [g_l1_loss_i[0] for g_l1_loss_i in g_l1_loss_] # extract only the first element

                end = timeit.default_timer()
                batch_timings.append(end - start)
                for nr in range(self.iteration):
                    d_fk_losses[nr].append(d_fk_loss[nr])
                d_rl_losses.append(d_rl_loss)
                g_adv_losses.append(g_adv_loss)
                for nr in range(self.iteration):
                    g_l1_losses[nr].append(g_l1_loss[nr]/self.weights[nr])

                s = '{}/{} (epoch {}), d_rl_loss = {:.5f}, '.format(counter,
                                                                      config.epoch * num_batches,
                                                                      curr_epoch,
                                                                      d_rl_loss)
                for nr in range(self.iteration):
                    s += 'd_fk_loss' + str(nr) + ' = {:.5f} '.format(d_fk_loss[nr])
                s += 'g_adv_loss = {:.5f} '.format(g_adv_loss)
                for nr in range(self.iteration):
                    s += 'g_l1_loss' + str(nr) + ' = {:.5f} '.format(g_l1_loss[nr]/self.weights[nr])
                s += ' time/batch = {:.5f}, mtime/batch = {:.5f} '.format(end - start, np.mean(batch_timings))
                print(s)

                batch_idx += num_devices
                counter += num_devices
                if (counter / num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)

                    np.savetxt(os.path.join(save_path, 'd_rl_losses.txt'),d_rl_losses)
                    for nr in range(self.iteration):
                        np.savetxt(os.path.join(save_path, 'd_fk_losses' + str(nr) + '.txt'),d_fk_losses[nr])
                    np.savetxt(os.path.join(save_path, 'g_adv_losses.txt'),g_adv_losses)
                    for nr in range(self.iteration):
                        np.savetxt(os.path.join(save_path, 'g_l1_losses' + str(nr) + '.txt'),g_l1_losses[nr])

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    # re-set batch idx
                    batch_idx = 0
                    # check if we have to deactivate L1
                    if curr_epoch >= config.l1_remove_epoch and self.deactivated_l1 == False:
                        print('** Deactivating L1 factor! **')
                        self.sess.run(tf.assign(self.l1_lambda, 0.))
                        self.deactivated_l1 = True
                    # check if we have to start decaying noise (if any)
                    if curr_epoch >= config.denoise_epoch and self.deactivated_noise == False:
                        # apply noise std decay rate
                        decay = config.noise_decay
                        if not hasattr(self, 'curr_noise_std'):
                            self.curr_noise_std = self.init_noise_std
                        new_noise_std = decay * self.curr_noise_std
                        if new_noise_std < config.denoise_lbound:
                            print('New noise std {} < lbound {}, setting 0.'.
                                  format(new_noise_std, config.denoise_lbound))
                            print('** De-activating noise layer **')
                            # it it's lower than a lower bound, cancel out completely
                            new_noise_std = 0.
                            self.deactivated_noise = True
                        else:
                            print(
                                'Applying decay {} to noise std {}: {}'.format(
                                    decay, self.curr_noise_std, new_noise_std))
                        self.sess.run(
                            tf.assign(self.disc_noise_std, new_noise_std))
                        self.curr_noise_std = new_noise_std
                if curr_epoch >= config.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    '''
                    # skip summary
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    '''
                    break
        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)

    def clean(self, x):
        """ clean a utterance x
            x: numpy array containing the normalized noisy waveform
        """
        c_res = None
        for beg_i in range(0, x.shape[0], self.canvas_size):
            if x.shape[0] - beg_i < self.canvas_size:
                length = x.shape[0] - beg_i
                pad = (self.canvas_size) - length
            else:
                length = self.canvas_size
                pad = 0
            x_ = np.zeros((self.batch_size, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length],np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]
            print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
            fdict = {self.gtruth_noisy[0]: x_}
            canvas_w = self.sess.run(self.Gs[-1][0], feed_dict=fdict)[0]
            canvas_w = canvas_w.reshape((self.canvas_size))
            print('canvas w shape: ', canvas_w.shape)
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                # get rid of last padded samples
                canvas_w = canvas_w[:-pad]
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        # deemphasize
        c_res = de_emph(c_res, self.preemph)
        return c_res

