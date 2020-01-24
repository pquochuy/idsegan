#!/bin/bash

# train segan
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph --init_l1_weight 100. --batch_size 100 --g_nl prelu --save_freq 50 --preemph 0.95 --epoch 100 --bias_deconv True --bias_downconv True --bias_D_conv True --e2e_dataset '../data/segan.tfrecords' --synthesis_path dwavegan_samples

# clean noisy signals with checkpoint 48500
mkdir cleaned_testset_wav_16k_48500
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph --batch_size 100 --g_nl prelu --weights SEGAN-48500 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_48500/'

# clean noisy signals with checkpoint 48550
mkdir cleaned_testset_wav_16k_48550
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph --batch_size 100 --g_nl prelu --weights SEGAN-48550 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_48550/'

# clean noisy signals with checkpoint 48600
mkdir cleaned_testset_wav_16k_48600
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph --batch_size 100 --g_nl prelu --weights SEGAN-48600 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_48600/'

# clean noisy signals with checkpoint 48650
mkdir cleaned_testset_wav_16k_48650
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph --batch_size 100 --g_nl prelu --weights SEGAN-48650 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_48650/'

# clean noisy signals with checkpoint 48700
mkdir cleaned_testset_wav_16k_48700
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph --batch_size 100 --g_nl prelu --weights SEGAN-48700 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_48700/'

