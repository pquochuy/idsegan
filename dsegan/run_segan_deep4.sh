#!/bin/bash

# train isegan with the depth N=4
CUDA_VISIBLE_DEVICES="0,1,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_deep4 --init_l1_weight 100. --batch_size 50 --g_nl prelu --save_freq 50 --preemph 0.95 --epoch 100 --bias_deconv True --bias_downconv True --bias_D_conv True --e2e_dataset '../data/segan.tfrecords' --depth 4 --synthesis_path dwavegan_samples_deep4

# clean noisy signals with checkpoint 97000
mkdir cleaned_testset_wav_16k_deep4_97000
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_deep4 --batch_size 50 --g_nl prelu --weights SEGAN-97000 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_deep4_97000/' --depth 4

# clean noisy signals with checkpoint 97100
mkdir cleaned_testset_wav_16k_deep4_97100
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_deep4 --batch_size 50 --g_nl prelu --weights SEGAN-97100 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_deep4_97100/' --depth 4

# clean noisy signals with checkpoint 97200
mkdir cleaned_testset_wav_16k_deep4_97200
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_deep4 --batch_size 50 --g_nl prelu --weights SEGAN-97200 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_deep4_97200/' --depth 4

# clean noisy signals with checkpoint 97300
mkdir cleaned_testset_wav_16k_deep4_97300
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_deep4 --batch_size 50 --g_nl prelu --weights SEGAN-97300 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_deep4_97300/' --depth 4

# clean noisy signals with checkpoint 97400
mkdir cleaned_testset_wav_16k_deep4_97400
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_deep4 --batch_size 50 --g_nl prelu --weights SEGAN-97400 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_deep4_97400/' --depth 4


