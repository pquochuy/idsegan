CUDA_VISIBLE_DEVICES="0,1,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_recur4 --init_l1_weight 100. --batch_size 50 --g_nl prelu --save_freq 50 --preemph 0.95 --epoch 100 --bias_deconv True --bias_downconv True --bias_D_conv True --e2e_dataset '../data/segan.tfrecords' --recursitivity 4 --synthesis_path dwavegan_samples_recur4
mkdir cleaned_testset_wav_16k_recur4_97000
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_recur4 --batch_size 50 --g_nl prelu --weights SEGAN-97000 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_recur4_97000/' --recursitivity 4
mkdir cleaned_testset_wav_16k_recur4_97100
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_recur4 --batch_size 50 --g_nl prelu --weights SEGAN-97100 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_recur4_97100/' --recursitivity 4
mkdir cleaned_testset_wav_16k_recur4_97200
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_recur4 --batch_size 50 --g_nl prelu --weights SEGAN-97200 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_recur4_97200/' --recursitivity 4
mkdir cleaned_testset_wav_16k_recur4_97300
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_recur4 --batch_size 50 --g_nl prelu --weights SEGAN-97300 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_recur4_97300/' --recursitivity 4
mkdir cleaned_testset_wav_16k_recur4_97400
CUDA_VISIBLE_DEVICES="0,-1" python main.py --init_noise_std 0. --save_path segan_allbiased_preemph_recur4 --batch_size 50 --g_nl prelu --weights SEGAN-97400 --preemph 0.95 --bias_deconv True --bias_downconv True --bias_D_conv True --test_wav_dir '../data/noisy_testset_wav_16k/' --save_clean_path './cleaned_testset_wav_16k_recur4_97400/' --recursitivity 4


