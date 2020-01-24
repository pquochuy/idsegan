clear all
close all
clc

Ncp = 5;

list_file = dir('../data/clean_testset_wav_16k/*.wav');

% five last ISEGAN checkpoints
cp = {'97000','97100','97200','97300','97400'};

% the number of iterations N
iteration = [2,3,4];
ret_iter = cell(numel(iteration), 1);

for iter = 1 : numel(iteration)
    ret = zeros(Ncp, 5);
    for c = 1 : Ncp
        ret_c = zeros(numel(list_file),5);
        parfor f = 1 : numel(list_file)
            disp(list_file(f).name);
            clean_wav = ['../data/clean_testset_wav_16k/', list_file(f).name];
            noisy_wav = ['../isegan/cleaned_testset_wav_16k_iter', ... 
                num2str(iteration(iter)),'_',cp{c},'/', list_file(f).name];
            spesq = pesq(clean_wav, noisy_wav);
            [~,ssnr] = comp_snr(clean_wav, noisy_wav);
            [Csig,Cbak,Covl] = composite(clean_wav,noisy_wav);
            ret_c(f,:) = [spesq, Csig,Cbak,Covl, ssnr];
        end
        ret(c, :) = mean(ret_c);
    end
    ret_iter{iter} = ret;
end

for iter = 1 : numel(iteration)
    disp(['ISEGAN',num2str(iteration(iter)),' - Mean:'])
    disp(mean(ret_iter{iter}));
    disp(['ISEGAN',num2str(iteration(iter)),' - Std:'])
    disp(std(ret_iter{iter}));
end

