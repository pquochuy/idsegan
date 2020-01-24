clear all
close all
clc

Ncp = 5;

list_file = dir('../data/clean_testset_wav_16k/*.wav');

% five last DSEGAN checkpoints
cp = {'97000','97100','97200','97300','97400'};

% the number of depth N
depth = [2,3,4];
ret_deep = cell(numel(depth), 1);

for deep = 1 : numel(depth)
    ret = zeros(Ncp, 5);
    for c = 1 : Ncp
        ret_c = zeros(numel(list_file),5);
        parfor f = 1 : numel(list_file)
            disp(list_file(f).name);
            clean_wav = ['../data/clean_testset_wav_16k/', list_file(f).name];
            noisy_wav = ['../dsegan/cleaned_testset_wav_16k_deep', ...
                num2str(depth(deep)),'_',cp{c},'/', list_file(f).name];
            spesq = pesq(clean_wav, noisy_wav);
            [~,ssnr] = comp_snr(clean_wav, noisy_wav);
            [Csig,Cbak,Covl] = composite(clean_wav,noisy_wav);
            ret_c(f,:) = [spesq, Csig,Cbak,Covl, ssnr];
        end
        ret(c, :) = mean(ret_c);
    end
    ret_deep{deep} = ret;
end

for deep = 1 : numel(depth)
    disp(['DSEGAN',num2str(depth(deep)),' - Mean:'])
    disp(mean(ret_deep{deep}));
    disp(['DSEGAN',num2str(depth(deep)),' - Std:'])
    disp(std(ret_deep{deep}));
end