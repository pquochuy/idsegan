clear all
close all
clc

Ncp = 5;

list_file = dir('../data/clean_testset_wav_16k/*.wav');

ret = zeros(Ncp, 5);

% five last SEGAN checkpoints
cp = {'48500', '48550', '48600', '48650', '48700'};

for c = 1 : Ncp
    ret_c = zeros(numel(list_file),5);
    parfor f = 1 : numel(list_file)
        disp(list_file(f).name);
        clean_wav = ['../data/clean_testset_wav_16k/', list_file(f).name];
        noisy_wav = ['../segan/cleaned_testset_wav_16k_', cp{c}, '/', list_file(f).name];
        spesq = pesq(clean_wav, noisy_wav);
        [~,ssnr] = comp_snr(clean_wav, noisy_wav);
        [Csig,Cbak,Covl] = composite(clean_wav,noisy_wav);
        ret_c(f,:) = [spesq, Csig, Cbak, Covl, ssnr];
    end
    ret(c, :) = mean(ret_c);
end

disp("SEGAN - Mean:")
disp(mean(ret));
disp("SEGAN - Std:")
disp(std(ret));