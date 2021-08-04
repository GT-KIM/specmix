clear all; clc;
cleanpath = 'test_Specmix/clean/';
enhancedpath = 'test_Specmix/estimated/';
filelist = dir(cleanpath);

metrics = [0. 0. 0. 0. 0.];

n = length(filelist);
for k = 3:n
    cleanFile = strcat(cleanpath,'\',filelist(k).name);
    enhancedFile = strcat(enhancedpath,'\',filelist(k).name);
    [pesq, CSIG, CBAK, COVL, segSNR] = composite(cleanFile, enhancedFile);
    fprintf("idx : %d pesq : %f CSIG : %f CBAK : %f COVL : %f SSNR : %f\n",k-2,pesq, CSIG, CBAK, COVL, segSNR);
    metrics(1) = metrics(1) + pesq;
    metrics(2) = metrics(2) + CSIG;
    metrics(3) = metrics(3) + CBAK;
    metrics(4) = metrics(4) + COVL;
    metrics(5) = metrics(5) + segSNR;
end
metrics = metrics / (n-2);
fprintf("TEST RESULTS\n");
fprintf("pesq : %f CSIG : %f CBAK : %f COVL : %f SSNR : %f",metrics(1),metrics(2),metrics(3),metrics(4),metrics(5));

%[pesq_mos, Csig, Cbak, Covl, segSNR] = composite(cleanFile, enhancedFile);