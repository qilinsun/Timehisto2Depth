clear all;
close all;
clc;

%% SPAD Parameters

DCR = 3000;             % Dark Count Rate (counts/s)
PDP = 0.3;              % Photon Detection Probability (-)
APP = 0.01;             % After Pulsing Probability (-)
CTP = 0.001;            % Crosstalk Probability (-)
t_dead = 10e-9;         % Dead time (s)
noise_back = 5e11;      % Background noise (counts/s)

c = 3e8;                % Speed of light (m/s)
exp = 0.005;            % Film exposition (m)
res_xy = 300;           % Space Resolution (pixels)
res_t = 4096;           % Time Resolution (pixels)

load('jitter.mat');     % Jitter curves
mu_noise = mu_noise_1;
counts = counts_1;
t = t_1;

M = 1e4;                % Measurements

%% Processing and Simulation

dt = exp/c;
folder_in = 'hdr_render';     % Render streaks
folder_out = 'hdr_spad';      % SPAD streaks
mkdir(folder_out);

%distcomp.feature( 'LocalUseMpiexec', false);
%profile on; % profile viewer

tic
hdr_sum = zeros(res_xy,res_xy);
hdr_spad = zeros(res_xy,res_xy,res_t);
for i = 1:res_xy
    hdr = hdrread(sprintf('%s/img_%04.0f.hdr',folder_in,i-1));
    hdr = hdr(:,:,1);   % Get only one RGB channel for speedup (comment if needed)
    for j = 1:size(hdr,1)
        k_vec = randsample(1:size(hdr,2),M,true,hdr(j,:,1));    % Importance sampling
        jitter_vec = round(randsample(t_1,M,true,counts_1)/dt); idx = 1;
        for k = k_vec
            if (hdr(j,k) ~= 0)&&(rand < PDP) % Photon detected
                jitter = jitter_vec(idx); idx = idx + 1;                
                % Time jitter              
                if (k+jitter) < 0
                    hdr_spad(i,j,1) = hdr_spad(i,j,1) + 1;
                elseif (k+jitter) > size(hdr,2)
                    hdr_spad(i,j,end) = hdr_spad(i,j,end) + 1;
                else
                    hdr_spad(i,j,k+jitter) = hdr_spad(i,j,k+jitter) + 1;
                end               
                % Dead time + Afterpulse
                afterpulse = round(t_dead/dt); n = 1;
                while (k + afterpulse <= size(hdr,2))                    
                    if (rand < APP^n)
                        hdr_spad(i,j,k+afterpulse) = hdr_spad(i,j,k+afterpulse) + 1;
                    end
                    afterpulse = afterpulse + round(t_dead/dt);
                    n = n + 1;
                end
                % Crosstalk left
                if (rand < CTP)&&(j - 1 > 0)
                    hdr_spad(i,j-1,k) = hdr_spad(i,j-1,k) + 1;
                end
                % Crosstalk right
                if (rand < CTP)&&(j + 1 <= size(hdr,1))
                    hdr_spad(i,j+1,k) = hdr_spad(i,j+1,k) + 1;
                end
                % Crosstalk top
                if (rand < CTP)&&(i + 1 <= size(hdr,1))
                    hdr_spad(i+1,j,k) = hdr_spad(i+1,j,k) + 1;
                end
                % Crosstalk bottom
                if (rand < CTP)&&(i - 1 > 0)
                    hdr_spad(i-1,j,k) = hdr_spad(i-1,j,k) + 1;
                end
            end
        end
    
        disp(['Line ' num2str(i) ' - ' num2str(j/size(hdr,1)*100) ' %']);
    end   
    % Background noise
    mu_noise_back = mu_noise*M/sum(counts)*size(hdr,2)/length(t);
    RTN = poissrnd(mu_noise_back,size(hdr,1),size(hdr,2));
    hdr_spad(i,:,:) = hdr_spad(i,:,:) + reshape(RTN,1,size(hdr,1),size(hdr,2));
end

%% Output: Write streak images

hdr_spad = repmat(hdr_spad,1,1,1,3); % Copy value to three RGB channels (comment if needed)

for i = 1:res_xy
    hdrwrite(squeeze(hdr_spad(i,:,:,:)),sprintf('%s/img_SPAD_%04.0f.hdr',folder_out,i-1));
    disp(['Writing Line ' num2str(i) ' - ' num2str(i/res_xy*100) ' %']);
end
toc



