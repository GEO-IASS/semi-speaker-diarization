function [selected_frames_w,MFCCs,ubm,agmm,re_adapt_frames,iter_flag] = semi_supervised_find(speech,Fs,chosen_start,chosen_end,mix)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to perform  clustering using segment of pure speaker provided 
% by user. Running window procedure applied to estimate P(x/speaker) and
% P(x/ubm) for each hop of running window
%
%   INPUTS
%   speech - vector containing speech data to be searched 
%   Fs - sampling rate of speech data
%   chosen_start - start point (in frames) of pure speaker segment input by
%                 user
%   chosen_end - end point (in frames) of pure speaker segment by user
%   1 second of speech data = 100 frames 
%   mix - number of mixtures in GMM 
% 
%   OUTPUTS
%   selected_frames_w - segments from audio file which contain speaker in chosen 
%                       segment based on weights
%   MFCCs             - MFCCs computed on speech file
%  
%   ubm               - UBM which has been built on the entire entire
%                       speech file
%   agmm              - Adapted GMM (speaker model) built on user entered data  
%   re_adapt_frames   - frames of MFCCs having high speaker activity which
%                       is used to update the model
%   iter_flag         - flag indicating whether new data available for
%                       re-adaptation of the model ('1' if NO new data available and '0'
%                       otherwise)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializing parameters
px_speaker = 0;
px_ubm = 0;

% % Initializing parameters for MFCC extraction
% Tw = 25;           % analysis frame duration (ms)
% Ts = 10;           % analysis frame shift (ms)
% alpha = 0.97;      % preemphasis coefficient
% R = [ 300 3700 ];  % frequency range to consider
% M = 20;            % number of filterbank channels 
% C = 13;            % number of cepstral coefficients/window length
% L = 22;            % cepstral sine lifter parameter
% 
% % creating hamming window
% window = hamming(C);
% 
% % Extracting MFCC's for whole file
% [MFCCs, FBEs, frames] = mfcc(speech, Fs, Tw, Ts, alpha, window, R, M, C, L);
% MFCCs=MFCCs(2:end,:);

% creating MFCCs based feature vector from speech signal
feature = feature_create_new(speech,Fs,12);
MFCCs = feature;

% Initializing UBM parameters
nmix = mix;           
final_niter = 10;
ds_factor = 1;
nWorkers = 1;
ubm = gmm_em(MFCCs(:), nmix, final_niter, ds_factor, nWorkers);

% Adapting the GMM to mfcc segment chosen by user
chosen = MFCCs(chosen_start:chosen_end);
% GMM Adaptation parameter
tau = 19;
agmm=mapAdapt(chosen, ubm, tau, 'm');

% Running window parameters
start_point = 1;
end_point = 200;
hop = 40;
count =1;

%% Running window procedure
while(end_point < length(MFCCs))
    
    chosen = MFCCs(start_point:end_point);
    
    % Computing P(x/speaker) and P(x/ubm) for current window
    P_x_speaker = likelihood(chosen,agmm);
    P_x_ubm = likelihood(chosen,ubm);
    pi_speaker = 0.5;
    pi_ubm = 0.5;
    %iter = 1;
    epsilon = 1;
    while(epsilon>= 0.005)
         temp = pi_speaker;
         gamma_speaker = (pi_speaker.*P_x_speaker) ./ ((pi_speaker.*P_x_speaker) + (pi_ubm.*P_x_ubm));
         gamma_ubm = (pi_ubm.*P_x_ubm) ./ ((pi_speaker.*P_x_speaker) + (pi_ubm.*P_x_ubm));
         N = sum(gamma_speaker) + sum(gamma_ubm);
         pi_speaker = sum(gamma_speaker)/N;
         pi_ubm = sum(gamma_ubm)/N;
         epsilon = abs(pi_speaker - temp);
    end 
    px_speaker(count) = pi_speaker;
    px_ubm(count) = pi_ubm;
    segment_prob(count,1) = start_point;
    segment_prob(count,2) = end_point;
    segment_prob(count,3) = pi_speaker;
    %segment_prob(count,4) = sum(log10(P_x_speaker)) - sum(log10(P_x_ubm));
    count = count + 1;
    
    % Update
    start_point = start_point + hop;
    end_point = end_point + hop;
end

% Median filtering to remove artifacts
segment_prob(:,3) = medfilt1(segment_prob(:,3));
%segment_prob(:,4) = medfilt1(segment_prob(:,4));

%% Based on weights
% Segmenting frames of speaker 
thresh = 0.5;
selected_ind_w = find(segment_prob(:,3)>=thresh);
selected_frames_w = segment_prob(selected_ind_w,1:2);

% collecting start points of segments 
shift = [0 selected_frames_w(1:end-1,1)'];
diff = abs(selected_frames_w(:,1)' - shift);
seg_ind = find(diff ~= hop);
col1 = selected_frames_w(seg_ind,1);

% collecting end points of segments
shift = [0 selected_frames_w(1:end-1,2)'];
diff = abs(selected_frames_w(:,2)' - shift);
seg_ind = find(diff ~= hop);
col2 = selected_frames_w(seg_ind(2:end) - 1,2);
col2 = vertcat(col2,selected_frames_w(end,2));

segment_array = horzcat(col1,col2);
selected_frames_w = segment_array;

%% Based on P(x/speaker)
% % Segmenting frames of speaker 
% thresh = 0;
% selected_ind_P = find(segment_prob(:,4)>thresh);
% selected_frames_P = segment_prob(selected_ind_P,1:2);
% 
% % collecting start points of segments 
% shift = [0 selected_frames_P(1:end-1,1)'];
% diff = abs(selected_frames_P(:,1)' - shift);
% seg_ind = find(diff ~= hop);
% col1 = selected_frames_P(seg_ind,1);
% 
% % collecting end points of segments
% shift = [0 selected_frames_P(1:end-1,2)'];
% diff = abs(selected_frames_P(:,2)' - shift);
% seg_ind = find(diff ~= hop);
% col2 = selected_frames_P(seg_ind(2:end) - 1,2);
% col2 = vertcat(col2,selected_frames_P(end,2));
% 
% segment_array = horzcat(col1,col2);
% selected_frames_P = segment_array;

%% selecting frames for re-adapting GM model (Re-adaptation threshold at 0.9)
iter_flag = 0;
re_adapt_ind = find(segment_prob(:,3) >= 0.9); 
    re_adapt_frames = segment_prob(re_adapt_ind,1:2);
if(~isempty(re_adapt_ind))
    % collecting start points of segments 
    shift = [0 re_adapt_frames(1:end-1,1)'];
    diff = abs(re_adapt_frames(:,1)' - shift);
    seg_ind = find(diff ~= hop);
    col1 = re_adapt_frames(seg_ind,1);

    % collecting end points of segments
    shift = [0 re_adapt_frames(1:end-1,2)'];
    diff = abs(re_adapt_frames(:,2)' - shift);
    seg_ind = find(diff ~= hop);
    col2 = re_adapt_frames(seg_ind(2:end) - 1,2);
    col2 = vertcat(col2,re_adapt_frames(end,2));

    segment_array = horzcat(col1,col2);
    re_adapt_frames = segment_array;

    % Adaptation "worthy" data frames
    re_adapt_frames = vertcat([chosen_start chosen_end],re_adapt_frames);
    
    % Removing overlapping data frames
    no_overlap = zeros(1,re_adapt_frames(end,2));
    [r,c] = size(re_adapt_frames);
    for i=1:r
        added = zeros(1,re_adapt_frames(end,2));
        added(re_adapt_frames(i,1):re_adapt_frames(i,2)) = 1;
        no_overlap = no_overlap | added;
    end
    re_adapt_frames = find(no_overlap);
else
iter_flag = 1;
end

end
