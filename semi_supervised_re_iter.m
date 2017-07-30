function [selected_frames_w,re_adapt_frames,iter_flag] = semi_supervised_re_iter(MFCCs,ubm,new_agmm,adapt_frames)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to perform sucessive steps of refinement in finding regions of 
% speaker activity using re-adapted GMM  
% Running window procedure applied to estimate P(x/speaker) and P(x/ubm) 
% for each hop of running window
%
%   INPUTS
%   MFCCs            - MFCCs previously computed on speech file
%   ubm              - UBM previously built on speech file
%   new_agmm         - New Adapted GM model to be used for speaker activity 
%                      detection  
%  adapt_frames      - current concatanated frames of MFCCs used to update the 
%                      current agmm model
%   OUTPUTS
%   selected_frames_w - segments from audio file which contain speaker in chosen 
%                       segment based on weights
%   selected_frames_P - segments from audio file which contain speaker in
%                       chosen segment based on P(x/speaker)
%   re_adapt_frames   - updated concatanated frames of MFCCs having high speaker activity which
%                       is used to update the new model
%   iter_flag         - flag indicating whether new data available for
%                       re-adaptation of the model ('1' if NO new data available and '0'
%                       otherwise)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializing parameters
px_speaker = 0;
px_ubm = 0;

% Running window parameters
start_point = 1;
end_point = 200;
hop = 40;
count =1;

%% Running window procedure
while(end_point < length(MFCCs))
    
    chosen = MFCCs(start_point:end_point);
    
    % Computing P(x/speaker) and P(x/ubm) for current window
    P_x_speaker = likelihood(chosen,new_agmm);
    P_x_ubm = likelihood(chosen,ubm);
    pi_speaker = 0.5;
    pi_ubm = 0.5;
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
    count = count + 1;
    
    % Update
    start_point = start_point + hop;
    end_point = end_point + hop;
end

% Median filtering to remove artifacts
segment_prob(:,3) = medfilt1(segment_prob(:,3));

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

%% selecting frames for re-adapting GM model (Re-adaptation threshold at 0.9)
iter_flag = 0;
re_adapt_ind = find(segment_prob(:,3) >= 0.9); 
if(~isempty(re_adapt_ind))
    re_adapt_frames = segment_prob(re_adapt_ind,1:2);
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
    
    % Removing overlapping data frames
    no_overlap = zeros(1,re_adapt_frames(end,2));
    [r,c] = size(re_adapt_frames);
    for i=1:r
        added = zeros(1,re_adapt_frames(end,2));
        added(re_adapt_frames(i,1):re_adapt_frames(i,2)) = 1;
        no_overlap = no_overlap | added;
    end
    re_adapt_frames = find(no_overlap);
    
    % Adaptation "worthy" data frames
    re_adapt_frames = horzcat(adapt_frames,re_adapt_frames);
    re_adapt_frames = unique(re_adapt_frames);
    
else
iter_flag = 1;
re_adapt_frames = [];
end

end




