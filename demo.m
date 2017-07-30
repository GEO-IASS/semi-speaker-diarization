% demo to demonstrate semi-supervised speaker diarization 
clear all;clc;

% Read speech signal
[speech, Fs, nbits] = wavread('CONCAT_2_4_9731.wav');

% Read ground-truth label file
new_label_data = load('CONCAT_2_4_9731_LABEL.txt');

% Input segment of speech file whose speaker you want to spot throughout
% the file (enter start-end points in frames 1 second = 100 frames)
chosen_start = 950;
chosen_end = 1550;
mix = 16;
[s_w,MFCCs,ubm,agmm,adapt_frames,iter_flag] = semi_supervised_find(speech,Fs,chosen_start,chosen_end,mix);
iter_cnt = 1;
[s,speaker_label] = error_calc_new(new_label_data,s_w,chosen_start,chosen_end,length(MFCCs));

% displaying label of speaker you're spotting
speaker_label

% If no Re-Adaptation frames available stop
if(iter_flag)
   return;
end

% else update model and repeat
% model update 
[new_agmm] = model_update(ubm,MFCCs,adapt_frames);

% Model Update and Iteration
for i=1:3
    [s_w,re_adapt_frames,iter_flag] = semi_supervised_re_iter(MFCCs,ubm,new_agmm,adapt_frames);
    iter_cnt = iter_cnt + 1;     
    [s,speaker_label] = error_calc_new(new_label_data,s_w,chosen_start,chosen_end,length(MFCCs));
    
    % printing error estimate parameters at the end of every iteration
    i
    s.true_positive
    s.false_positive
    s.false_negative
    
    if(iter_flag)
      break;
    end
        
    if(i ~= 3)
      [new_agmm] = model_update(ubm,MFCCs,re_adapt_frames);
      adapt_frames = re_adapt_frames;
    end
end


