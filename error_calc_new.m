function [s,speaker_label] = error_calc_new(ground_truth,segment_boundary,start_point,end_point,mfccs_len)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function to compute error parameters such as false positive, false
%negative, true positive given some segmented results and some ground truth
% INPUTS
% ground_truth - the array containing the start point, end
%                points of segments,ground truth labels (nX3 matrix)
% segment_boundary - the boundaries where the speaker in ref_segment occurs(nX2 matrix)  
% (start_point - end_point) - a segment of audio file (in frames) whose 
%                             speaker is of interest to us(must be a pure 
%                             speaker segment)
% mfccs_len - length of mfccs feature vector computed for given audio file 

% OUTPUTS
% s - struct variable containing 3 parameters listed below
% false_positive  - false positve measure of observed to actual data
% false_negative - false negative measure of observed to actual data
% true_positive - true positive measure of observed to actual data
% speaker_label - label of speaker you're trying to spot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% creating output structure variable 
field1 = 'false_positive';  value1 = 0;
field2 = 'false_negative';  value2 = 0;
field3 = 'true_positive';  value3 = 0;
s = struct(field1,value1,field2,value2,field3,value3);

% converting ground_truth to frames
ground_truth(:,1:2) = round(ground_truth(:,1:2)*100);
ground_truth(1,1) = 1;
if(ground_truth(end,2)> mfccs_len)
    ground_truth(end,2) = mfccs_len;
end

if(segment_boundary(end,2) > mfccs_len)
    segment_boundary(end,2) = mfccs_len;
end
% finding speaker label in ref_segment
start_ind = abs(ground_truth(:,1) - start_point);
[start_val,start_ind] = min(start_ind);

end_ind = abs(ground_truth(:,2) - end_point);
[end_val,end_ind] = min(end_ind);
labels_array = ground_truth(start_ind:end_ind,3);
speaker_label = mode(labels_array);

% creating waveforms where '1' indicates label is marked '0' indicates label not marked 
truth_waveform = zeros(1,mfccs_len);
hypo_waveform =  zeros(1,mfccs_len);

%if(ground_truth(end,2) > mfccs_len)
%    ground_truth(end,2) = mfccs_len;
%end

% creating waveform for ground truth where ref_segment occurs
[r,c] = size(ground_truth);
for i=1:r
    if(ground_truth(i,3) == speaker_label)
    truth_waveform(ground_truth(i,1):ground_truth(i,2)) = 1;
    end
end

% creating waveform for hypothesized segment boundaries
[r,c] = size(segment_boundary);
for i=1:r
    hypo_waveform(segment_boundary(i,1):segment_boundary(i,2)) = 1;
end

% looking for false positive segments
mismatch_waveform = truth_waveform - hypo_waveform;
mismatch_ind = find(mismatch_waveform == -1);
mismatch = length(mismatch_ind);

% looking for true positive segments
true_index = find(truth_waveform);
hypo_index = find(hypo_waveform);
correct = sum(truth_waveform.*hypo_waveform);

% looking for false negative segments
fn_waveform = truth_waveform - hypo_waveform;
fn_ind = find(fn_waveform == 1);
fn = length(fn_ind);

%ind_label = find(ground_truth(:,3) == speaker_label);

% number of samples of class label_name
true_label_size = sum(truth_waveform);

s.false_positive = mismatch / (mfccs_len - true_label_size);
s.true_positive = correct / true_label_size;
s.false_negative = fn / true_label_size;

end

