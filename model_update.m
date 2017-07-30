function [new_agmm] = model_update(ubm,MFCCs,adapt_frames)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to update Adapted GMM 
% INPUTS
% old-agmm     - old Adapted GMM to be updated with new data
% ubm          - UBM w.r.t which adaptation was done   
% MFCCs        - MFCCs computed on whole speech file 
% adapt_frames - frames of speech used to update old_agmm 
%  
% OUTPUTS
% new_agmm     - new GMM updated using frames given
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[r,c] = size(adapt_frames);

% concatanating data frames for Re-Adaptation 
%data = [];
%for i=1:r
%    data = horzcat(data,MFCCs(adapt_frames(i,1):adapt_frames(i,2)));
%end
%data = data(2:end);

% Re-Adapting agmm
tau = 19;
%for i=1:r
new_agmm = mapAdapt(MFCCs(adapt_frames), ubm, tau, 'm');
%end

end

