function [PCA_feature] = feature_create_new(speech,Fs,k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to create desired PCA feature vector using k as number of PCA components
%
% INPUTS
% speech - vector containing speech data
% Fs     - sampling frequency
% k  	 - number of principle components taken
% OUTPUTS
% feature - final PCA reduced feature vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializing parameters for MFCC extraction
Tw = 25;           % analysis frame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [ 300 3700 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 13;            % number of cepstral coefficients/window length
L = 22;            % cepstral sine lifter parameter

% creating hamming window
window = hamming(C);

% Extracting MFCC's for whole file
[MFCCs, FBEs, frames] = mfcc(speech, Fs, Tw, Ts, alpha, window, R, M, C, L);

% computing delta MFCCs
N = 2;
MFCCs = [zeros(13,N) MFCCs zeros(13,N)];
for i=N+1:length(MFCCs)-N
	sum = 0;
	for j=1:N
		sum = sum + (j*(MFCCs(:,i+j) - MFCCs(:,i-j))) ./ (2*j.^2);
	end
	delta(:,i-N) = sum;
end	
MFCCs = MFCCs(:,N+1:end-N);

% computing delta-delta MFCCs
delta = [zeros(13,N) delta zeros(13,N)];
for i=N+1:length(delta)-N
	sum = 0;
	for j=1:N
		sum = sum + (j*(delta(:,i+j) - delta(:,i-j))) ./ (2*j.^2);
	end
	delta_delta(:,i-N) = sum;
end	
delta = delta(:,N+1:end-N);

% computing delta-delta-delta MFCCs
delta_delta = [zeros(13,N) delta_delta zeros(13,N)];
for i=N+1:length(delta_delta)-N
	sum = 0;
	for j=1:N
		sum = sum + (j*(delta_delta(:,i+j) - delta_delta(:,i-j))) ./ (2*j.^2);
	end
	delta_3(:,i-N) = sum;
end
delta_delta = delta_delta(:,N+1:end-N);

% concatanating to create new feature vector
% not using energy of MFCCs but including energies of other terms
comb_feature = vertcat(MFCCs(2:end,:), delta, delta_delta, delta_3);
trans_mat = princomp(comb_feature');
PCA_feature = trans_mat' * comb_feature;
PCA_feature = PCA_feature(1:k,:);
PCA_feature = num2cell(PCA_feature,1);

end


