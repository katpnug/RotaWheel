function [ BLOBclassified ] = Bbox_classification_multicolor_cluster_left( input_path )
%{
USAGE:
bbox_final = bbox_classified(BLOB)
 
Input ---------------------------------------------------------------------
    BLOB: structure with noisy bounding box insertion (color based)
          * generated from Bbox_insert_June2017.m
Output --------------------------------------------------------------------
    bbox_final: classified (front, hind, neither) bbox
 
METHOD:
classified based on position and bbox area information for various cases
    CASE 0: 0 bounding boxes
    CASE 1: 1 bounding box
    CASE 2: 2 bounding boxes
    CASE 3: >2 bounding boxes
 
    current: use MLE to classify front/hind
 
ASSUMPTIONS: (from previous hand labeled data)
    area of paw bbox    : mu/sigma
    x pixel range of paw: mu/sigma
 
**NOTE: BLOB.AREA is different front width x height of rectangle...
 
Katrina P Nguyen, June 2017
%}
%% Starting to process the current clip
cd(input_path);
file_name = dir([pwd,'/*oBLOB*.mat']);
load(file_name(1).name);

% =========================================================================
% REDEFINE BLOB AREA: using [x y width height] coord for calculation...
% Otherwise, these are values of blob PIXELS in BW image from bbox inser
BLOB_GREEN.AREA = cellfun(@(x) x(:,3).*x(:,4),BLOB_GREEN.BBOX,'uniformoutput',false);
BLOB_GREEN.CENTROID = cellfun(@(x) [x(:,1)+x(:,3)/2 x(:,2)+x(:,4)/2],BLOB_GREEN.BBOX,'uniformoutput',false);

BLOB_ORANGE.AREA = cellfun(@(x) x(:,3).*x(:,4),BLOB_ORANGE.BBOX,'uniformoutput',false);
BLOB_ORANGE.CENTROID = cellfun(@(x) [x(:,1)+x(:,3)/2 x(:,2)+x(:,4)/2],BLOB_ORANGE.BBOX,'uniformoutput',false);
% =========================================================================
BLOBclassified = struct;
BLOBclassified.FRONTclass = {};
BLOBclassified.HINDclass  = {};

num_frames = size(BLOB_BODY_object.BBOX,2); % number of frames

% find sum of prob from bbox area and position
% x-y pos
load('handlabel_distribution_left.mat');
BLOBclassified.wheelcenter = wheel_cam_center(1,:);

% -------------------------------------------------------------------------
% need to normalize center of ss ellipse to wheel center...
% method: center merged ellipse to selected wheel center; adjust front and
%         hind appropriately (only adjusting x-pos)
adjustConst = BLOBclassified.wheelcenter(1) - wheel_center_gTruth;

front_stats.mu_xpos_ss  = muFront_xpos_ss + adjustConst;
front_stats.mu_ypos_ss  = muFront_ypos_ss;
front_stats.sig_xpos_ss = sigFront_xpos_ss;
front_stats.sig_ypos_ss = sigFront_ypos_ss;
front_stats.cov_pos_ss = covFront_pos_ss;

front_stats.mu_xpos  = muFront_xpos_ss + adjustConst;
front_stats.sig_xpos = sigFront_xpos_ss;
front_stats.mu_ypos  = muFront_ypos_ss;
front_stats.sig_ypos = sigFront_ypos_ss;
front_stats.cov_pos  = covFront_pos_ss;

front_stats.mu_area = muFront_area;
front_stats.sig_area = sigFront_area;

hind_stats.mu_xpos_ss   = muHind_xpos_ss + adjustConst;
hind_stats.mu_ypos_ss   = muHind_ypos_ss;
hind_stats.sig_xpos_ss = sigHind_xpos_ss;
hind_stats.sig_ypos_ss = sigHind_ypos_ss;
hind_stats.cov_pos_ss = covHind_pos_ss;

hind_stats.mu_xpos  = muHind_xpos_ss + adjustConst;
hind_stats.sig_xpos = sigHind_xpos_ss;
hind_stats.mu_ypos  = muHind_ypos_ss;
hind_stats.sig_ypos = sigHind_ypos_ss;
hind_stats.cov_pos  = covHind_pos_ss;

hind_stats.mu_area = muHind_area;
hind_stats.sig_area = sigHind_area;

hind_stats.dist_thresh = 100; % initiate distance threshold
front_stats.dist_thresh = 50;

for frame_iter = 1:num_frames
     noisy_bbox = imread('/home/katrinan/wheel1/Frame000001.png'); % temp...don't need frames to classify

    [ BLOBclassified noisy_bbox front_stats] = classify_paw_front_cluster(BLOB_ORANGE, BLOBclassified, 'FRONT',...
        'FRONTclass', front_stats.dist_thresh, frame_iter, num_frames, noisy_bbox, front_stats, BLOBclassified.wheelcenter,...
        [[250 200 0];[0 250 0];[0 100 255];[255 0 0]]);
    [ BLOBclassified final_bbox hind_stats] = classify_paw_hind_cluster(BLOB_GREEN, BLOBclassified, 'HIND',...
        'HINDclass', hind_stats.dist_thresh, frame_iter, num_frames, noisy_bbox, hind_stats, BLOBclassified.wheelcenter,...
        [[250 200 0];[0 250 0];[0 100 255];[255 0 0]]);
end
[~,folder]=fileparts(pwd);
save(strcat('BLOBclassified_',folder,'_',date),'BLOBclassified');
clearvars -except bname files directoryNames
cd ..
end