%{
LED syncing
BBox insertion with 2 color threshold

KPN January 2018
%}
function [ x ] = Bbox_insert_sync_multicolor_cluster_left_AVI( input_path )
%Data Structures for saving relevant information
%Saving all the blob analysis output in Struct BLOB
BLOB_COUNTER = 1 ;

%% Fix the background frame
BLOB_BODY_object = struct;
BLOB_GREEN = struct;
BLOB_ORANGE = struct;
BLOB_BLUE = struct;
BLOB_PINK = struct;
BLOB_YELLOW = struct;

%% Starting to process the current clip
cd(input_path);
movie_struct = dir([pwd,'/*.avi']);
movie_name = movie_struct.name;

wheel_cam_center = [430 410;445 15];
wheel_rect = [251.0000  235.0000  360.0000  292.0000]; % limit bounding box search area for wheel dots
paw_rect = [107 98 595 525];
LED_rect = [390 450 97 181];

% load camera parameters for undistortion
load('home/katrinan/wheel1/cameraParams_left.mat');

vid = VideoReader( movie_name );
frame_num = 1;

while hasFrame(vid)
    if frame_num == 1
        ref = readFrame(vid);
        ref = flip(ref,2); % flip horizontally
        y_pos_ref  = [550:630];%  <-----CHECK THESE FOR LED CAPTURE!!!
        x_pos_ref = [375:475]; %  <-----------------------------------
        ref2 = ref(y_pos_ref,x_pos_ref,3); % ref frame LED square       
        opto_ref = ref(500:600,700:800,3);
    end

    file_name = sprintf('Frame%06.00f',frame_num);
    %% frame_nummage Processing 1 Basic contrast enhancement and filtering    
    try % sometimes (splitting process) PNG files are corrupted. skip!
        vid.CurrentTime = (frame_num-1)/100; % every time read frame, idx + 1
        frm = readFrame(vid);
        frm = flip(frm,2); % flip horizontally
        frm2 = frm(y_pos_ref,x_pos_ref,3); % current frame LED square
    catch
        warning('Corrupted image frame file!');
        frame_num = frame_num + 1;
        continue % if can't read, move onto next iteration
    end
    
    mn(frame_num) =  mean(mean(frm2))-mean(mean(ref2)); % diff in pixel intensity
    if mn(frame_num) < 30 % threshold for LED sync
        fprintf('Delete %s: %02.03f of %02.03f, %d\n',file_name,vid.CurrentTime,vid.Duration,mn(frame_num));
        frame_num = frame_num + 1;
        continue % if LED off, move onto next iteration
    else
        %Simple Contrast Enhancement
        fprintf('%s: %02.03f of %02.03f, %d\n',file_name,vid.CurrentTime,vid.Duration,mn(frame_num));
        A_feet = frm;
        A_feet = undistortImage(A_feet,cameraParams_left);
        % body center, wheel/camera subtraction
        [ BLOB_BODY_object RGB_BODY_object] = body_object_sub( A_feet, A_feet, BLOB_BODY_object, BLOB_COUNTER, file_name, wheel_cam_center );
        
        %% paw segementation
        % insert square to normalize colors without paws.
        A_feet = insertShape(A_feet,'filledrectangle',[750 25 75 50],'LineWidth',7,'color',[0 255 0]);
        A_feet = insertShape(A_feet,'filledrectangle',[650 25 75 50],'LineWidth',7,'color',[255 150 0]);
        A_feet = insertShape(A_feet,'filledrectangle',[550 25 75 50],'LineWidth',7,'color',[0 0 255]);
        A_decorr = decorrstretch(A_feet); % enhance color difference in image
       
        %Filtering for feet extraction
        % NOTE: RIGHT side - green = front (75), orange = hind (200)
        %       LEFT side  - green = hind,  orange = front
        [BLOB_GREEN RGB_BODY_object] = extract_color_blob(A_feet, A_decorr, RGB_BODY_object, [115 120 120], 100 , file_name, 'GREEN', 1, 'green', BLOB_GREEN, BLOB_COUNTER, paw_rect, LED_rect);
        [BLOB_ORANGE RGB_BODY_object]= extract_color_blob(A_feet, A_decorr, RGB_BODY_object, [200 50 60], 30, file_name, 'ORANGE', 3, [255 150 0], BLOB_ORANGE, BLOB_COUNTER, paw_rect, LED_rect);
        
        % color dot speed marker
        [BLOB_BLUE RGB_BODY_object]  = extract_color_blob(A_feet, A_decorr, RGB_BODY_object, [90 150 115], 20, ...
            file_name, 'BLUE', 2, 'blue', BLOB_BLUE, BLOB_COUNTER, wheel_rect, LED_rect);        
        
        %% opto LED on/off indicator
        optoLED = frm(500:600,700:800,3);
        opto_status(BLOB_COUNTER) = mean(mean(optoLED))-mean(mean(opto_ref));
        
        BLOB_COUNTER = BLOB_COUNTER + 1;
    end
    pause(0.01);
    frame_num = frame_num + 1;
end
[~,folder]=fileparts(pwd);
save(strcat('oBLOB_',folder,'_',movie_name(1:end-4),'_',date),'BLOB_GREEN','BLOB_ORANGE','BLOB_BLUE','BLOB_BODY_object','opto_status','mn','wheel_cam_center');

try
	movie_names_clip = dir([pwd,'/*.avi']); % find all appropriate file names
	delete(movie_names_clip.name); % permanently delete all files
catch
	warning('no movie (.avi)');
end

Bbox_classification_multicolor_cluster_left(input_path);

fprintf('Done\n')

cd .. % step out of folder
clearvars -except bname files directoryNames
exit

end


function [ BLOB RGB ] = extract_color_blob(A_feet, A_decorr, RGB, RGB_thresh, p_pixel, file_name, paw_color, case_num, Bbox_color, BLOB, BLOB_COUNTER, ref_box, led_box)

% RGB threshold values to segment paws from frame
% note the sign differences
switch case_num
    case {1}
        R = A_decorr(:,:,1) <  RGB_thresh(1);  % red threshold
        G = A_decorr(:,:,2) >  RGB_thresh(2); % green threshold
        B = A_decorr(:,:,3) < RGB_thresh(3) & A_decorr(:,:,3) > 50; % blue threshold
        A_delta = (R & G & B);     % logical, binarized image
    case {2}
        R = A_feet(:,:,1) < RGB_thresh(1);  % red threshold
        G = A_feet(:,:,2) <= RGB_thresh(2); % green threshold
        B = A_feet(:,:,3) > RGB_thresh(3); % blue threshold
        A_delta = (R & G & B);     % logical, binarized image
    case {3}
        R = A_decorr(:,:,1) > RGB_thresh(1);  % red threshold
        G = A_decorr(:,:,2) < RGB_thresh(2); % green threshold
        B = A_decorr(:,:,3) < RGB_thresh(3); % blue threshold
        A_delta = (R & G & B);     % logical, binarized image
    case {4}
        R = A_decorr(:,:,1) > RGB_thresh(1);  % red threshold
        G = A_decorr(:,:,2) < RGB_thresh(2); % green threshold
        B = A_decorr(:,:,3) > RGB_thresh(3); % blue threshold
        A_delta = (R & G & B);     % logical, binarized image
    case {5}
        R = A_decorr(:,:,1) < RGB_thresh(1);  % red threshold
        G = A_decorr(:,:,2) > RGB_thresh(2); % green threshold
        B = A_decorr(:,:,3) < RGB_thresh(3); % blue threshold
        A_delta = (R & G & B);     % logical, binarized image
end

A_delta_bin = bwareaopen(A_delta, p_pixel); % remove small objects from binary image, P pixels (p_pixel different for front vs hind)
se          = strel('square',3); % create square structuring element with width W
A_dilated   = imdilate(A_delta_bin,se);

%% Image Processing 2 Blob analysis and connected components
%Processing for feet extraction--------------------------------------------
%BlobAnalysis: built in MATLAB object to compute stats for
%connected regions in binary image
AH                       = vision.BlobAnalysis; % init blob analysis system object
AH.ExcludeBorderBlobs    = 0; %exlclude blobs that contain at least 1 image border pixel
AH.MinimumBlobArea       = 50; % mininum blob area in pixels %350
AH.LabelMatrixOutputPort = 1; % return label matrix

%step: computes and returns stats of input binary image with blob
%analysis object
[AREA,CENTROID,BBOX,LABEL] = step(AH,A_dilated);

score = ones(size(BBOX,1),1);
%Struct for saving all outputs of the blob analysis
%select strongest bbox from overlapping clusters
% sBBOX : selected bbox returned as M x 4 matrix (1 bbox per row)
% sScore: confidence score
[sBBOX,sSCORE] = selectStrongestBbox(BBOX,AREA,'RatioType','Min','OverlapThreshold', 0.5);
% find artificial bbox and exclude as an option
% overlap_BBOX = bboxOverlapRatio(sBBOX,ref_box);
% sBBOX = sBBOX(any(overlap_BBOX==0,2),:);
% find frame ref box and only include any bbox in this area
overlap_BBOX = bboxOverlapRatio(sBBOX,ref_box);
sBBOX = sBBOX(any(overlap_BBOX~=0,2),:);

overlap_BBOX_LED = bboxOverlapRatio(sBBOX, led_box);
sBBOX = sBBOX(any(overlap_BBOX_LED==0,2),:); % exclude any overlap with LED

BLOB.BBOX{BLOB_COUNTER} = sBBOX;
rw = ismember(BBOX,sBBOX,'rows');

if size(sBBOX,1) ~= 0
    BLOB.AREA{BLOB_COUNTER} = AREA(rw,:);
    BLOB.CENTROID{BLOB_COUNTER} = CENTROID(rw,:);
    BLOB.FRM{BLOB_COUNTER} = file_name;
else
    BLOB.AREA{BLOB_COUNTER} = AREA;
    BLOB.CENTROID{BLOB_COUNTER} = CENTROID;
    BLOB.FRM{BLOB_COUNTER} = file_name;
end

RGB = insertShape(RGB,'rectangle', sBBOX,'LineWidth',2,'color',Bbox_color);
%Feet extraction ends------------------------------------------------------
clear rw

end
function [ BLOB RGB_BODY ] = body_object_sub( A_feet, RGB, BLOB, BLOB_COUNTER, file_name, mask_centers )
% RGB color thresholds for body
R = A_feet(:,:,1) < 100;
G = A_feet(:,:,2) < 90;
B = A_feet(:,:,3) < 90;
% binarize
bin = R & G & B;
% wheel circle mask
[rr cc] = meshgrid(1:832,1:632);
w = sqrt((rr-mask_centers(1,1)).^2+(cc-mask_centers(1,2)+10).^2)<=200; % radius of wheel: 325
W = ~w; % black wheel, white background
out = W & bin; % subtract wheel from binarized image

% camera circle mask
[rr cc] = meshgrid(1:832,1:632);
c = sqrt((rr-mask_centers(2,1)).^2+(cc-mask_centers(2,2)).^2)<=100; % radius of camera: 125
out2 = ~c & out;   % subtract circle came
final = out2;

erode = bwareaopen(final,1); % don't really want to erode here, splits body up
se_body = strel('square',15); % create square structuring element with width W
dilate = imdilate(erode,se_body);

%FOR LEFT SIDE MARCH DATA TO REMOVE A BLACK STRIP
% dilate(515:632,1:832) = 0;
%Processing for feet extraction--------------------------------------------
%BlobAnalysis: built in MATLAB object to compute stats for
%connected regions in binary image
AH = vision.BlobAnalysis; % init blob analysis system object
AH.ExcludeBorderBlobs = 0; %exlclude blobs that contain at least 1 image border pixel
AH.MinimumBlobArea = 5000; % mininum blob area in pixels %350
AH.LabelMatrixOutputPort = 1; % return label matrix

%step: computes and returns stats of input binary image with blob
%analysis object
[AREA,CENTROID,BBOX,LABEL] = step(AH,dilate);
score = ones(size(BBOX,1),1);
%Struct for saving all outputs of the blob analysis
%select strongest bbox from overlapping clusters
% sBBOX : selected bbox returned as M x 4 matrix (1 bbox per row)
% sScore: confidence score
[sBBOX,sSCORE] = selectStrongestBbox(BBOX,AREA,'RatioType','Min','OverlapThreshold', 0.5);

% rectangular mask around led post
overlap_BBOX = bboxOverlapRatio(sBBOX,[380 565 106 62]);
sBBOX = sBBOX(any(overlap_BBOX==0,2),:);

BLOB.BBOX{BLOB_COUNTER} = sBBOX;
rw = ismember(BBOX,sBBOX,'rows');
CENTROID = nanmean(CENTROID(rw,:),1);

if size(sBBOX,1) ~= 0
    BLOB.AREA{BLOB_COUNTER} = AREA(rw,:);
    BLOB.CENTROID{BLOB_COUNTER} = CENTROID;
    BLOB.FRM{BLOB_COUNTER} = file_name;
    
    RGB_BODY = insertShape(RGB,'FilledCircle' , [CENTROID,5],'color','white');
    RGB_BODY = insertShape(RGB_BODY,'rectangle', sBBOX,'LineWidth',2,'color','white');
else
    BLOB.AREA{BLOB_COUNTER} = AREA;
    BLOB.CENTROID{BLOB_COUNTER} = CENTROID;
    BLOB.FRM{BLOB_COUNTER} = file_name;
    
    RGB_BODY = insertShape(RGB,'rectangle', sBBOX,'LineWidth',2,'color','white');
end
%Feet extraction ends------------------------------------------------------
clear rw
end