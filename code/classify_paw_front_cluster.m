function [ BLOBclassified final_bbox stats] = classify_paw_front_cluster(BLOB, BLOBclassified, pawID, PAWclass, dist_thresh, frame_iter, num_frames, noisy_bbox, stats, wheel_center, bboxColor)
%% constants
frame_name = BLOB.FRM;
init_frame = strcat(frame_name{1});
BLOBclassified.wheelcenter = wheel_center;
image_size   = [632 832]; %size(imread(init_frame)); % image size in pixels [row col z]
classLabel   = {'FRONT','HIND','MERGED','NOISE'}; % possibly bbox labels
confidence   = 'first frame';
alpha        = 0.75;
pA           = 1;
pP           = 1;
f            = 5; % number of parts to divide merged bbox into
LL_thresh    = -30;
min_dist = 25;
forbidden_zone = [1.0000   0.0000   34.0000  634.0000;... % left edge
    1.0000   0.0000  832.0000   50.0000;... % top edge
    767.0000    2.0000   65.0000  631.0000;...% right edge
    369.0000  552.0000  110.0000   73.0000;...% LED box
    350.0000  425.0000   96.0000  150.0000;...% LED box 2
    245.0000   97.0000  274.0000  125.0000]; % camera box

FS = 20;
LW = 7;
%%
disp(sprintf('Frame %s of %s',BLOB.FRM{frame_iter},BLOB.FRM{num_frames}));

% =========================================================================
% first, we should elimate any bbox in forbidden zone (defined above)
% bboxOverlapRatio: compute bounding box overlap ratio
%                   anything > 0 will be discarded
BLOBclassified.FZnoise{frame_iter} = bboxOverlapRatio(BLOB.BBOX{frame_iter},forbidden_zone);

% identify bbox info for rejected data (in forbidden zone)
% find ALL rows that contain a logical 1; num col in FZnoise = num FZ
rejected_bbox = BLOB.BBOX{frame_iter}(any(BLOBclassified.FZnoise{frame_iter}~=0,2),:);
% label forbidden zone noise
final_bbox = insertObjectAnnotation(noisy_bbox,'rectangle',rejected_bbox,'NOISE','color','r','FontSize',FS,'linewidth',LW);

% define new var with updated bbox WITHOUT noise from forbidden zone
BLOBclassified.BBOX{frame_iter} = BLOB.BBOX{frame_iter}(~any(BLOBclassified.FZnoise{frame_iter}~=0,2),:); % bbox
BLOBclassified.AREA{frame_iter} = BLOB.AREA{frame_iter}(~any(BLOBclassified.FZnoise{frame_iter}~=0,2),:); % area
BLOBclassified.CENTROID{frame_iter} = BLOB.CENTROID{frame_iter}(~any(BLOBclassified.FZnoise{frame_iter}~=0,2),:); % area
% =========================================================================
% next, merge all bounding boxes with distances LESS than some defined
% threshold. current value: 50

% pairwise distance between centroids (euclidean)
bbox_dist = pdist2(double([BLOBclassified.CENTROID{frame_iter}(:,1),BLOBclassified.CENTROID{frame_iter}(:,2)]),...
    double([BLOBclassified.CENTROID{frame_iter}(:,1),BLOBclassified.CENTROID{frame_iter}(:,2)]));
% grab only upper triangle of symmetric matrix
bbox_dist_upper = triu(bbox_dist);
[row col] = find(bbox_dist_upper<min_dist & bbox_dist_upper~=0);

% overlapping bboxes also need to be merged
[row_overlap col_overlap]=find(diag(fliplr(bboxOverlapRatio(BLOBclassified.BBOX{frame_iter},BLOBclassified.BBOX{frame_iter})))>0 & ...
    diag(fliplr(bboxOverlapRatio(BLOBclassified.BBOX{frame_iter},BLOBclassified.BBOX{frame_iter})))~=1);
% index through bboxes that need to be merged together
% need to be careful with multiple merges
% ex. broken BBs for BOTH front and hind; need to iterate and merge
%     separately, otherwise will be one giant BB
while ~isempty(row) | ~isempty(row_overlap)
    bbox2merge = BLOBclassified.BBOX{frame_iter}(unique([row(~isempty(row)),col(~isempty(row)),row_overlap']),:);
    area2merge = BLOBclassified.AREA{frame_iter}(unique([row(~isempty(row)),col(~isempty(row)),row_overlap']),:);
    centroid2merge = BLOBclassified.CENTROID{frame_iter}(unique([row(~isempty(row)),col(~isempty(row)),row_overlap']),:);
    % convert [x y width height] to [xmin ymin xmax ymax]
    xmin = bbox2merge(:,1);
    ymin = bbox2merge(:,2);
    xmax = xmin + bbox2merge(:,3) - 1;
    ymax = ymin + bbox2merge(:,4) - 1;
    % Compose the merged bounding boxes using the [x y width height] format.
    bbox_merge     = [min(xmin) min(ymin) max(xmax)-min(xmin)+1 max(ymax)-min(ymin)+1];
    area_merge     = bbox_merge(:,3).*bbox_merge(:,4);
    centroid_merge = [bbox_merge(:,1)+bbox_merge(:,3)/2 bbox_merge(:,2)+bbox_merge(:,4)/2];
    % redefine final BBOX output
    merge_idx = ~ismember(BLOBclassified.BBOX{frame_iter}(:,1:4),bbox2merge(:,1:4),'rows'); % to update, otherwise problematic with repeats
    BLOBclassified.BBOX{frame_iter} = [BLOBclassified.BBOX{frame_iter}(merge_idx,:); bbox_merge]; % bbox
    BLOBclassified.AREA{frame_iter} = [BLOBclassified.AREA{frame_iter}(merge_idx,:); area_merge]; % area
    BLOBclassified.CENTROID{frame_iter} = [BLOBclassified.CENTROID{frame_iter}(merge_idx,:); centroid_merge]; % area
    
    % update euclidean distances with merged boxes
    % pairwise distance between centroids (euclidean)
    bbox_dist = pdist2(double([BLOBclassified.CENTROID{frame_iter}(:,1),BLOBclassified.CENTROID{frame_iter}(:,2)]),...
        double([BLOBclassified.CENTROID{frame_iter}(:,1),BLOBclassified.CENTROID{frame_iter}(:,2)]));
    % grab only upper triangle of symmetric matrix
    bbox_dist_upper = triu(bbox_dist);
    [row col] = find(bbox_dist_upper<min_dist & bbox_dist_upper~=0);
    [row_overlap col_overlap]=find(diag(fliplr(bboxOverlapRatio(BLOBclassified.BBOX{frame_iter},BLOBclassified.BBOX{frame_iter})))>0 & ...
        diag(fliplr(bboxOverlapRatio(BLOBclassified.BBOX{frame_iter},BLOBclassified.BBOX{frame_iter})))~=1);
end

% find sum of prob from bbox area and position
% x-y pos

mu_xpos_ss  = stats.mu_xpos_ss;
mu_ypos_ss  = stats.mu_ypos_ss;
sig_xpos_ss = stats.sig_xpos_ss;
sig_ypos_ss = stats.sig_ypos_ss;
cov_pos_ss = stats.cov_pos_ss;

mu_xpos  = stats.mu_xpos;
sig_xpos = stats.sig_xpos;
mu_ypos  = stats.mu_ypos;
sig_ypos = stats.sig_ypos;
cov_pos  = stats.cov_pos;

p_front_pos   = pP*mvnpdf(double(BLOBclassified.CENTROID{frame_iter}),[mu_xpos mu_ypos],cov_pos);
p_front_area  = pA*normpdf(double(BLOBclassified.AREA{frame_iter}(:,1)),stats.mu_area,stats.sig_area);
BLOBclassified.logprob{frame_iter} = log([p_front_pos.*p_front_area repmat(nan,[size(p_front_area,1) 1])]);
paw_label = 1;

% CLEAN UP CLASSIFICATION
% i.e. NO repeats for FRONT/HIND/MERGED
% column [FRONT HIND MERGED] of max value in each row [# BBOX]
[~,cc] = max(BLOBclassified.logprob{frame_iter},[],2);
[val,ind] = unique(cc,'rows'); % find unique indices
[n bin] = histc(cc,val); % find num of reps
notunique = setdiff(1:size(cc,1),ind);   % NON unique indices

% create forbidden index list
forbiddenNDX = [setdiff(val,4)]; % 4 (noise) is okay to repeat; [setdiff(val(n==1),4)];
repeatedClass = intersect(cc,cc(notunique));
repeatedClass = cc(notunique);
repeatedClass = repeatedClass(~ismember(repeatedClass,4)); % get rid of class = 4...

while ~isempty(notunique) && sum(~ismember(cc(notunique),4))>0 % while there are repeats and are not noise...
    % double classification
    % higher values = more likely, so column 4 is more likely for noise
    % to pick which BB should be discarded as noise
    % take mean of row and select lowest as noise
    % [FRONT HIND -NOISE]..... note the NEG noise
    %         for rep_iter = 1:length(repeatedClass) % iterate through multi classes
    while ~isempty(repeatedClass)
        rep_iter = 1;
        multi_rowidx = find(cc==intersect(cc,repeatedClass(rep_iter)));
                
        % finding least prob for double classification option...
        [cc2]= find(min(BLOBclassified.logprob{frame_iter}(multi_rowidx,repeatedClass(rep_iter))) == ...
            BLOBclassified.logprob{frame_iter}(:,repeatedClass(rep_iter)));
        
        % set initial classification to updated noise assignment
        if isempty(setxor(1:2,[repeatedClass(rep_iter) forbiddenNDX']))
            cc(cc2) = 4;
        else
            cc(cc2) = 4;% cc3 % just push doubles to empty
        end
        [val,ind] = unique(cc,'rows'); % find unique indices
        [n bin] = histc(cc,val); % find num of reps
        notunique = setdiff(1:size(cc,1),ind);   % NON unique indices
        
        % update forbidden index list
        forbiddenNDX = [forbiddenNDX' setdiff(val,4)'];
        % update forbidden index list
        forbiddenNDX = [setdiff(val,4)]; % 3 (noise) is okay to repeat
        repeatedClass = intersect(cc,cc(notunique));
        repeatedClass = cc(notunique);
        repeatedClass = repeatedClass(~ismember(repeatedClass,4)); % get rid of class = 3...
        
    end
end
num_bbox = size(BLOBclassified.BBOX{frame_iter},1);
label = classLabel(cc);
color = bboxColor(cc,:);

% store appropriate bounding boxes as front/hind/noise
% cc = 0: fill in bbox as empty cells
BLOBclassified.(PAWclass)(:,frame_iter) = [{BLOBclassified.BBOX{frame_iter}(find(cc==0),:)}; BLOB.FRM(frame_iter); {'case 1'}; {confidence};...
    {BLOBclassified.logprob{frame_iter}(find(cc==0),:)}]; % front
switch num_bbox
    case {0} % num bbox = 0
        confidence = 'ambiguous';
        BLOBclassified.info(:,frame_iter) = ...
            {BLOB.FRM{frame_iter}; 'case 0'; confidence};
        final_bbox = insertObjectAnnotation(final_bbox,'rectangle',...
            BLOBclassified.BBOX{frame_iter},{''},'color',color,'FontSize',FS,'linewidth',LW);
        
    case {1} % num bbox = 1
        confidence = 'single bbox';
        BLOBclassified.info(:,frame_iter) = ...
            {BLOB.FRM{frame_iter}; 'case 1'; confidence};
        
        if ~isempty(max(find(~cellfun(@isempty, BLOBclassified.(PAWclass)(1,1:end-1))))) % need to take care of merged frames at beginning...
            lastBB       = BLOBclassified.(PAWclass){1,max(find(~cellfun(@isempty, BLOBclassified.(PAWclass)(1,1:end-1))))};
            lastCENTROID = [lastBB(1)+lastBB(3)/2 lastBB(2)+lastBB(4)/2];
        else
            lastCENTROID = [nan nan];
        end
        
        % again...check distances.
        PAWdist = pdist2(double(lastCENTROID),double(BLOBclassified.CENTROID{frame_iter}(find(cc==paw_label),:)));
        if PAWdist > dist_thresh % merged cannot be front...dist too far away
            confidence = 'ambiguous';
            BLOBclassified.(PAWclass)(:,frame_iter) = [{BLOBclassified.BBOX{frame_iter}(find(cc==4),:)}; BLOB.FRM(frame_iter); {'case 1'}; {confidence};...
                {BLOBclassified.logprob{frame_iter}(find(cc==4),:)};]; % front
            % if PAWdist > dist_thresh, label as noise
            cc = 4;
            label = classLabel(cc);
            color = bboxColor(cc,:);
            final_bbox = insertObjectAnnotation(final_bbox,'rectangle',...
                BLOBclassified.BBOX{frame_iter},label,'color',color,'FontSize',FS,'linewidth',LW);
        else
            confidence = 'unambiguous';
            BLOBclassified.(PAWclass)(:,frame_iter) = [{BLOBclassified.BBOX{frame_iter}(find(cc==paw_label),:)}; BLOB.FRM(frame_iter); {'case 1'}; {confidence};...
                {BLOBclassified.logprob{frame_iter}(find(cc==paw_label),:)}]; % front
            final_bbox = insertObjectAnnotation(final_bbox,'rectangle',...
                BLOBclassified.BBOX{frame_iter},label,'color',color,'FontSize',FS,'linewidth',LW);
        end
    otherwise
        % Goal: elimnate bbox to get down to 2! ...not really
        confidence = 'multi bbox';
        BLOBclassified.info(:,frame_iter) = ...
            {BLOB.FRM{frame_iter}; 'case 3'; confidence};
        
        if ~isempty(max(find(~cellfun(@isempty, BLOBclassified.(PAWclass)(1,1:end-1))))) % need to take care of merged frames at beginning...
            lastBB       = BLOBclassified.(PAWclass){1,max(find(~cellfun(@isempty, BLOBclassified.(PAWclass)(1,1:end-1))))};
            lastCENTROID = [lastBB(1)+lastBB(3)/2 lastBB(2)+lastBB(4)/2];
        else
            lastCENTROID = [nan nan];
        end
        
        % again...check distances.
        PAWdist = pdist2(double(lastCENTROID),double(BLOBclassified.CENTROID{frame_iter}(find(cc==paw_label),:)));
        if PAWdist > dist_thresh % merged cannot be front...dist too far away
            confidence = 'ambiguous';
            % if PAWdist > dist_thresh, label as noise
            cc(cc~=4) = 4;
            label = classLabel(cc);
            color = bboxColor(cc,:);
            BLOBclassified.(PAWclass)(:,frame_iter) = [{BLOBclassified.BBOX{frame_iter}(find(cc==paw_label),:)}; BLOB.FRM(frame_iter); {'case 1'}; {confidence};...
                {BLOBclassified.logprob{frame_iter}(find(cc==paw_label),:)}]; % front
            
            final_bbox = insertObjectAnnotation(final_bbox,'rectangle',...
                BLOBclassified.BBOX{frame_iter},label,'color',color,'FontSize',FS,'linewidth',LW);
        else
            confidence = 'unambiguous';
            BLOBclassified.(PAWclass)(:,frame_iter) = [{BLOBclassified.BBOX{frame_iter}(find(cc==paw_label),:)}; BLOB.FRM(frame_iter); {'case 1'}; {confidence};...
                {BLOBclassified.logprob{frame_iter}(find(cc==paw_label),:)}]; % front
            final_bbox = insertObjectAnnotation(final_bbox,'rectangle',...
                BLOBclassified.BBOX{frame_iter},label,'color',color,'FontSize',FS,'linewidth',LW);
        end
        
        % if label = front, hind, merged...kick merge to noise
        if all(ismember([paw_label 4],cc))
            label = classLabel(cc); % FRONT, HIND
            color = bboxColor(cc,:);
            BLOBclassified.(PAWclass)(:,frame_iter) = [{BLOBclassified.BBOX{frame_iter}(find(cc==paw_label),:)}; BLOB.FRM(frame_iter); {'case 2'}; {confidence};...
                {BLOBclassified.logprob{frame_iter}(find(cc==paw_label),:)}]; % front
            final_bbox = insertObjectAnnotation(final_bbox,'rectangle',...
                BLOBclassified.BBOX{frame_iter},label,'color',color,'FontSize',FS,'linewidth',LW);
        end
        
end

% if both F and H have been classified, and there exists a bbox, front
% CAN't be behind hind paw > reclassify as noise

if size(BLOBclassified.FRONTclass,2) == size(BLOBclassified.HINDclass,2) && ...
        ~isempty(BLOBclassified.FRONTclass{1,frame_iter}) && ~isempty(BLOBclassified.HINDclass{1,frame_iter})
    
    front_xpos = double(BLOBclassified.FRONTclass{1,frame_iter}(1) + BLOBclassified.FRONTclass{1,frame_iter}(3)/2);
    hind_xpos  = double(BLOBclassified.HINDclass{1,frame_iter}(1) + BLOBclassified.HINDclass{1,frame_iter}(3)/2);
    
    if front_xpos > hind_xpos
        confidence = 'ambiguous';
        % if PAWdist > dist_thresh, label as noise
        cc(cc==1) = 4;
        label = classLabel(cc);
        color = bboxColor(cc,:);
        BLOBclassified.(PAWclass)(:,frame_iter) = [{BLOBclassified.BBOX{frame_iter}(find(cc==paw_label),:)}; BLOB.FRM(frame_iter); {'case 1'}; {confidence};...
            {BLOBclassified.logprob{frame_iter}(find(cc==paw_label),:)}]; % front
        final_bbox = insertObjectAnnotation(final_bbox,'rectangle',...
            BLOBclassified.BBOX{frame_iter},label,'color',color,'FontSize',FS,'linewidth',LW);
    end
end

% add const to relax dist threshold if previous frame label was noise...
if frame_iter ~= 1 && isempty(BLOBclassified.(PAWclass){1,frame_iter-1})
    dist_const = 200;
else
    dist_const = 0; % if frame was labeled, don't relax
end

aa = cc; % to get LL for appropriate labeled BB..noise was a problem
aa(cc==4) = 2;
if ismember([paw_label],cc(find(diag(BLOBclassified.logprob{frame_iter}(:,aa)) > LL_thresh==1)))
    % set confidence flag
    confidence = 'unambiguous';
    stats.dist_thresh = 50 + dist_const;
    
    % front
    stats.mu_xpos  = double(BLOBclassified.CENTROID{frame_iter}(find(cc==paw_label),1)); % x centroid for front
    stats.sig_xpos = double(BLOBclassified.BBOX{frame_iter}(find(cc==paw_label),3));    % width of bbox for front
    stats.mu_ypos  = double(BLOBclassified.CENTROID{frame_iter}(find(cc==paw_label),2));
    stats.sig_ypos = double(BLOBclassified.BBOX{frame_iter}(find(cc==paw_label),4));    % height of bbox for front
    % set var(xpos) and var(ypos) to bbox width and height
    stats.cov_pos = [sig_xpos^2/2 -sig_ypos/sig_xpos;...
        -sig_ypos/sig_xpos sig_ypos^2/2];
    
else
    % set confidence flag
    confidence = 'ambiguous';
    stats.dist_thresh = 75 + dist_const;
    
    % write decay to gaussian steady state here
    stats.mu_xpos = alpha*(mu_xpos - mu_xpos_ss) + mu_xpos_ss;
    stats.sig_xpos = alpha*(sig_xpos - sig_xpos_ss) + sig_xpos_ss;
    stats.mu_ypos = alpha*(mu_ypos - mu_ypos_ss) + mu_ypos_ss;
    stats.sig_ypos = alpha*(sig_ypos - sig_ypos_ss) + sig_ypos_ss;
    stats.cov_pos = alpha*(cov_pos - cov_pos_ss) + cov_pos_ss;
end


end



