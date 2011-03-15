function [fg,bg] = get_james_nearest(LEN, target_name)
%% Get LEN nearest images to landmark_gps (with some distractors) to
%% create the foreground set "fg"
%% Get LEN faraway images to create the background set "bg"

res = precompute_james_nearest;

% %Get gps coordinates from wikipedia search
% landmark_gps = cell(0,1);
% landmark_name = cell(0,1);

% landmark_gps{end+1} = [48.853033; 2.34969]; %notre dame
% landmark_name{end+1} = 'Notre Dame de Paris';

% landmark_gps{end+1} = [27.174799; 78.042111]; % taj mahal
% landmark_name{end+1} = 'Taj Mahal';

% landmark_gps{end+1} = [48.8583; 2.2945]; % Eiffel Tower
% landmark_name{end+1} = 'Eiffel Tower';

% landmark_gps{end+1} = [41.890169; 12.492269]; % Colosseum
% landmark_name{end+1} = 'Colosseum';

% landmark_gps{end+1} = [45.4375; 12.335833]; % Venice City
% landmark_name{end+1} = 'Venice City';

if ~exist('target_name','var')
  target_name = 'Eiffel Tower';
end

match = find(ismember(res.landmark_name,target_name));
if length(match) == 0
  fprintf(1,'Error name not in database\n');
  match = 1;
end

fg = res.landmark_fg{match};
bg = res.landmark_bg{match};

fg = get_james_bg(length(fg),fg);
bg = get_james_bg(length(bg),bg);

fprintf(1,'got here\n');
% landmark_gps = landmark_gps{match};

% if ~exist('LEN','var')
%   LEN = 1000;
% end

% %Load gps coordinates of all 6.5 million images
% load all_gps.mat

% %Get distances between gps of all images to to landmark gps
% distances = get_gps_ball(gps,landmark_gps);

% %Take top 1000 closest images to landmark
% [aa,bb] = sort(distances);
% subset = bb(1:1000);
% %fprintf(1,'Total of %d images within %.3fkm of %s\n',...
% %        1000,aa(1000),landmark_name);

% %Create an image set datastructure
% fg1 = get_james_bg(LEN,subset);
% fg2 = get_james_bg(LEN,bb(end-15000:end));

% %create a set of 100 "far away" images
% %fg = cat(2, fg1, fg2);
% fg = fg1;
% bg = get_james_bg(LEN,bb(end-2000:end));

% for i = 1:length(subset)
%   fprintf(1,'index %d is image %d\n',i,subset(i));
% end