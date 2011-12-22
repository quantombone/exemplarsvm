function res = precompute_james_nearest
%Load landmark/name/fg/bg sets

filer = '/nfs/baikal/tmalisie/siggraph2011/landmark_fgs.mat';

if fileexists(filer)
  res = load(filer);
  return;
end

%Get gps coordinates from wikipedia search
landmark_gps = cell(0,1);
landmark_name = cell(0,1);

landmark_fg = cell(0,1);
landmark_bg = cell(0,1);

landmark_gps{end+1} = [48.853033; 2.34969]; %notre dame
landmark_name{end+1} = 'notre_dame'; %'Notre Dame de Paris';

landmark_gps{end+1} = [27.174799; 78.042111]; % taj mahal
landmark_name{end+1} = 'taj';%'Taj Mahal';

landmark_gps{end+1} = [48.8583; 2.2945]; % Eiffel Tower
landmark_name{end+1} = 'eiffel_tower';%'Eiffel Tower';

landmark_gps{end+1} = [41.890169; 12.492269]; % Colosseum
landmark_name{end+1} = 'colliseum';%'Colosseum';

landmark_gps{end+1} = [45.4375; 12.335833]; % Venice City
landmark_name{end+1} = 'venice';%'Venice City';

%Load gps coordinates of all 6.5 million images
load all_gps.mat
gps = double(gps);

for i = 1:length(landmark_name)
  fprintf(1,'%d/%d\n',i,length(landmark_name));
  
  %Get distances between gps of all images to to landmark gps
  distances = get_gps_ball(gps,landmark_gps{i});
  
  %Take top 10000 closest images to landmark
  [aa,bb] = sort(distances);
  
  %"fg" are the close images
  fg = bb(1:10000);
  
  %"bg" are the faraway images (randomly dispersed from all far images)
  bg = bb(1000000:end);
  subinds = round(linspace(1,length(bg),1000));
  bg = bg(subinds);
  
  landmark_fg{end+1} = fg;
  landmark_bg{end+1} = bg;  
end

save(filer,'landmark_gps','landmark_name','landmark_fg', ...
     'landmark_bg');

res.landmark_gps = landmark_gps;
res.landmark_name = landmark_name;
res.landmark_fg = landmark_fg;
res.landmark_bg = landmark_bg;
