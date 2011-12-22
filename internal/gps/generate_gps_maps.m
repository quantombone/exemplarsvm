function generate_gps_maps;
%% load the data sources from abhinav

index = 7;
index2 = 11;
fgindex = 13;
%cls = 'Sydney Opera';

index = 6;
index2 = 1;
fgindex = 12;
cls = 'Tower Bridge';

load(['/nfs/onega_no_backups/users/ashrivas/codes/iccv2011/' ...        
      'new_result_paint2gps_our.mat']);

dist = nRes{index}.dist{index2};
coords = nRes{index}.best_gps_coords{index2};
best_dist = nRes{index}.best_dist{index2};
fgs = load('/nfs/onega_no_backups/users/ashrivas/landmark_fgs_new.mat');

r = load(['/nfs/onega_no_backups/users/ashrivas/iccv2011/' ...
          'new_paintings/local/iccv2011/new_data_london_bridge/' ...
          'london_bridge0.4/4.mat']);

%r = load(['/nfs/onega_no_backups/users/ashrivas/iccv2011/' ...
%          'new_paintings/local/iccv2011/new_data_sydney_opera/' ...
%          'sydney_opera0.4/5.mat']);



NNN = 50;

inds = fgs.landmark_fg{index};
order = r.new_ordering(1:NNN);

fg = fgs.landmark_fg{fgindex};
coords = coords(1:NNN,:);
best_dist = best_dist(1:NNN);

for i = 1:length(order)
  file_names{i} = james_name(fg(order(i)));
end

cls2 = sprintf('%s-NNN=%d',strrep(cls,' ','-'),NNN);

kml_distribution(coords',best_dist,mean(coords,1)',cls2,file_names);
