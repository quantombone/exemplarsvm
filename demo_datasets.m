%Here is a data-set demo
%A. James Hays' 6.5million geotagged Flickr images

%Get gps coordinates from wikipedia search
%landmark_gps = [48.853033; 2.34969]; %notre dame
landmark_name = 'Golden Gate';
landmark_gps = [37.819722; -122.478611]; %golden gate

%Load gps coordinates of all 6.5 million images
load all_gps.mat

%Get distances between gps of all images to to landmark gps
distances = get_gps_ball(gps,landmark_gps);

%Take top 1000 closest images to landmark
[aa,bb] = sort(distances);
subset = bb(1:1000);
fprintf(1,'Total of %d images within %.3fkm of %s\n',...
        1000,aa(1000),landmark_name);

%Create an image set datastructure
fg = get_james_bg(100,subset);

%create a set of 1000 "far away" images
bg = get_james_bg(1000,bb(10000:end));

%show images
figure(1)
clf
N = 5;
for i = 1:N*N
  subplot(N,N,i)
  imagesc(convert_to_I(fg{i}))
  axis image
  axis off
  title(sprintf('Distances = %.3fkm\n',aa(i)))
end

return;
%%get exemplar from image 11 (which is of notre dame facade)
initialize_exemplars_fg(fg(11));

%%learn stuff 
update_models_fg;

%%
models = load_all_exemplars;

