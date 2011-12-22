function [distances] =  ...
    get_gps_ball(gps, landmark_gps)
%Get distances of im2gps images from landmark (in meters)

if ~exist('landmark_gps','var')
  %notre dame
  notre_dame_gps = [48.853; 2.3498];
  landmark_gps = notre_dame_gps;
end

%cartesian coords of everything
cart = gps2cart(gps')';
landmark_cart = gps2cart(landmark_gps')';
distances = sqrt(sum(bsxfun(@minus,cart,landmark_cart).^2,1));
%There might be some numerical errors when performing this computation
%distances = sqrt(distSqr_fast(cart,landmark_cart));
