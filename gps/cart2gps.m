function gps_coords = cart2gps(cart_coords)
%James Hays, jhhays@cs.cmu.edu
%inverse of gps2cart function

% z is the north south axis
% south latitude  is negative (according to flickr)
% west  longitude is negative (according to flickr)
 
% x axis runs from prime meridian (+1) to international dateline (-1)
% y axis runs from 90 east        (+1) to 90 west                (-1)

% does this follow the right hand rule?

%gps coordinate  0 , 0        (equator, prime meridian) is  1, 0, 0
%gps coordinate  0 , 180/-180 (equator, intl date line) is -1, 0, 0
%               90 , *        (north pole)              is  0, 0, 1
%              -90 , *        (south pole)              is  0, 0,-1
%                0 , 90W                                is  0,-1, 0
%                0 , 90E                                is  0, 1, 0

num_coords = size(cart_coords,1);

gps_coords = zeros(num_coords, 2);% [lat long] in each row

rho = 1;
for i = 1:num_coords
    cart_coords(i,:) = cart_coords(i,:) ./ norm( cart_coords(i,:) );
    
    phi   = acos( cart_coords(i,3) / rho );
    theta = atan2( cart_coords(i,2) / (rho * sin(phi)), cart_coords(i,1)/ (rho * sin(phi))) + 2*pi;
    
    gps_coords(i,1) = -((phi - (pi/2)) / (pi/2)) * 90;
    gps_coords(i,2) = ((theta - 2*pi) / (-pi/2)) * 90;

end

