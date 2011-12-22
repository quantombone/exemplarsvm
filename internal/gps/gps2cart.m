function cart_coords = gps2cart(gps_coords)
% Code by James Hays, jhhays@cs.cmu.edu

%converts gps coordinates into cartesian coordinates on a sphere.
%gps_coords is a nx2 matrix, with each row being a latitude and longitude.

% assuming that the earth is spherical, with no altitude (rho = 1)

% see http://en.wikipedia.org/wiki/Spherical_coordinates

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

num_coords = size(gps_coords,1);

cart_coords = zeros(num_coords, 3);% [x y z] in each row

rho = 1;
for i = 1:num_coords
    phi   = (pi/2) - (pi/2) * (gps_coords(i,1)/90);  %latitude, north/south angle, range [0  , pi]
    theta =  2*pi  - (pi/2) * (gps_coords(i,2)/90);  %longitude, east/west  angle, range [ pi, 3 pi] ?
    
    cart_coords(i,1) = rho * sin(phi) * cos(theta); %x
    cart_coords(i,2) = rho * sin(phi) * sin(theta); %y 
    cart_coords(i,3) = rho * cos(phi);              %z
end

%my description might not be consistent with my math right now.  Actually,
%seems ok.

%this sphere has a diameter of one, whereas the earth has a mean radius of
% 6,371km  ( http://en.wikipedia.org/wiki/Earth )
% so if we want to work in units of kilometers, we can simply multiply by
% 6371.  Then the distance between two points (especially for nearby
% points) is a reasonable estimate of distance.  The true distance would be
% a great circle distance, probably, but whatever.

cart_coords = cart_coords * 6371;