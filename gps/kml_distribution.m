function kml_distribution(gps,values,mygps,name)
%Given some gps locations as well as the values associated with
%each gps coordinate, write the resulting distribution as a Google
%Earth Loadable kml file

BASEDIR = '/nfs/baikal/tmalisie/gps/';
URLDIR = 'http://balaton.graphics.cs.cmu.edu/tmalisie/gps/';

filename = sprintf('%s/%s_distribution.kml',BASEDIR,name);

%get the extent of entire data set
Gmin1 = -90; %min(gps(1,:));
Gmax1 = 90; %max(gps(1,:));

Gmin2 = -180;
Gmax2 = 180;

Gmin1 = min(gps(1,:));
Gmax1 = max(gps(1,:));
Gmin2 = min(gps(2,:));
Gmax2 = max(gps(2,:));

resolution = [500 1000];

iii = linspace(Gmin1,Gmax1,2);
jjj = linspace(Gmin2,Gmax2,2);

counter = 1;
for i = 1:length(iii)-1
  for j = 1:length(jjj)-1
    fprintf(1,'Processing Grid Overlay Chunk: (%d,%d)\n',i,j);
    tic
    [img{counter},alphas{counter},min1{counter},max1{counter}, min2{counter},max2{counter}] ...
        = kml_density_image(gps,values,resolution, [iii(i) iii(i+1) ...
                    jjj(j) jjj(j+1)]); 
    toc
    counter = counter+1;
  end
end

for i = 1:length(img)
  imname{i} = sprintf('%s_%d.png',name,i);
  imwrite(img{i},[BASEDIR imname{i}],'Alpha',alphas{i});
end

fid = fopen(filename,'w');
fprintf(fid,'<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n');
fprintf(fid,['<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n' ...
             '<Document>\n']);
fprintf(fid,'<name>Distribution: %s</name>\n',name);
fprintf(fid,'<description>number of gps coords = %d</description>\n',...
        length(values));

for i = 1:length(img)
  kml_add_overlay(fid,[URLDIR imname{i}],...
              min1{i},max1{i},min2{i},max2{i});
end

kml_add_placemark(fid,mygps,'image location');

fprintf(fid,'</Document>\n</kml>\n');
fclose(fid);
