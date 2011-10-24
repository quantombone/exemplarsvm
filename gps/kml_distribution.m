function kml_distribution(gps,values,mygps,name,file_names)
%Given some gps locations as well as the values associated with
%each gps coordinate, write the resulting distribution as a Google
%Earth Loadable kml file

BASEDIR = '/nfs/baikal/tmalisie/gps/siggraph2011/';
URLDIR = 'http://balaton.graphics.cs.cmu.edu/tmalisie/gps/siggraph2011/';

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

resolution = [500 500];
d1 = 30*(Gmax1-Gmin1)/resolution(1);
d2 = 30*(Gmax2-Gmin2)/resolution(2);

Gmin1 = Gmin1 - d1;
Gmax1 = Gmax1 + d1;
Gmin2 = Gmin2 - d2;
Gmax2 = Gmax2 + d2;

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
  a = alphas{i}.^.5;
  imwrite(img{i},[BASEDIR imname{i}],'Alpha',a);
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

%kml_add_placemark(fid,mygps,'image location');

fprintf(fid,'<Folder>\n');
for i = 1:length(file_names)
  desc_string = ...
      sprintf('<![CDATA[\n<img width=\"400\" src=\"%s/%s\"/>\n]]>',...
              'http://balaton.graphics.cs.cmu.edu/tmalisie/flickr_data',...
              file_names{i});

  fprintf(fid,['<Placemark>\n'...
               '<description>%s</description>\n<Point>\n' ...
               '<coordinates>%f,%f</coordinates>' ...
               '<open>1</open>\n'...
               '\n</Point>\n</Placemark>\n'],desc_string,...
          gps(2,i),...
          gps(1,i));
end

fprintf(fid,'</Folder>\n');
fprintf(fid,'</Document>\n</kml>\n');
fclose(fid);
