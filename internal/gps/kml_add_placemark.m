function kml_add_placemark(fid,gps,name,description)
% add a placemark to a kml file with optional name/description
fprintf(fid,'<Placemark>\n');
if exist('name','var')
  fprintf(fid,'<name>%s</name>\n',name);
end
if exist('description','var')
  fprintf(fid,'<description>%s</description>\n',description);
end

fprintf(fid,'<Point>\n<coordinates>\n');
fprintf(fid,'%f,%f',gps(2),gps(1));
fprintf(fid,'</coordinates>\n</Point>\n</Placemark>\n');
