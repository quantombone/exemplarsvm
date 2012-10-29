function labelme2pascal(D, databasename, HOMELMIMAGES, folderdestination)
%
% HOMELMIMAGES = 'http://labelme.csail.mit.edu/Images'; 
% pascalfolder = '/databases/'
% databasename = 'SUN';
%
% labelme2pascal(D, databasename, HOMELMIMAGES, pascalfolder)

Nimages = length(D);

% Create folders
mkdir(fullfile(folderdestination, databasename, 'JPEGImages'))
mkdir(fullfile(folderdestination, databasename, 'Annotations'))


% LOAD - AND WRITE IMAGES
disp('Writing images')
for n = 1:Nfiles
    filename = D(n).annotation.filename;
    filename_annotation =  strrep(filename,'.jpg','.xml');

    % Load image
    img = LMimread(D, n, HOMELMIMAGES); % Load image
    [nrows ncols cc] = size(img);
    
    % Write image
    imwrite(img, fullfile(folderdestination, databasename, 'JPEGImages', filename), 'jpg', 'quality', 100);
    
    Nobjects = LMcountobject(D(n));
    
    % Translate annotation to pascal format
    clear v
    v.annotation.folder = databasename;
    v.annotation.filename = filename;
    v.annotation.source.database = databasename;
    v.annotation.size.width = ncols;
    v.annotation.size.height = nrows;
    v.annotation.size.depth = cc;
    v.annotation.segmented = 0;
    
    boundingbox = LMobjectboundingbox(D(n).annotation); % [xmin ymin xmax ymax]
    for m = 1:Nobjects
        v.annotation.object(m).name = D(n).annotation.object(m).name;
        v.annotation.object(m).truncated = D(n).annotation.object(m).crop;
        v.annotation.object(m).difficult = D(n).annotation.object(m).crop;
        v.annotation.object(m).bndbox.xmin = boundingbox(m,1);
        v.annotation.object(m).bndbox.ymin = boundingbox(m,2);
        v.annotation.object(m).bndbox.xmax = boundingbox(m,3);
        v.annotation.object(m).bndbox.ymax = boundingbox(m,4);
    end

    % Write annotation file    
    writeXML(fullfile(folderdestination, databasename, 'Annotations', filename_annotation), v);
end
