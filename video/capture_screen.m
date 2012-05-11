function imgData = capture_screen(capture_region)
%Given a subset of the entire screen's visible area (capture_region), capture
%a screenshot using the java Robot class, if capture_region is not present
%then capture entire viewable area Returns an image imgData
% NOTE: This is all done purely in java, without having to write the
% image to disk!
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

robo = java.awt.Robot;
t = java.awt.Toolkit.getDefaultToolkit();

% fid = fopen('/tmp/coords.txt','r');
% coords = fscanf(fid,'%d %d %d %d');
% fclose(fid);

% fid = fopen('/tmp/coords.txt','w');
% fprintf(fid,'1 1 1 1\n');
% fclose(fid);

% [aa,bb]=unix(' kill -SIGINT `ps aux | grep python | grep rectframe.py | head -1 | awk ''{print($2)}''`');
% fprintf(1,'just sent signal 1\n');
% pause(.1);

if exist('capture_region','var')
  %rectangle = java.awt.Rectangle(capture_region(1),capture_region(2),capture_region(3), ...
  %capture_region(4));
  rectangle = java.awt.Rectangle(capture_region(1),...
                                 capture_region(2),...
                                 capture_region(3)-capture_region(1)+1,...
  capture_region(4)-capture_region(2)+1);

else
  rectangle = java.awt.Rectangle(t.getScreenSize());  

end
javaImage = robo.createScreenCapture(rectangle);

H=javaImage.getHeight;
W=javaImage.getWidth;
imgData = zeros([H,W,3],'uint8');
pixelsData = reshape(typecast(javaImage.getData.getDataStorage, ...
                              'uint32'),W,H)';

imgData(:,:,3) = bitshift(bitand(pixelsData,256^1-1),-8*0);
imgData(:,:,2) = bitshift(bitand(pixelsData,256^2-1),-8*1);
imgData(:,:,1) = bitshift(bitand(pixelsData,256^3-1),-8*2);


% fid = fopen('/tmp/coords.txt','w');
% fprintf(fid,'%d %d %d %d\n',coords(1),coords(2),coords(3),coords(4));
% fclose(fid);
% [aa,bb]=unix([' kill -SIGINT `ps aux | grep python | grep rectframe.py' ...
%               ' | head -1 | awk ''{print($2)}''`']);
% fprintf(1,'just sent signal 2\n');
% pause(.1);
