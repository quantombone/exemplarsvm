function imgData = capture_screen(capture_region)
%Given a subset of the entire screen's visible area (capture_region), capture
%a screenshot using the java Robot class, if capture_region is not present
%then capture entire viewable area Returns an image imgData
% NOTE: This is all done purely in java, without having to write the
% image to disk!
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

robo = java.awt.Robot;
t = java.awt.Toolkit.getDefaultToolkit();

if exist('capture_region','var')
  rectangle = java.awt.Rectangle(capture_region(1),capture_region(2),capture_region(3), ...
                                 capture_region(4));
else
  rectangle = java.awt.Rectangle(t.getScreenSize());  
end
javaImage = robo.createScreenCapture(rectangle);

%NOTE: this part is not necessary because we can cast the data into
%a matlab structure
%filehandle = java.io.File('screencapture.jpg');
%javax.imageio.ImageIO.write(image,'jpg',filehandle);
%imageview('screencapture.jpg');

H=javaImage.getHeight;
W=javaImage.getWidth;
imgData = zeros([H,W,3],'uint8');
pixelsData = reshape(typecast(javaImage.getData.getDataStorage, ...
                              'uint32'),W,H)';

imgData(:,:,3) = bitshift(bitand(pixelsData,256^1-1),-8*0);
imgData(:,:,2) = bitshift(bitand(pixelsData,256^2-1),-8*1);
imgData(:,:,1) = bitshift(bitand(pixelsData,256^3-1),-8*2);

%NOTE: this part is not necessary because we initialized the
%robot's viewable area to "capture_region" before taking the screenshot
%if exist('capture_region','var')
%    imgData = imgData(capture_region(2):capture_region(4),capture_region(1):capture_region(3),:);
%end

%imgData = max(0.0,min(1.0,imresize(imgData,2)));
