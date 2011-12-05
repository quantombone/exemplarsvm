function imgData = ScreenCapture(subber)
%Given a subset of the entire screen's visible area (subber), capture a screenshot
%using the java Robot class, if subber is not present then capture entire
%viewable area
%Returns an image imgData

robo = java.awt.Robot;
t = java.awt.Toolkit.getDefaultToolkit();

if exist('subber','var')
  rectangle = java.awt.Rectangle(subber(1),subber(2),subber(3), ...
                                 subber(4));
else
  rectangle = java.awt.Rectangle(t.getScreenSize());  
end
javaImage = robo.createScreenCapture(rectangle);
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
%if exist('subber','var')
%    imgData = imgData(subber(2):subber(4),subber(1):subber(3),:);
%end

%imgData = max(0.0,min(1.0,imresize(imgData,2)));
