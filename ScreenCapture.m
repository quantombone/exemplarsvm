function imgData = ScreenCapture(subber)

robo = java.awt.Robot;
t = java.awt.Toolkit.getDefaultToolkit();
rectangle = java.awt.Rectangle(t.getScreenSize());
image = robo.createScreenCapture(rectangle);
%filehandle = java.io.File('screencapture.jpg');
%javax.imageio.ImageIO.write(image,'jpg',filehandle);
%imageview('screencapture.jpg');
javaImage = image;
H=javaImage.getHeight;
W=javaImage.getWidth;
imgData = zeros([H,W,3],'uint8');
pixelsData = reshape(typecast(javaImage.getData.getDataStorage,'uint32'),W,H)';
imgData(:,:,3) = bitshift(bitand(pixelsData,256^1-1),-8*0);
imgData(:,:,2) = bitshift(bitand(pixelsData,256^2-1),-8*1);
imgData(:,:,1) = bitshift(bitand(pixelsData,256^3-1),-8*2);

if exist('subber','var')
    imgData = imgData(subber(2):subber(4),subber(1):subber(3),:);
end

%imgData = max(0.0,min(1.0,imresize(imgData,2)));