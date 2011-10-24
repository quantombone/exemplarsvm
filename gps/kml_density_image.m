function [img,alphas,min1,max1,min2,max2] = ...
    kml_density_image(gps,values,resolution,params)
%Compute the Density Image over the gps range specified by gps

if ~exist('params','var')

  % for viewing the entire dataset
  min1 = min(gps(1,:));
  max1 = max(gps(1,:));
  %min2 = min(gps(2,:));
  %max2 = max(gps(2,:));
  min2 = -180;
  max2 = 180;
  
  %europe extent
  %min1 = 35;
  %max1 = 55;
  %min2 = -10;
  %max2 = 20;
else
  min1 = params(1);
  max1 = params(2);
  min2 = params(3);
  max2 = params(4);
end

inds1=floor((gps(1,:) - min1)/ (max1-min1) * (resolution(1)-1))+1;
inds2=floor((gps(2,:) - min2)/ (max2-min2) * (resolution(2)-1))+1;

%crop to within these subsets
oks = (inds1 >= 1 & inds1 <= resolution(1) & ...
       inds2 >= 1 & inds2 <= resolution(2));

%make into image

imgsum = full(sparse(double(inds1(oks)),double(inds2(oks)),double(values(oks)),...
                  resolution(1),resolution(2)));

imgcount = full(sparse(double(inds1(oks)),double(inds2(oks)),1,...
                  resolution(1),resolution(2)));


img = flipud(imgsum./(imgcount+eps));

mmm = colormap(jet);

% mmm = colormap(gray);
% mstart = [1 0 0];
% mend = [1 1 1];
% mmm = repmat(linspace(0,1,200)',[1 3]);
% mstart = repmat(mstart,[200 1]);
% mend = repmat(mend,[200 1]);
% mmm = mmm.*mend+ (1-mmm).*mstart;
% mmm = mmm(end:-1:1,:);

% %if empty get null image
% if sum(img(:))==0
%   img = reshape(mmm(1,:),[1 1 3]);
%   fprintf(1,'Got Null image\n');
%   return;
% end

img = imfilter(img,fspecial('gaussian',[61 61],9));
img(img<0) = 0;
% if sum(img(:))==0
%  img = reshape(mmm(1,:),[1 1 3]);
%  fprintf(1,'Got Null image\n');
%  return;
% end

%alphas = double(img~=0);
alphas = double(img / max(img(:)));

%img = log(img);
%img(isinf(img))=0;
img = img / max(img(:))*.9999;

img = reshape(mmm(floor(img*size(mmm,1))+1,:),...
              size(img,1),size(img,2),3);


