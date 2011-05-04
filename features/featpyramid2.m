function [feat, scale] = featpyramid2(im, sbin, interval)

% [feat, scale] = featpyramid(im, sbin, interval);
% Compute feature pyramid.
%
% sbin is the size of a HOG cell - it should be even.
% interval is the number of scales in an octave of the pyramid.
% feat{i} is the i-th level of the feature pyramid.
% scale{i} is the scaling factor used for the i-th level.
% feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

%% here we pad data to always be a multiple of the bin

sc = 2 ^(1/interval);
imsize = [size(im, 1) size(im, 2)];
%max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));
%feat = cell(max_scale + interval, 1);
%scale = zeros(max_scale + interval, 1);
%pads = cell(max_scale + interval, 1);
% our resize function wants floating point values
im = double(im);

DOD = 1;
for i = 1:200
  scaler = 1/sc^(i-1);

  if scaler < .01
    return
  end
  
  scale(i) = scaler;

  
  if (DOD && (scale(i) > 1))
    scaled = resize(im,scale(i)/2);
  else
    scaled = resize(im,scale(i));
  end
  
  
  
  if min([size(scaled,1) size(scaled,2)])<=5
    return
  end
  if (DOD && (scale(i) > 1))
    feat{i} = features(scaled,sbin/2);
  else
    feat{i} = features(scaled,sbin);
  end

  %if we get zero size feature, backtrack one
  if (size(feat{i},1)*size(feat{i},2)) == 0
    feat = feat(1:end-1);
    scale = scale(1:end-1);
    return;
  end

  %recover lost bin!!!
  feat{i} = padarray(feat{i}, [1 1 0], 0);
  
  if max([size(feat{i},1) size(feat{i},2)])<=5
    return;
  end
  
end

% function [im2,pads] = scale_before_resize(im,sbin,scale)
% %doesnt help really
% offer = ceil([size(im,1) size(im,2)] * (scale/sbin)) / (scale/ ...
%                                                   sbin);

% pads = offer - [size(im,1) size(im,2)];

% pads = round(pads);
% im2 = im;
% im2(end+1:end+pads(1),end+1:end+pads(2),:)=0;

% im2 = resize(im2,scale);
