function [feat, scale] = featpyramid2(im, sbin, interval)
% [feat, scale] = featpyramid2(im, sbin, interval);

sc = 2 ^(1/interval);

im = double(im);

MAXSCALE = 1.0;
MINSCALE = .01;

MAXLEVELS = 200;
MINDIMENSION = 5;

% Start at MAXSCALE, and keep going down by the increment sc, until
% we reach MAXLEVELS or MINSCALE
for i = 1:MAXLEVELS
  scaler = MAXSCALE / sc^(i-1);
  
  if scaler < MINSCALE
    return
  end
  
  scale(i) = scaler;
  scaled = resize(im,scale(i));
  
  %if minimum dimensions is less than or equal to 5, exit
  if min([size(scaled,1) size(scaled,2)])<=MINDIMENSION
    return
  end

  feat{i} = features(scaled,sbin);

  %if we get zero size feature, backtrack one, and dont produce any
  %more levels
  if (size(feat{i},1)*size(feat{i},2)) == 0
    feat = feat(1:end-1);
    scale = scale(1:end-1);
    return;
  end

  %recover lost bin!!!
  feat{i} = padarray(feat{i}, [1 1 0], 0);

  %if the max dimensions is less than or equal to 5, dont produce
  %any more levels
  if max([size(feat{i},1) size(feat{i},2)])<=MINDIMENSION
    return;
  end
  
end
