function I = toI(I)
%Get pixel representation of image because the input I can actually
%be one of: real image, string, function.
%Output will be a [M x N x 3] image matrix

%NOTE: convert_to_I should always load image
% if iscell(I)
%   %if we are given a cell array
  
%   return;
% end

%if we have a string, then it is a path
if isstr(I) 
  
  %if string ends in ')', then it is a function call
  if I(end)~=')'
    if (length(I)>=7 && strcmp(I(1:7),'http://'))
      fprintf(1,'Warning: loading image from URL\n');
    end
    try
      I = imread(I);
    catch
      fprintf(1,'Cannot load image: %s\n',I);
      I = zeros(0,0,3);
    end
  else
    I = eval(I);
  end
  %if we have a function, then call it with default arguments
elseif isa(I,'function_handle')
  I = I();
elseif isnumeric(I)
  %If we get here, then we have a numeric representation, so it is
  %an image
elseif isstruct(I) && isfield(I,'I')
  I = convert_to_I(I.I);
elseif iscell(I)
  try
    I = convert_to_I(I{1});
  catch
    error('convert_to_I: invalid input');
  end
else
  fprintf(1,['WARNING: convert_to_I given non-image type as input\' ...
             'n']);
  error('convert_to_I: invalid input');
end

%now we have a real image
I = im2double(I);

%Do not alow single channel images!!!
if size(I,3) == 1
  I = repmat(I, [1 1 3]);
end

if 0
  MAXSIZE = 300;
  if max(size(I))>MAXSIZE
    factor = MAXSIZE/max(size(I));
    I = max(0.0,min(1.0,imresize(I,factor)));
  end
end
