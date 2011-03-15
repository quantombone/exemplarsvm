function I = convert_to_I(I)
%Get pixel representation of image because the input I can actually
%be one of: real image, string, function.
%Output will be a [M x N x 3] image matrix

%if we have a string, then it is a path
if isstr(I) 
  if I(end)~=')'
    I = im2double(imread(I));
  else
    I = eval(I);
  end
  %if we have a function, then call it with default arguments
elseif isa(I,'function_handle')
  I = I();
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
