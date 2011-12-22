function warped = mywarppos(hg_size, I, sbin, bbox)

% warped = warppos(name, model, c, pos)
% Warp positive examples to fit model dimensions.
% Used for training root filters from positive bounding boxes.

pixels = hg_size * sbin;
h = bbox(4) - bbox(2) + 1;
w = bbox(3) - bbox(1) + 1;

cropsize = (hg_size+2) * sbin;

padx = sbin * w / pixels(2);
pady = sbin * h / pixels(1);
x1 = round(bbox(1)-padx);
x2 = round(bbox(3)+padx);
y1 = round(bbox(2)-pady);
y2 = round(bbox(4)+pady);
window = subarray(I, y1, y2, x1, x2, 1);
warped = imresize(window, cropsize(1:2), 'bilinear');
