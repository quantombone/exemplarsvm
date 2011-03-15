function seg2 = faces2colors(seg)
sizer = size(seg);
colors = jet(5);
colors(end+1,:) = 0;
colors = colors(end:-1:1,:);
seg2 = colors(seg(:)+1,:);
seg2 = reshape(seg2,[sizer(1) sizer(2) 3]);

