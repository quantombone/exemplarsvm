function f = features_pedro2(I,sbin);

filter = [1 2 1; 2 4 2; 1 2 1];
filter = filter / sum(filter(:));
I2 = imfilter(I,filter,'symmetric');

f = features_pedro(I2,sbin);
