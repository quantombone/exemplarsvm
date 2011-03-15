function I2 = circshift2(I,shifter)

I2 = circshift(I,shifter);

if shifter(1) > 0
  I2(1:min(size(I,1),shifter(1)),:)=0;
elseif shifter(1) < 0
  laster = size(I2,1);
  I2(1+max(1,laster+shifter(1)):laster,:) = 0;
else  
end


if shifter(2) > 0
  I2(:,1:min(size(I,2),shifter(2)),:)=0;
elseif shifter(2) < 0
  laster = size(I2,2);
  I2(:,1+max(1,laster+shifter(2)):(laster),:) = 0;
else
end


