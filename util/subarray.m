function B = subarray(A, i1, i2, j1, j2, pad)

% % B = subarray(A, i1, i2, j1, j2, pad)
% % Extract subarray from array
% % pad with boundary values if pad = 1
% % pad with zeros if pad = 0

% dim = size(A);
% B = zeros(i2-i1+1, j2-j1+1, dim(3));
% if pad
%   for i = i1:i2
%     for j = j1:j2
%       ii = min(max(i, 1), dim(1));
%       jj = min(max(j, 1), dim(2));
%       B(i-i1+1, j-j1+1, :) = A(ii, jj, :);
%     end
%   end
% else
%   for i = max(i1,1):min(i2,dim(1))
%     for j = max(j1,1):min(j2,dim(2))
%       B(i-i1+1, j-j1+1, :) = A(i, j, :);
%     end
%   end
% end


% B = subarray(A, i1, i2, j1, j2, pad)
% Extract subarray from array
% pad with boundary values if pad = 1
% pad with zeros if pad = 0

if ~exist('pad','var')
  pad = 1;
end

us = i1:i2;
vs = j1:j2;

dim = size(A);
B = zeros(i2-i1+1, j2-j1+1, dim(3));

goodu = find(us>=1 & us<=size(A,1));
goodv = find(vs>=1 & vs<=size(A,2));
B((goodu(1)):(goodu(end)),...
  goodv(1):goodv(end),:) = A(us(goodu(1)):us(goodu(end)), ...
                             vs(goodv(1)):vs(goodv(end)),:);

if pad == 0
  return
end

L = (goodu(1)-1);
B(1:L,goodv(1):goodv(end),:) = ...
    repmat(A(us((goodu(1))),...
             vs(goodv(1)):vs(goodv(end)),:),L,1);

L = (size(B,1)-goodu(end));
B(end-L+1:end,goodv(1):goodv(end),:) = ...
    repmat(A(us((goodu(end))), ...
             vs(goodv(1)):vs(goodv(end)),:),L,1);

L = (goodv(1)-1);
B(:,1:L,:) = repmat(B(:,goodv(1),:),1,L);

L = size(B,2)-(goodv(end));
B(:,size(B,2)-L+1:end,:) = repmat(B(:,goodv(end),:),1,L);

