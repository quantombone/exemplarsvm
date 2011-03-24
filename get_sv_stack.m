function Isv = get_sv_stack(svids, bg)
%Create an image which visualizes the negative support vectors
%Tomasz Malisiewicz (tomasz@cmu.edu)

N = length(svids);

svids = svids(1:N);
ucurids = unique([svids.curid]);
svims = cell(N,1);

PADDER = 100;

for i = 1:length(ucurids)
  I = convert_to_I(bg{ucurids(i)});
  

  I = pad_image(I, PADDER);
  hits = find([svids.curid]==ucurids(i));
  for j = 1:length(hits)
    
    bb = round(svids(hits(j)).bb + PADDER);
    try
      svims{hits(j)} = I(bb(2):bb(4),bb(1):bb(3),:);
    catch
      
      svims{hits(j)} = rand(bb(4)-bb(2)+1,bb(3)-bb(1)+1,3);
    end
   
    if svids(hits(j)).flip == 1
      svims{hits(j)} = flip_image(svims{hits(j)});
    end
  end
end

%resize to average size
newsize = round([mean(cellfun(@(x)size(x,1),svims)) mean(cellfun(@ ...
                                                  (x)size(x,2),svims))]);
svims = cellfun2(@(x)max(0.0,min(1.0,imresize(x,newsize))),svims);

K = ceil(sqrt(N));
if length(svims)<K*K
  svims{K*K} = zeros(newsize(1),newsize(2),3);
end
for j = (N+1):(K*K)
  svims{j} = zeros(newsize(1),newsize(2),3);
end
sstack = cat(4,svims{:});
ms = mean(sstack,4);
svims(2:end) = svims(1:end-1);
svims{1} = ms;

svims = reshape(svims,K,K)';
for j = 1:K
  svrows{j} = cat(2,svims{j,:});
end


Isv = cat(1,svrows{:});

