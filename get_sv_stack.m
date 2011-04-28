function Isv = get_sv_stack(svids, bg, m, K1, K2)
%Create an image which visualizes the negative support vectors
%Tomasz Malisiewicz (tomasz@cmu.edu)

if iscell(svids)
  svids = [svids{:}];
end

if exist('m','var')
  hogpic = (HOGpicture(m.model.w));
  
  NC = 200;
  colorsheet = jet(NC);
  dists = hogpic(:);    
  dists = dists - min(dists);
  dists = dists / (max(dists)+eps);
  dists = round(dists*(NC-1)+1);
  colors = colorsheet(dists,:);
  hogpic = reshape(colors,[size(hogpic,1) size(hogpic,2) 3]);
end


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
% newsize = ([mean(cellfun(@(x)size(x,1),svims)) mean(cellfun(@ ...
%                                                   (x)size(x,2), ...
%                                                   svims))]);
% newsize = 100/newsize(1) * newsize;
% newsize = round(newsize);
VOCinit;
Ibase = imread(sprintf(VOCopts.imgpath,m.curid));
Ibase = im2double(Ibase);
Ibase = Ibase(m.gt_box(2):m.gt_box(4),m.gt_box(1):m.gt_box(3),:);
newsize = [size(Ibase,1) size(Ibase,2)];

newsize = 100/newsize(1) * newsize;
newsize = round(newsize);


svims = cellfun2(@(x)max(0.0,min(1.0,imresize(x,newsize))),svims);

%K = ceil(sqrt(N));
%K2 = 1;
%K1 = 12;

%K1 = K;
%K2 = K;
if length(svims)<K1*K2
  svims{K1*K2} = zeros(newsize(1),newsize(2),3);
end

for j = (N+1):(K1*K2)
  svims{j} = zeros(newsize(1),newsize(2),3);
end
sstack = cat(4,svims{:});

%% compute the mean image
ms = mean(sstack,4);
svims(4:end) = svims(1:end-3);
svims{2} = ms;
svims{3} = max(0.0,min(1.0,imresize(hogpic,[size(ms,1),size(ms, ...
                                                  2)])));
%VOCinit;

svims{1} = max(0.0,min(1.0,imresize(Ibase,[size(ms,1),size(ms,2)])));
 
svims = reshape(svims,K1,K2)';
for j = 1:K2
  svrows{j} = cat(2,svims{j,:});
end

Isv = cat(1,svrows{:});

