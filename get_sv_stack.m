function Isv = get_sv_stack(m, K1, K2)
% Create a K1xK2 image which visualizes the detection windows, as
% well as information about the trained exemplar m
% The first shows shows [exemplar image, w+, w-, sum(w.*x,3),
% mean0, mean 1, ... ,mean N]
% Second row first icon starts the top detections, with a coloring
% indicating the set they belong to
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('K1','var')
  K1 = 5;
  K2 = 5;
end

if ~exist('K2','var')
  K2 = K1;
end

K1 = max(K1,5);
K2 = max(K2,5);

%% sort by score
r = m.model.w(:)'*m.model.svxs - m.model.b;
[aa,bb] = sort(r,'descend');
m.model.svbbs = m.model.svbbs(bb, :);
m.model.svxs = m.model.svxs(:, bb);

%NO NMS
%inds = nms_objid(m.model.svids);
%m.model.svids = m.model.svids(inds);
%m.model.nsv = m.model.nsv(:,inds);

svids = m.model.svids;

if exist('m','var')
  hogpic = (HOGpicture(m.model.w));
  hogpic = jettify(hogpic);
end

N = length(svids);
N = min(N,K1*K2);

svids = svids(1:N);
svids = [svids{:}];

if N > 0
  ucurids = unique([svids.curid]);
else
  ucurids = [];
end
svims = cell(N,1);

%PADDER = 100;

VOCinit;
for i = 1:length(ucurids)
  %Ibase = convert_to_I(bg{ucurids(i)});
  curidstring = sprintf('%06d',ucurids(i));
  Ibase = im2double(imread(sprintf(VOCopts.imgpath,curidstring)));
  %I = pad_image(Ibase, PADDER);
  hits = find([svids.curid]==ucurids(i));
  for j = 1:length(hits)

    cb = svids(hits(j)).bb;
    
    d1 = max(0,1 - cb(1));
    d2 = max(0,1 - cb(2));
    d3 = max(0,cb(3) - size(Ibase,2));
    d4 = max(0,cb(4) - size(Ibase,1));
    mypad = max([d1,d2,d3,d4]);
    PADDER = round(mypad)+2;
    I = pad_image(Ibase,PADDER);
    
    bb = round(svids(hits(j)).bb + PADDER);
    svims{hits(j)} = I(bb(2):bb(4),bb(1):bb(3),:);
   
    if svids(hits(j)).flip == 1
      svims{hits(j)} = flip_image(svims{hits(j)});
    end
        
  end
end

VOCinit;
try
  % Ibase = imread(sprintf(VOCopts.imgpath,m.curid));
  % Ibase = im2double(Ibase);
  % PADDER = 100;
  % Ibase = pad_image(Ibase,PADDER);
  % cb = m.model.coarse_box+PADDER;
  % Ibase = Ibase(round(cb(2):cb(4)), round(cb(1):cb(3)),:);
  [aaa,bbb,Ibase] = get_exemplar_icon({m},1);
catch
  Ibase = zeros(m.model.hg_size(1)*10,m.model.hg_size(2)*10,3);
end

newsize = [size(Ibase,1) size(Ibase,2)];
newsize = 100/newsize(1) * newsize;
newsize = round(newsize);
newsize = newsize + 10;

svims = cellfun2(@(x)max(0.0,min(1.0,imresize(x,newsize))),svims);
if ~isfield(m.model.svids{1},'set')
  [negatives,vals,pos,test,indicator] = ...
      find_set_membership(m.model.svids,m.cls);

else
  indicator = cellfun(@(x)x.set,m.model.svids);
  negatives = find(indicator==1);
  vals = find(indicator==2);
  pos = find(indicator==3);
  test = find(indicator==4);
end

svimstack = cat(4,svims{:});
NMS = K2-4;
if NMS == 1
  cuts(1) = size(svimstack,4);
else
  cuts = round(linspace(1,size(svimstack,4),NMS+1));
  cuts = cuts(2:end);
end
%cuts = (1:NMS);


for i = 1:length(cuts)
  mss{i} = mean(svimstack(:,:,:,1:cuts(i)),4);
end

PADSIZE = 5;

for i = 1:numel(svims)
  %% find membership here
  if sum(i == negatives)  %train- red
    svims{i} = pad_image(svims{i},PADSIZE,[1 0 0]);
  elseif sum(i == pos) %trainval+ green
    svims{i} = pad_image(svims{i},PADSIZE,[0 1 0]);
  elseif sum(i == vals) %val- blue
    svims{i} = pad_image(svims{i},PADSIZE,[0 0 1]);
  else %test gray
    svims{i} = pad_image(svims{i},PADSIZE,[.5 .5 .5]);
  end
end

if length(svims)<K1*K2
  svims{K1*K2} = zeros(newsize(1)+PADSIZE*2,...
                       newsize(2)+PADSIZE*2,3);
end

for j = (N+1):(K1*K2)
  svims{j} = zeros(newsize(1)+PADSIZE*2,...
                   newsize(2)+PADSIZE*2,3);
end

mx = mean(m.model.x,2);
raw_picture = HOGpicture(reshape(mx-mean(mx(:)),m.model.hg_size));
pos_picture = HOGpicture(m.model.w);
neg_picture = HOGpicture(-m.model.w);
spatial_picture = sum(m.model.w.*reshape(mean(m.model.x,2), ...
                                         size(m.model.w)),3);
% for q = 1:size(m.model.nsv,2)
%   sp(:,:,q) =  sum(m.model.w.*reshape(m.model.nsv(:,q), ...
%                                       size(m.model.w)),3);
% end  

% for a = 1:size(sp,1)
%   for b = 1:size(sp,2)
%     v = squeeze(sp(a,b,:));
%     spatial_picture(a,b) = (spatial_picture(a,b)-mean(v))./(std(v)+eps);
%   end
% end

spatial_picture = imresize(jettify(spatial_picture),[size(pos_picture,1) ...
                    size(pos_picture,2)],'nearest');

raw_picture = jettify(imresize(raw_picture,newsize,'nearest'));
pos_picture = jettify(imresize(pos_picture,newsize,'nearest'));
neg_picture = jettify(imresize(neg_picture,newsize,'nearest'));
spatial_picture = imresize(spatial_picture,newsize,'nearest');


%first four slots are reserved for the image
NSHIFT = length(mss) + 4;

svims((NSHIFT+1):end) = svims(1:end-NSHIFT);

%ex goes in slot 1
svims{1} = max(0.0,min(1.0,imresize(Ibase,[size(pos_picture,1),size(pos_picture, ...
                                                  2)])));
svims{2} = pos_picture;
svims{3} = neg_picture;
svims{4} = spatial_picture;
svims(4+(1:length(mss))) = mss;

for q = 1:(4+length(mss))
  svims{q} = pad_image(svims{q},PADSIZE,[1 1 1]);
end
%svims{3} = max(0.0,min(1.0,imresize(hogpic,[size(ms,1),size(ms, ...
%                                                  2)])));
%VOCinit;

%looks bad with padding
%svims = cellfun2(@(x)pad_image(x,2,[1 1 1]),svims);

svims = reshape(svims,K1,K2)';
for j = 1:K2
  svrows{j} = cat(2,svims{j,:});
end


Isv = cat(1,svrows{:});
