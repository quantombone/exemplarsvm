function geom_eval
%evaluate geometry and segmentation

basedir = '/nfs/baikal/tmalisie/iccvres/finalbusmats/';
files = dir([basedir '*mat']);
rows = cell(0,1);
my_score = zeros(0,1);
derek_score = zeros(0,1);
seg_score = zeros(0,1);
seg_score_box = zeros(0,1);
dumb_score = zeros(0,1);
os_score = zeros(0,1);

myc = zeros(2,2);
boxc = zeros(2,2);

counts_me = zeros(10,10,1);
counts_derek = zeros(10,10,1);
total = 0;
pixels_fg = [];
for i = 1:length(files)
  i
  res = load([basedir files(i).name]);  
  if size(res.output.WARNINGgt,3) == 1
    res.output.WARNINGgt = repmat(res.output.WARNINGgt,[1 1 3]);
  end
    
  gtbg = double(sum(res.output.WARNINGgt,3)==0);
  
  newresult = res.output.dresult;
  if size(newresult,1) > 0
    
    newresult(res.output.dresult==1)=2;
    newresult(res.output.dresult==2)=1;
  else
    newresult = ones(size(res.output.result,1),size(res.output.result, ...
                     2));
  end


  res.output.dresult = newresult;
  res.output.dresult(find(gtbg))=0;
  res.output.dresult = faces2colors(res.output.dresult);

  colors = jet(5);
  colors(end+1,:) = 0;
  
  [aa,bb] = ismember(reshape(res.output.result,[],3),colors, ...
                     'rows');

  myseg = reshape(bb,[size(res.output.result,1) ...
                      size(res.output.result,2)]);
  
  [aa,bb] = ismember(reshape(res.output.dresult,[],3),colors,'rows');
  dseg = reshape(bb,[size(res.output.dresult,1) ...
                      size(res.output.dresult,2)]);
  
  [aa,bb] = ismember(reshape(res.output.WARNINGgt,[],3),colors, ...
                     'rows');

  gtseg = reshape(bb,[size(res.output.dresult,1) ...
                      size(res.output.dresult,2)]);
  
  fg = logical(1-gtbg);

  unique(myseg(:))
  unique(gtseg(:))
  counts_me = counts_me + sparse(myseg(fg),gtseg(fg),1,10,10);
  counts_derek = counts_derek + sparse(dseg(fg),gtseg(fg),1,10,10);
  
  counts_me(3:5,3:5)
  
  fgmat = cat(2,1*fg,2*fg,3*fg,4*fg);
  
  imagesc(1)
  
  superI = cat(2,res.output.I,res.output.result,res.output.dresult, ...
               res.output.WARNINGgt);
  imagesc(superI)
  axis image
  axis off
  drawnow
    
  diff_derek = sum(abs(res.output.dresult - res.output.WARNINGgt), ...
                   3);
  diff_derek = diff_derek > 0;
  diff_me = sum(abs(res.output.result - res.output.WARNINGgt),3);
  diff_me = diff_me > 0;
  
  my_score(end+1) = 100*(1-mean(diff_me(find(fg))));
  derek_score(end+1) = 100*(1-mean(diff_derek(find(fg))));
  
  myfg = (sum(res.output.result,3)~=0);
  gtfg = (sum(res.output.WARNINGgt,3)~=0);

  myfg2 = myfg;
  [u,v] = find(myfg2);
  myfg2(min(u):max(u),min(v):max(v))=1;
  seg_score(end+1) = sum(myfg(:)==gtfg(:));
  seg_score_box(end+1) = sum(myfg2(:)==gtfg(:));
  dumb_score(end+1) = sum(gtfg(:)==0);
  os_score(end+1) = getosmatrix(double(myfg(:)),double(gtfg(:)));
  total = total + length(myfg(:));
  
  
  myc = myc + sparse(myfg(:)+1,gtfg(:)+1,1,2,2);
  boxc = boxc + sparse(myfg2(:)+1,gtfg(:)+1,1,2,2);
  
  %title(sprintf('Us = %.2f, Hoiem = %.2f',my_score,derek_score))
  %figure(1)
  %rows{end+1} = superI;
end

keyboard