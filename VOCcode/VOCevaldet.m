function [result] = VOCevaldet(test_set,BB,cls,params)
% Evaluates the set of bounding boxes on the dataset
% called test_set with the PASCAL VOC criterion.
%
if ~exist('params','var')
  params.evaluation_minoverlap = 0.5;
  params.display = 1;
end

fprintf('%s: pr: evaluating detections\n',cls);
       
% extract ground truth objects
npos=0;
gt(length(test_set))=struct('BB',[],'diff',[],'det',[],'objectid',[],'imageid',[]);
for i=1:length(test_set)
    % extract objects of class
    if numel(test_set{i})==0
      continue
    end

    if ~isfield(test_set{i},'objects') || ...
          length(test_set{i}.objects)==0
      %dont do anything if no objects, no gt!
    else
      clsinds=strmatch(cls,{test_set{i}.objects(:).class},'exact');
      gt(i).BB=cat(1,test_set{i}.objects(clsinds).bbox)';
      gt(i).diff=[test_set{i}.objects(clsinds).difficult];% | [test_set{i}.objects(clsinds).occluded];
      gt(i).det=false(length(clsinds),1);
      gt(i).imageid = gt(i).diff*0+i;
      gt(i).objectid = 1:length(gt(i).imageid);
      %skip difficult ones in evaluation
      npos=npos+sum(~gt(i).diff);
    end

end
npos
sss = tic;
[ap, rec, prec, fp, tp, is_correct, missed] = get_aps(params,cls,gt,npos,BB);

result.rec = rec;
result.prec = prec;
result.ap = ap;
result.fp = fp;
result.tp=tp;
result.npos=npos;
result.is_correct = is_correct;
result.missed = missed;

finaltime = toc(sss);
fprintf(1,'Time for computing AP: %.3fsec\n',finaltime);

function [ap,rec,prec,fp,tp,is_correct,missed] = get_aps(params,cls,gt,npos,BB);

[~,order] = sort(BB(:,end),'descend');
BB = BB(order,:);
confidence = BB(:,end);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);

tic;

is_correct = zeros(nd,1);
for d=1:nd

    % assign detection to ground truth object if any
    bb = BB(d,1:4);
    i = BB(d,11);
    ovmax=-inf;

    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end

    % assign detection as true positive/don't care/false positive
    if ovmax>=params.evaluation_minoverlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
		gt(i).det(jmax)=true;
                is_correct(d) = 1;
            else
                is_correct(d) = -1;
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

% compute average precision the old way
% ap=0;
% for t=0:0.1:1
%     p=max(prec(rec>=t));
%     if isempty(p)
%         p=0;
%     end
%     ap=ap+p/11;
% end
% apold = ap;
ap = VOCap(rec,prec);

if params.display
    % plot precision/recall
    figure(1)
    clf
    plot(rec,prec,'-','LineWidth',2);
    hold on;
    plot(rec,prec,'o');
    grid on;
    xlabel('recall','FontSize',20)
    ylabel('precision','FontSize',20)
    title(sprintf('class: %s, AP = %.3f, OS=%.3f',...
                  cls, ap, params.evaluation_minoverlap),...
          'FontSize',20);
    maxrec = max(rec)+.1;
    axis([0 1 0 1])
end

dets = cat(1,gt.det);
objs = cat(2,gt.objectid);
ims = cat(2,gt.imageid);
missed.dets = dets;
missed.objs = objs;
missed.ims = ims;
bads = find(missed.dets==0);
allbb=cat(2,gt.BB)';
missed.bbs = allbb(bads,:);
missed.dets = missed.dets(bads);
missed.objs = missed.objs(bads);
missed.ims = missed.ims(bads);


%save resdata.mat ims objs dets 
%keyboard
